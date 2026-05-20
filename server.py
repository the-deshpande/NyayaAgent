from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import os
import pypdf
import traceback
import json

# LangGraph and Memory Integration
from nyaya_agent.graph import build_graph
from nyaya_agent.memory.sqlite_store import ChatMemoryStore
from nyaya_agent.settings import SQLITE_PATH, setup_environment
from nyaya_agent.evaluate_rag import evaluate_retrieval

# Setup env vars for Transformers/Tokenizers if needed
setup_environment()

app = FastAPI(title="Nyaya Agent - Merged Architecture")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    persona: str = "explorer"
    session_id: str = "default_session"

# Initialize Graph and Memory Store
graph_app = build_graph()
store = ChatMemoryStore(SQLITE_PATH)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Get memory context
        ctx = store.get_context(request.session_id)
        
        # Inject persona into the prompt (a simple way to guide the agent)
        augmented_prompt = f"[User Persona: {request.persona}]\n{request.message}"

        # Invoke the LangGraph workflow
        out = graph_app.invoke(
            {
                "query": augmented_prompt,
                "conversation_summary": ctx.summary,
                "recent_messages": ctx.messages,
            }
        )
        
        # Extract response
        reply = (out.get("assistant_message") or out.get("chat_reply") or "").strip()
        memo = out.get("memo")
        
        if not reply and memo:
            # If no plain reply, but we have a memo, return a formatted version
            reply = f"**I have prepared a memo based on your request.**\n\n{memo.get('summary', '')}"

        # Persist exchange to SQLite
        store.append_exchange(request.session_id, augmented_prompt, reply)
        store.prune_sessions(6)
        
        # Get updated context to return the latest summary
        updated_ctx = store.get_context(request.session_id)

        return {
            "response": reply,
            "memo": memo,
            "summary": updated_ctx.summary
        }
    except Exception as e:
        print(f"ERROR IN CHAT: {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = ""
        if file.filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            text = content.decode("utf-8")
            
        # Ingest into ChromaDB for RAG!
        from nyaya_agent.retrieval import get_retriever, ChromaRetriever
        retriever = get_retriever()
        if isinstance(retriever, ChromaRetriever):
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            
            # Prepare for Chroma
            docs = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append({"title": file.filename, "source": file.filename})
                ids.append(f"{file.filename}-{i}-{str(uuid.uuid4())[:8]}")
                
            if docs:
                # Explicitly generate embeddings using InLegalBERT (dim 768)
                embeddings = retriever._embedding_model.encode(docs, normalize_embeddings=True).tolist()
                
                retriever._collection.add(
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            
        return {"filename": file.filename, "text": text}
    except Exception as e:
        print(f"UPLOAD ERROR: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/api/status")
async def status():
    return {"status": "Satyameva Jayate - Online (LangGraph Edition)"}

@app.get("/api/sessions")
async def get_sessions():
    try:
        return {"sessions": store.get_recent_sessions(6)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    try:
        ctx = store.get_context(session_id)
        return {
            "summary": ctx.summary,
            "messages": ctx.messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    try:
        store.delete_session(session_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate_rag")
async def evaluate_rag_endpoint():
    try:
        rating = evaluate_retrieval()
        return {"rating": rating}
    except Exception as e:
        print(f"ERROR IN EVALUATE RAG: {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# MOUNT STATIC FILES LAST
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face uses 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, loop="asyncio")
