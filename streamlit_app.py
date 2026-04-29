from __future__ import annotations

import json
import uuid
import warnings
from io import BytesIO

import markdown
from xhtml2pdf import pisa

warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from nyaya_agent.graph import build_graph
from nyaya_agent.memory import ChatMemoryStore
from nyaya_agent.evaluate_rag import evaluate_retrieval
from nyaya_agent.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHROMA_READY, SQLITE_PATH, setup_environment

setup_environment()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)




def generate_pdf_from_memo(memo: dict) -> bytes:
    md_content = memo.get("detailed_report", "")
    if not md_content:
        md_content = f"# Nyaya Agent Memo\n\n**Query:** {memo.get('query')}\n\n**Summary:**\n{memo.get('summary', 'No summary available.')}\n\n*Detailed report is missing.*"
        
    html_content = markdown.markdown(md_content)
    
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: a4 portrait; margin: 2cm; }}
            body {{ font-family: Helvetica, Arial, sans-serif; font-size: 12pt; line-height: 1.5; color: #333; }}
            h1 {{ color: #2C3E50; border-bottom: 1px solid #eee; padding-bottom: 5px; font-size: 24pt; }}
            h2 {{ color: #34495E; margin-top: 20px; font-size: 18pt; }}
            h3 {{ color: #7F8C8D; font-size: 14pt; }}
            p {{ margin-bottom: 10px; }}
            ul {{ margin-bottom: 15px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(src=full_html, dest=pdf_buffer)
    
    if pisa_status.err:
        logger.error("Failed to generate PDF")
        return b""
        
    return pdf_buffer.getvalue()


def _get_session_id() -> str:
    """Read session ID from the URL on every run. URL is the single source of truth.

    If no ?sid= in the URL (first-time visitor), generate one and write it
    via st.query_params. This triggers a rerun, but that's fine — we no longer
    depend on session_state for sid or chat context.
    """
    url_sid = st.query_params.get("sid")
    if url_sid:
        return url_sid

    # First visit — generate sid and persist to URL.
    new_sid = str(uuid.uuid4())
    logger.info(f"New session ID generated: {new_sid}")
    st.query_params["sid"] = new_sid
    return new_sid


@st.cache_resource
def get_memory_store():
    return ChatMemoryStore(SQLITE_PATH)

def main() -> None:
    if "app_started" not in st.session_state:
        logger.info("Starting Nyaya Agent Streamlit App")
        st.session_state["app_started"] = True
        
    st.set_page_config(page_title="Nyaya Agent", layout="wide")
    st.title("Nyaya Agent")
    st.caption("Case-focused chat · ChromaDB when enabled · SQLite memory (six messages + summary)")

    sid = _get_session_id()

    store = get_memory_store()
    ctx = store.get_context(sid)

    with st.sidebar:
        st.subheader("Status")
        st.write(f"**CHROMA_READY:** `{CHROMA_READY}`")
        st.write(f"**Chroma path:** `{CHROMA_PERSIST_DIR}`")
        st.write(f"**Collection:** `{CHROMA_COLLECTION}`")
        st.write(f"**SQLite:** `{SQLITE_PATH}`")
        if not CHROMA_READY:
            st.info("Retrieval is off. Set `CHROMA_READY=true` in `.env` after ingesting data.")
        if st.button("Evaluate RAG Pipeline"):
            with st.spinner("Evaluating RAG pipeline... This may take a minute."):
                try:
                    rating = evaluate_retrieval()
                    full_stars = int(rating)
                    half = (rating - full_stars) >= 0.5
                    stars = "★" * full_stars + ("½" if half else "") + "☆" * (5 - full_stars - (1 if half else 0))
                    st.success(f"RAG Quality: {stars}  **{rating} / 5**")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

        st.divider()
        st.subheader("Memo Actions")
        latest_memo = st.session_state.get("latest_memo")
        if "show_memo" not in st.session_state:
            st.session_state.show_memo = False

        def toggle_memo():
            st.session_state.show_memo = not st.session_state.show_memo

        memo_btn_label = "📄 Hide Latest Memo" if st.session_state.show_memo else "📄 View Latest Memo"
        st.button(memo_btn_label, disabled=not bool(latest_memo), use_container_width=True, on_click=toggle_memo)

        if st.session_state.show_memo and latest_memo:
            def log_download():
                logger.info("User initiated download of the structured memo PDF")

            pdf_bytes = generate_pdf_from_memo(latest_memo)

            if pdf_bytes:
                st.download_button(
                    label="📥 Download PDF Memo",
                    data=pdf_bytes,
                    file_name="nyaya_memo.pdf",
                    mime="application/pdf",
                    on_click=log_download,
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF.")
                
            with st.expander("Memo Contents", expanded=True):
                st.markdown(latest_memo.get("detailed_report", "No detailed report available."))

        st.divider()
        st.subheader("Context Summary")
        if ctx.summary:
            with st.expander("Summary Contents", expanded=False):
                st.markdown(ctx.summary)

    st.subheader("Chat")
    for m in ctx.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    @st.cache_resource
    def get_cached_graph():
        return build_graph()

    app = get_cached_graph()

    if prompt := st.chat_input("Ask about a case or compliance topic…"):
        logger.info(f"User entered prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                out = app.invoke(
                    {
                        "query": prompt,
                        "conversation_summary": ctx.summary,
                        "recent_messages": ctx.messages,
                    }
                )
            reply = (out.get("assistant_message") or out.get("chat_reply") or "").strip()
            if not reply:
                reply = json.dumps(out.get("memo") or {}, indent=2, ensure_ascii=False)[:8000]
            st.markdown(reply)
            
            if out.get("memo"):
                logger.info("New memo successfully generated by the graph")
                st.session_state.latest_memo = out["memo"]

        logger.info("Persisting exchange to SQLite")
        store.append_exchange(sid, prompt, reply)
        st.rerun()


if __name__ == "__main__":
    main()
