from __future__ import annotations

import json
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from nyaya_agent.graph import build_graph
from nyaya_agent.memory import ChatMemoryStore
from nyaya_agent.seed_chroma import seed_demo_corpus
from nyaya_agent.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHROMA_READY, SQLITE_PATH


def _session_id() -> str:
    if "nyaya_session_id" not in st.session_state:
        st.session_state.nyaya_session_id = str(uuid.uuid4())
    return st.session_state.nyaya_session_id


def main() -> None:
    st.set_page_config(page_title="Nyaya Agent", layout="wide")
    st.title("Nyaya Agent")
    st.caption("Case-focused chat · ChromaDB when enabled · SQLite memory (six messages + summary)")

    with st.sidebar:
        st.subheader("Status")
        st.write(f"**CHROMA_READY:** `{CHROMA_READY}`")
        st.write(f"**Chroma path:** `{CHROMA_PERSIST_DIR}`")
        st.write(f"**Collection:** `{CHROMA_COLLECTION}`")
        st.write(f"**SQLite:** `{SQLITE_PATH}`")
        if not CHROMA_READY:
            st.info("Retrieval is off. Set `CHROMA_READY=true` in `.env` after ingesting data.")
        if st.button("Seed demo Chroma documents (illustrative only)"):
            n = seed_demo_corpus()
            st.success(f"Upserted {n} demo chunks. Set `CHROMA_READY=true` and restart the app to query them.")

    store = ChatMemoryStore(SQLITE_PATH)
    sid = _session_id()
    ctx = store.get_context(sid)

    st.subheader("Chat")
    for m in ctx.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if ctx.summary:
        with st.expander("Rolling summary of older turns"):
            st.markdown(ctx.summary)

    app = build_graph()
    if prompt := st.chat_input("Ask about a case or compliance topic…"):
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
            if CHROMA_READY and out.get("memo"):
                with st.expander("Structured memo (JSON)"):
                    st.json(out["memo"])

        store.append_exchange(sid, prompt, reply)
        st.rerun()


if __name__ == "__main__":
    main()
