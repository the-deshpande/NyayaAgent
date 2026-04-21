from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent_state import NyayaState, RetrievedDoc
from nyaya_agent.llm import get_chat_model
from nyaya_agent.retrieval import Retriever
from nyaya_agent.settings import MAX_REACT_ITERATIONS


def _tool_args(tc: dict[str, Any]) -> dict[str, Any]:
    raw = tc.get("args") or tc.get("arguments") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def research_agent(
    state: NyayaState,
    *,
    retriever: Retriever | None = None,
    max_iterations: int | None = None,
) -> NyayaState:
    """Research agent: ReAct-style loop (max six tool rounds by default) over `search_legal_corpus`."""

    query = (state.get("query") or "").strip()
    r = retriever or Retriever()
    cap = max_iterations if max_iterations is not None else MAX_REACT_ITERATIONS

    if not query:
        return {"retrieved": []}

    @tool
    def search_legal_corpus(search_query: str) -> str:
        """Search the indexed Indian legal corpus (Kanoon, SEBI circulars, Gazette) in ChromaDB."""
        docs = r.search(search_query)
        if not docs:
            return "No documents retrieved. Try alternate keywords or verify ingestion."
        parts: list[str] = []
        for d in docs:
            parts.append(f"[{d.get('citation', 'N/A')}] {d.get('title', '')}\n{d.get('text', '')[:2000]}")
        return "\n\n---\n\n".join(parts)

    tools = [search_legal_corpus]
    model = get_chat_model().bind_tools(tools)

    merged: dict[str, RetrievedDoc] = {}

    def run_search(q: str) -> str:
        docs = r.search(q)
        for doc in docs:
            merged[doc["id"]] = doc
        if not docs:
            return "No documents retrieved. Try alternate keywords or verify ingestion."
        parts: list[str] = []
        for d in docs:
            parts.append(f"[{d.get('citation', 'N/A')}] {d.get('title', '')}\n{d.get('text', '')[:2000]}")
        return "\n\n---\n\n".join(parts)

    msgs: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
        SystemMessage(
            content=(
                "You are the Research agent for Nyaya. Use `search_legal_corpus` to retrieve sources. "
                "You may issue multiple searches (refine queries) within the tool budget. "
                "When satisfied, stop calling tools."
            )
        ),
        HumanMessage(content=f"User query:\n{query}"),
    ]

    for _ in range(cap):
        ai: AIMessage = model.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            name = tc.get("name")
            tid = tc.get("id") or ""
            args = _tool_args(tc)
            if name == "search_legal_corpus":
                sq = (args.get("search_query") or args.get("query") or query).strip()
                obs = run_search(sq)
            else:
                obs = f"Unknown tool: {name}"
            msgs.append(ToolMessage(content=obs, tool_call_id=tid))

    return {"retrieved": list(merged.values()) if merged else r.search(query)}
