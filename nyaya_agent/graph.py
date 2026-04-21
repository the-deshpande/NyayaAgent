from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_state import NyayaState
from nyaya_agent.agents import compliance_agent, research_agent, synthesis_agent
from nyaya_agent.nodes.plain_chat import plain_chat_node
from nyaya_agent.retrieval import Retriever, get_retriever
from nyaya_agent.settings import CHROMA_READY, MAX_REACT_ITERATIONS


def _entry_route(_: NyayaState) -> str:
    return "rag" if CHROMA_READY else "plain"


def build_graph(*, retriever: Retriever | None = None):
    """Three-agent RAG graph when `CHROMA_READY` is true; otherwise a single plain-chat node."""

    r = retriever if retriever is not None else get_retriever()

    g = StateGraph(NyayaState)

    g.add_node(
        "research",
        lambda s: research_agent(s, retriever=r, max_iterations=MAX_REACT_ITERATIONS),
    )
    g.add_node("compliance", compliance_agent)
    g.add_node("synthesis", synthesis_agent)
    g.add_node("plain_chat", plain_chat_node)

    g.add_conditional_edges(START, _entry_route, {"rag": "research", "plain": "plain_chat"})

    g.add_edge("research", "compliance")
    g.add_edge("compliance", "synthesis")
    g.add_edge("synthesis", END)
    g.add_edge("plain_chat", END)

    return g.compile()


def default_retriever() -> Retriever:
    """Backward-compatible name for scripts; delegates to `get_retriever`."""

    return get_retriever()
