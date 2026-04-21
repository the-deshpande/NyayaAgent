from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent_state import NyayaState
from nyaya_agent.llm import get_chat_model


def plain_chat_node(state: NyayaState) -> NyayaState:
    """General chat when ChromaDB is not ready (`CHROMA_READY=false`)."""

    query = (state.get("query") or "").strip()
    summary = (state.get("conversation_summary") or "").strip()
    recent = state.get("recent_messages") or []

    blocks: list[str] = []
    if summary:
        blocks.append(f"Rolling summary of earlier conversation:\n{summary}")
    for m in recent:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            blocks.append(f"{role}: {content}")
    blocks.append(f"user: {query}")
    human = HumanMessage(content="\n\n".join(blocks) if blocks else query)

    sys = SystemMessage(
        content=(
            "You are Nyaya Agent (India-focused legal research assistant). "
            "The knowledge-base retrieval layer is OFF for this session — do not claim you read specific "
            "case law, circulars, or gazette files unless the user pasted them. "
            "Give structured general guidance and ask clarifying questions. "
            "Be concise."
        )
    )

    try:
        model = get_chat_model()
        out = model.invoke([sys, human])
        text = (out.content or "").strip()
    except Exception as e:
        text = (
            "The chat model could not complete this request (network, quota, or credentials). "
            f"Details: {e!s}"
        )
    memo = {
        "mode": "plain_chat",
        "reply": text,
        "retrieval_enabled": False,
    }
    return {
        "chat_reply": text,
        "assistant_message": text,
        "memo": memo,
        "retrieved": [],
        "findings": [],
    }
