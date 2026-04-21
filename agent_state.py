from __future__ import annotations

from typing import Literal, TypedDict


SourceType = Literal["case_law", "circular", "legislation", "unknown"]


class RetrievedDoc(TypedDict):
    id: str
    source_type: SourceType
    title: str
    citation: str
    url: str | None
    text: str


class ComplianceFinding(TypedDict):
    requirement: str
    gap: str
    risk_rating: Literal["low", "medium", "high"]
    remediation: str
    citations: list[str]


class NyayaState(TypedDict, total=False):
    # Input
    query: str
    # SQLite-backed chat context (injected by UI / CLI before invoke)
    conversation_summary: str
    recent_messages: list[dict]  # {"role": "user"|"assistant", "content": str}
    # Research output
    retrieved: list[RetrievedDoc]
    # Compliance output
    findings: list[ComplianceFinding]
    # Final memo output (JSON-serializable)
    memo: dict
    # Plain-chat path (when Chroma is not ready)
    chat_reply: str
    # Short natural-language reply for chat UIs (RAG or plain)
    assistant_message: str
