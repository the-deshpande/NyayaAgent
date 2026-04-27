from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from agent_state import NyayaState
from nyaya_agent.llm import get_chat_model

logger = logging.getLogger(__name__)


def synthesis_agent(state: NyayaState) -> NyayaState:
    """Synthesis Agent: produce the structured legal memo JSON.

    Proposal mapping: "Converts research and compliance outputs into a structured JSON legal memo."
    """

    query = (state.get("query") or "").strip()
    retrieved = state.get("retrieved") or []
    findings = state.get("findings") or []

    logger.info("Synthesis agent started. Generating structured memo.")

    highest_risk = _highest_risk(findings)

    if highest_risk == "high":
        logger.info("Highest risk is high; no memo generated, bypassing LLM.")
        return {
            "memo": {},
            "assistant_message": "There is not enough information to provide an accurate response."
        }

    # Generate detailed markdown report for PDF conversion
    markdown_lines = [
        f"# Legal Compliance Memo",
        f"**Generated At:** {datetime.now(timezone.utc).isoformat()}",
        f"**Query:** {query}",
        "",
        "## Executive Summary",
        f"**Total Findings:** {len(findings)}",
        f"**Highest Risk Level:** {highest_risk.upper()}",
        "",
    ]
    
    if findings:
        markdown_lines.append("## Compliance Findings")
        for idx, f in enumerate(findings, 1):
            markdown_lines.extend([
                f"### Finding {idx}: {f.get('requirement', 'Unknown Requirement')}",
                f"**Risk Rating:** {f.get('risk_rating', 'medium').upper()}",
                "",
                f"**Gap/Issue:** {f.get('gap', '')}",
                "",
                f"**Remediation:** {f.get('remediation', '')}",
                "",
                "**Citations:**",
            ])
            for c in f.get('citations', []):
                markdown_lines.append(f"- {c}")
            markdown_lines.append("")
    
    if retrieved:
        markdown_lines.append("## References & Sources")
        for idx, doc in enumerate(retrieved, 1):
            title = doc.get("title", "Unknown")
            citation = doc.get("citation", "N/A")
            markdown_lines.append(f"{idx}. **{title}** [{citation}]")
            
    markdown_report = "\n".join(markdown_lines)

    memo = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system": "nyaya-agent",
        },
        "query": query,
        "citations": [
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "citation": d.get("citation"),
                "url": d.get("url"),
                "source_type": d.get("source_type"),
            }
            for d in retrieved
        ],
        "risk_summary": {
            "highest_risk": highest_risk,
            "finding_count": len(findings),
        },
        "compliance_findings": findings,
        "detailed_report": markdown_report,
    }

    assistant_message = _memo_to_assistant_blurb(memo)
    memo["summary"] = assistant_message
    
    logger.info("Synthesis agent finished memo generation.")
    return {"memo": memo, "assistant_message": assistant_message}

def _memo_to_assistant_blurb(memo: dict) -> str:
    """One short natural-language turn for chat UIs (optional LLM polish)."""

    try:
        logger.info("Generating natural language assistant summary for memo")
        model = get_chat_model()
        sys = SystemMessage(
            content="Summarize the following legal compliance memo. Provide a concise, professional executive summary of the findings and risks. Do not output JSON."
        )
        human = HumanMessage(content=str(memo)[:12000])
        out = model.invoke([sys, human])
        t = (out.content or "").strip()
        if t:
            logger.info("Successfully generated assistant blurb")
            return t
    except Exception as e:
        logger.warning(f"Failed to generate assistant blurb: {e}")
        pass

    rs = memo.get("risk_summary") or {}
    n = rs.get("finding_count", 0)
    hr = rs.get("highest_risk", "low")
    return f"Memo ready: {n} compliance finding(s); highest risk level: {hr}. See structured memo for citations."


def _highest_risk(findings: list[dict]) -> str:
    order = {"low": 1, "medium": 2, "high": 3}
    best = "low"
    for f in findings:
        r = f.get("risk_rating")
        if r in order and order[r] > order[best]:
            best = r
    return best

