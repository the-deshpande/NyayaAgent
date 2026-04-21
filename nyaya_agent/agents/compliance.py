from __future__ import annotations

from agent_state import ComplianceFinding, NyayaState


def compliance_agent(state: NyayaState) -> NyayaState:
    """Compliance Agent: identify gaps and propose remediation.

    Proposal mapping: "Performs clause analysis and remediation drafting."

    This is a deterministic scaffold (no LLM) so it runs today; you can later swap in
    an LLM-powered analyzer using the retrieved texts.
    """

    query = (state.get("query") or "").strip()
    retrieved = state.get("retrieved") or []

    findings: list[ComplianceFinding] = []

    if not query:
        return {"findings": findings}

    # Minimal, explainable baseline: if there are no retrieved sources, we flag uncertainty.
    if not retrieved:
        findings.append(
            {
                "requirement": "Evidence-backed compliance assessment",
                "gap": "No sources were retrieved from the knowledge base, so the assessment cannot be grounded.",
                "risk_rating": "high",
                "remediation": "Ingest the relevant corpus into ChromaDB and re-run retrieval; ensure the query includes the regulated entity, activity, and timeframe.",
                "citations": [],
            }
        )
        return {"findings": findings}

    # If we do have retrieval, create a single “summary” finding as a placeholder.
    citations = [d.get("citation", "N/A") for d in retrieved if d.get("citation")]
    findings.append(
        {
            "requirement": "Map applicable obligations to the user query",
            "gap": "Automated obligation extraction is not implemented yet (Phase 3 in proposal).",
            "risk_rating": "medium",
            "remediation": "Implement clause extraction + obligation taxonomy; generate a checklist and verify each item against the company’s controls/policies.",
            "citations": citations[:5],
        }
    )

    return {"findings": findings}

