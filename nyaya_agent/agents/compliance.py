from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent_state import ComplianceFinding, NyayaState
from nyaya_agent.llm import get_chat_model

logger = logging.getLogger(__name__)


def compliance_agent(state: NyayaState) -> NyayaState:
    """Compliance Agent: identify gaps and propose remediation using an LLM.

    Checks retrieved documents against the query and gives a risk_rating of low, medium, or high.
    Filters out findings with a 'high' rating.
    """

    query = (state.get("query") or "").strip()
    retrieved = state.get("retrieved") or []

    logger.info(f"Compliance agent started. Analyzing {len(retrieved)} retrieved documents.")

    findings: list[ComplianceFinding] = []

    if not query:
        logger.info("No query provided, returning empty findings.")
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

    # Prepare context for the LLM
    context_blocks = []
    for idx, doc in enumerate(retrieved):
        title = doc.get("title", "Unknown Title")
        citation = doc.get("citation", "N/A")
        text = doc.get("text", "")
        # Limit text length to avoid context overflow
        context_blocks.append(f"Document {idx+1}:\nTitle: {title}\nCitation: {citation}\nContent: {text[:2000]}\n")
    
    context_str = "\n".join(context_blocks)
    
    system_prompt = (
        "You are an expert legal compliance analyzer.\n"
        "You will be provided with a user query and a set of retrieved legal documents.\n"
        "Your task is to analyze the documents against the query and identify compliance requirements and gaps.\n"
        "For each finding, provide:\n"
        "- requirement: The compliance requirement identified.\n"
        "- gap: The gap or issue found relative to the user query.\n"
        "- risk_rating: The risk rating of this gap (must be exactly one of: \"low\", \"medium\", \"high\").\n"
        "- remediation: Suggested remediation for the gap.\n"
        "- citations: A list of citations from the provided documents that support this finding.\n\n"
        "Output a STRICTLY VALID JSON object with a single key \"findings\" containing a list of these finding objects.\n"
        "Do not include any extra text, only the JSON."
    )
    
    user_prompt = f"User Query: {query}\n\nRetrieved Documents:\n{context_str}"

    try:
        model = get_chat_model()
        sys_msg = SystemMessage(content=system_prompt)
        hum_msg = HumanMessage(content=user_prompt)
        
        logger.info("Invoking LLM for compliance gap analysis...")
        out = model.invoke([sys_msg, hum_msg])
        response_text = (out.content or "").strip()
        logger.debug("Received response from LLM.")
        
        # Clean up possible markdown wrappers
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        parsed_output = json.loads(response_text)
        raw_findings = parsed_output.get("findings", [])
        logger.info(f"Successfully parsed {len(raw_findings)} raw finding(s) from LLM output.")
        
        # Filter out high risk ratings
        for f in raw_findings:
            if f.get("risk_rating", "").lower() != "high":
                findings.append(
                    {
                        "requirement": str(f.get("requirement", "")),
                        "gap": str(f.get("gap", "")),
                        # Cast to Literal type using string value, fallback to "medium"
                        "risk_rating": f.get("risk_rating", "medium").lower(), # type: ignore
                        "remediation": str(f.get("remediation", "")),
                        "citations": list(f.get("citations", [])),
                    }
                )
                
    except Exception as e:
        logger.error(f"Error during LLM compliance analysis: {e}")
        # Fallback in case of parsing error or LLM failure
        citations = [d.get("citation", "N/A") for d in retrieved if d.get("citation")]
        findings.append(
            {
                "requirement": "Map applicable obligations to the user query",
                "gap": f"Automated obligation extraction failed during LLM processing. Error: {str(e)}",
                "risk_rating": "medium",
                "remediation": "Check LLM output and parsing logic.",
                "citations": citations[:5],
            }
        )

    logger.info(f"Compliance agent finished. Generated {len(findings)} finding(s) after filtering.")
    return {"findings": findings}

