# Nyaya Agent: Project Proposal

**Nyaya Agent** is an agentic legal research system for Indian law: **three specialist agents** (Research, Compliance, Synthesis), **three national legal data sources** (Indian Kanoon, SEBI/RBI circulars, Gazette of India), retrieval-augmented generation over a unified **ChromaDB** corpus, orchestration with **LangGraph** `StateGraph` on a **shared `TypedDict` state**, **ReAct** tool loops with **`max_iterations = 6` per agent**, a **Streamlit** chat frontend, **SQLite** session storage (**six messages**: three user and three assistant, plus a **rolling summary** of older turns), a **code toggle** to use Chroma when ready or fall back to plain chat, **RAGAS** evaluation when retrieval is on, and optional **Word/PDF** export.

**Editions:** an **open-source** stack (local inference, rupee cost dominated by electricity or rented GPU) and a **proprietary** API stack (token-metered frontier models and managed services). Both follow the same agent graph, iteration limits, and sprint phases described below.

---

## 1. Problem Statement

Legal research in India is manual, subscription-gated, and error-prone. A compliance officer or advocate must navigate three structurally different data types in parallel: **Indian Kanoon** case law (unstructured judicial narrative), **SEBI/RBI** circulars (semi-structured regulatory PDFs with tables), and the **Gazette of India** (structured numbered legislation). Multi-source, multi-format work commonly takes **4–8 hours per matter**.

**Why three sources**

| Source | Nature | Format / access | Avg. tokens (order of magnitude) | Primary use |
| :--- | :--- | :--- | :--- | :--- |
| **Indian Kanoon (API)** | Judicial case law | JSON + HTML narrative; free REST API (e.g. 1,000 requests/day tier) | ~8,000 per judgment | Precedents, ratio decidendi, citator-style follow-up |
| **SEBI / RBI circulars** | Regulatory (sectoral) | Semi-structured PDFs (clauses, tables, schedules); public bulk download | ~2,000 per circular | Compliance gaps, amendment tracking |
| **Gazette of India** | Primary legislation | Numbered provisions; public PDFs (e.g. eGazette) | ~5,000 per act | Statutory definitions, amendment chains |

Together these cover **judicial reasoning**, **regulatory compliance**, and **statutory authority**.

**Nyaya Agent** addresses this with a **three-agent RAG pipeline**: all ingested material is stored and retrieved from **ChromaDB**; a **LangGraph** orchestrator routes work across **Research**, **Compliance**, and **Synthesis** agents; output is a **structured legal memo** (citations, risk rating, optional compliance-gap narrative). The user interacts through a **chat-style** interface (Streamlit) to discuss a **particular case**, with **SQLite** holding a short message window plus a **rolling summary** of older turns.

---

## 2. Project Scope & Objectives

- **Corpus:** Ingest Indian Kanoon API responses, SEBI/RBI (and similar) circulars, and Gazette PDFs; chunk, embed, and index **only** in **ChromaDB** (metadata: source type, document id, citation fields, section labels where extractable).
- **Orchestration:** **LangGraph** `StateGraph` with a **shared application state** modeled as a **`TypedDict`** passed between nodes; orchestrator may classify intent and route to the appropriate **starting** agent before handoffs.
- **Reasoning pattern:** Each specialist agent uses a **ReAct-style** tool loop with a **hard cap of `max_iterations = 6` per agent** (no unbounded tool loops).
- **Chat UX:** Streamlit chat for case-focused dialogue; optional structured memo runs when retrieval is enabled.
- **Session memory:** **SQLite** stores the **last six messages** (**three user**, **three assistant**, chronological). Older turns are **condensed into a summary** stored **alongside** the same session; the model context uses **summary + six-message window** plus system prompts.
- **Chroma readiness toggle:** A **boolean in code** (e.g. `CHROMA_READY`). When **true**, ChromaDB is assumed available and the RAG + three-agent path runs. When **false**, user traffic uses a **general-purpose chat model** without vector retrieval, while SQLite memory rules still apply.
- **Evaluation:** **RAGAS** metrics when retrieval is on (faithfulness, answer relevance, context precision).
- **Frontend:** **Streamlit** (primary UI); optional Word/PDF export via `python-docx` and **WeasyPrint** (or equivalent) as the product matures.
- **Deployment:** Budget GPU cloud (e.g. RunPod / Vast.ai), academic credits, and Streamlit Cloud for UI hosting — cost table in Section 7.

---

## 3. Why Three Agents

Configurations from one to five agents were weighed on **tool confusion**, **orchestration overhead**, **context pressure**, and **output coherence**. **Three specialists** is the chosen balance.

### 3.1 Agent roles and tools (design targets)

- **Agent 1 — Research:** All retrieval against the unified corpus. Intended tools: **LegalResearch**, **StatuteLookup**, **GraphCitator**, **AmendmentHistory** (names reflect behaviour; implementation maps them to ChromaDB queries and metadata filters, not a separate graph database in the baseline product).
- **Agent 2 — Compliance:** Clause- and obligation-level analysis. Intended tools: **ComplianceCheck**, **ObligationExtractor**, **RemediationDraft**, **LegalDateCalc** (compares retrieved regulatory chunks to facts or contract clauses supplied in session).
- **Agent 3 — Synthesis:** Memo and risk output. Intended tools: **WriteMemo**, **RiskScore**, **GraphSearch**, **Export** (structured JSON memo; optional Word/PDF).

### 3.2 LangGraph orchestrator

- **`StateGraph`** over a **shared `TypedDict`** state: all agents read/write the same typed state (e.g. query, retrieved chunks, compliance artifacts, draft memo, trace flags).
- **Routing:** Orchestrator **classifies intent** and can set the **first** agent (Research, Compliance, or Synthesis) before fixed or conditional edges between agents.
- **ReAct enforcement:** Each agent’s tool-use loop runs under **`max_iterations = 6`** (forced ceiling per agent per user turn or per graph invocation, per implementation policy).

### 3.3 Per-agent context budget (token planning)

For cost and window planning, assume **~9,500 tokens per agent invocation** as a working budget:

| Component | Tokens |
| :--- | ---: |
| System prompt | 800 |
| Retrieved chunks | 4,000 |
| Session history (from SQLite window + summary) | 1,200 |
| Tool outputs | 1,500 |
| Agent reasoning + final output | 2,000 |
| **Total** | **9,500** |

Summarisation for memory compression should target keeping **history contribution** within the allocated **1,200** tokens where possible.

---

## 4. End-to-End Query Flow (Seven Layers)

One natural-language request moves through:

1. **Input:** User submits a query in Streamlit (case thread, optional attachments later).
2. **Orchestration:** LangGraph classifies intent and selects the **starting** specialist agent.
3. **Agent reasoning:** The active agent runs a **ReAct loop** over its tools (**≤ 6 iterations**).
4. **Retrieval (when Chroma is ready):** Tool calls resolve to **ChromaDB** — vector similarity over embedded chunks, **metadata filters** (source, act, circular id, case citation), and optional **local** sparse ranking (e.g. BM25 over chunk text) fused in-process if implemented; retrieved text is grounded in the three legal sources only.
5. **Corpus:** Chunks originate only from **Kanoon**, **SEBI/RBI circulars**, and **Gazette** ingests described in Section 1.
6. **Memory:** **SQLite** supplies the **six-message** tail plus **stored summary** of older dialogue for session continuity.
7. **Output:** **Synthesis** produces the **structured JSON legal memo**; UI can show **live ReAct trace**, render memo, and support **Word/PDF** export.

When **`CHROMA_READY` is false**, layers 4–5 are skipped for that interaction: the chat model answers from **SQLite context + general knowledge**, with clear UX that answers are **not** corpus-grounded.

---

## 5. Technical Architecture

### 5.1 Open-source edition (baseline stack)

| Layer | Library / tool | Notes |
| :--- | :--- | :--- |
| LLM (agents 1–3) | Mistral-7B-Instruct via **Ollama** | Local; zero per-token API cost; ~32k context |
| LLM (orchestrator routing) | **LangGraph** + same or smaller local model | `StateGraph` routing; shared **TypedDict** state |
| LLM (memory / summary) | Small local model (e.g. Gemma-2-2B via Ollama) | Summarisation; ~8k context |
| Embeddings | **InLegalBERT** (768-d) or equivalent | Indian-legal-tuned encoder on CPU/GPU |
| Sparse retrieval (optional) | **rank-bm25** | Exact term matching; complements vectors |
| Reranker (optional) | **ms-marco-MiniLM** (cross-encoder) | Local; ~200ms per batch on CPU (order of magnitude) |
| Vector store | **ChromaDB** | Persistent; cosine similarity; rich **metadata**; single store for all three sources |
| Agent framework | **LangChain** + **LangGraph** | ReAct pattern; **max_iterations=6** per agent |
| NER / extraction | **spaCy** + India-aware rules | Case / statute / party hints for chunk metadata |
| Document loaders | **PyPDF2** + **Unstructured** (hi-res) | PDFs and scanned gazette OCR path |
| Session store | **SQLite** | Six-message window + **summary** column(s) per session |
| UI / export | **Streamlit** + **python-docx** + **WeasyPrint** | Chat UI; memo to Word/PDF |

### 5.2 Proprietary edition (alternative stack)

| Layer | Library / service | Notes |
| :--- | :--- | :--- |
| LLM (agents 1–3) | Claude Sonnet-class (Anthropic) | Large context; strong structured legal output |
| LLM (orchestrator) | Same family | Router over agents |
| LLM (memory) | GPT-4o-mini-class (OpenAI) | Fast, cheap summarisation |
| Embeddings | text-embedding-3-large (1024-d) or equivalent | API embeddings |
| Sparse retrieval | rank-bm25 (same idea as OSS) | Optional complement |
| Reranker | Cohere Rerank v3 (API) | Paid rerank per query batch |
| Vector store | Pinecone Serverless (example) | Managed vectors |
| Graph DB (optional extension) | Neo4j AuraDB | Only if product later adds graph persistence; **not** required for baseline Chroma-only corpus |
| Agent framework | LangGraph (LangChain) | Same **TypedDict** + **ReAct** + **six-iteration** cap |
| NER / extraction | spaCy + LLM-assisted JSON | Higher-accuracy structured fields |
| Doc loaders | LlamaParse + PyPDF2 | Tables and complex PDFs |
| Memory | PostgreSQL + summary buffer | Production session store |
| UI / export | FastAPI + React + WeasyPrint | Alternative to Streamlit-only deployments |

### 5.3 ChromaDB and toggle (product rules)

- **Single ChromaDB instance** holds every embedded chunk from the three sources; ingestion pipelines write here only (no parallel mandatory second vector store in baseline).
- **`CHROMA_READY == true`:** Health check passes; collections exist; agents call retrieval tools backed by Chroma.
- **`CHROMA_READY == false`:** No Chroma calls for user replies; a **single chat LLM** path serves the Streamlit session using SQLite memory only; UI should indicate **non-grounded** mode.

### 5.4 SQLite memory schema (conceptual)

- **Per session:** `session_id`, optional `case_label`, `created_at`, `updated_at`.
- **Rolling messages:** store up to **six** rows (or a JSON array) with roles `user` / `assistant` and timestamps; on each new pair beyond six, **summarise** evicted content and **append** to `conversation_summary` (or merge summaries).
- **Model input assembly:** `system` + `conversation_summary` + ordered **six** messages.

---

## 6. Evaluation (RAGAS)

When Chroma retrieval is active, track at minimum:

- **Faithfulness:** Claims supported by retrieved chunks, not free invention.
- **Answer relevance:** Alignment between user question and final answer.
- **Context precision:** Whether retrieved chunks are on-topic (penalise noise).

Use RAGAS (or equivalent harness) on a fixed eval set with golden references where available.

---

## 7. Deployment & Cost Strategy

### 7.1 Hosting options (45-day sprint framing)

| Strategy | Provider | Est. 45-day cost (order of magnitude) | Pros / cons |
| :--- | :--- | :--- | :--- |
| Budget GPU cloud | RunPod / Vast.ai | **Rs. 3,200 – 5,000** | Best value; you manage images and disks |
| Student / academic | Google Cloud / Azure credits | **Rs. 0** (credits) | Time-limited; quota friction |
| UI hosting | Streamlit Cloud | **Rs. 0** | Free tier for public or demo apps |

### 7.2 Open-source edition — token-based cost model (illustrative)

**Assumptions:** ~50 queries/day × 45 days ≈ **2,250 queries**; **~9,500 tokens** per agent budget as in Section 3.3; local Ollama inference → **Rs. 0** token line item; electricity or rented GPU dominates.

| Component | Inference | Tokens / call | Volume (45d) | Rupee cost (illustrative) |
| :--- | :--- | :--- | :--- | :--- |
| Mistral-7B (agents) | Local GPU | ~9,500 / query | 2,250 queries | Rs. 0 (local) |
| Gemma-2-2B (summary) | Local GPU | ~1,200 / summary | ~500 compressions | Rs. 0 (local) |
| InLegalBERT embeddings | Local CPU | ~512 / chunk | ~50k chunks one-time ingest | Rs. 0 (local) |
| BM25 + reranker | Local CPU | — | All queries | Rs. 0 (local) |
| ChromaDB | Local disk | — | ~1 GB index scale | Rs. 0 (OSS) |
| Indian Kanoon API | Network | ~500 tokens/doc | e.g. 5k docs ingest | Rs. 0 on free tier |
| GPU electricity | e.g. A100-class ~200 W while active | ~0.017 kWh/query | 2,250 queries @ Rs. 8/kWh | **~Rs. 310** |
| Cloud VM (optional) | Rented GPU hourly | n/a | 45 days variable | **Rs. 0 – ~2,025** depending on SKU and hours |

**Total (OSS sprint):** about **Rs. 310 – Rs. 2,335** dominated by **electricity or cloud GPU**, not tokens.

**Rule of thumb:** At Rs. 8/kWh and ~5 minutes wall time per query on a 200 W GPU, order **Rs. 0.14/query** electricity-only.

### 7.3 Proprietary edition — token-based cost model (illustrative)

**Assumptions:** Same 2,250 queries; ~6,000 input + 2,000 output tokens per routed query for primary LLM; separate smaller budget for orchestrator routing and memory compression. Dollar rates move with vendor pricing; rupee figures below are **order-of-magnitude** from a token audit.

| Component | Rate (indicative) | Tokens / call | Volume | INR (indicative) |
| :--- | :--- | :--- | :--- | :--- |
| Claude-class Sonnet (3 agents) | ~$3/M input, ~$15/M output | ~6k in + 2k out / query | 2,250 queries | **~Rs. 8,200** |
| Same (orchestrator routing) | same | ~500 in + 200 out / route | 2,250 routes | **~Rs. 780** |
| GPT-4o-mini-class (memory) | ~$0.15/M in, ~$0.60/M out | ~1.2k in + 300 out / compression | ~500 compressions | **~Rs. 62** |
| OpenAI text-embedding-3-large | ~$0.13/M tokens | 512 / chunk | ~50k chunks ingest | **~Rs. 270** |
| Cohere Rerank v3 | ~$2 / 1k queries | 1 call/query | 2,250 | **~Rs. 373** |
| Pinecone reads | free tier then per-million | ~20 reads/query | 45k reads | **Rs. 0** on free tier |
| Small CPU VM (API-only inference) | hourly | n/a | 45 days | **~Rs. 3,200** |

**Total (proprietary sprint):** about **Rs. 12,885 – Rs. 16,085** depending on rerank usage and VM size.

**Optimisations (both editions):** reduce `fetch_k` / returned chunks to cut input tokens; compress session history aggressively before each agent; cache repeated statute or circular lookups in SQLite (order **30%** repeat-hit savings in some workloads); optional hard **spend callback** to halt runs at a configured rupee ceiling for API editions.

### 7.4 Side-by-side comparison

| Dimension | Open-source edition | Proprietary edition |
| :--- | :--- | :--- |
| 45-day total (indicative) | Rs. 310 – Rs. 2,335 | Rs. 12,885 – Rs. 16,085 |
| LLM cost per query | Rs. 0 (local) | ~Rs. 3.6+ (API, order of magnitude) |
| Input tokens per query (budget) | ~9,500 local (unbilled) | ~6,000 billed |
| Output tokens per query | ~2,000 local (unbilled) | ~2,000 billed |
| Embedding cost (one-time ingest) | Rs. 0 (InLegalBERT local) | ~Rs. 270 (API embed) |
| Rerank cost per query | Rs. 0 (MiniLM local) | ~Rs. 0.17 (API rerank) |
| GPU / compute | 16 GB VRAM class recommended | Often CPU + API only |
| Data privacy | On-prem possible; no vendor LLM egress | Data sent to LLM / embed / rerank vendors |
| Context window | ~32k (typical local 7B) | ~200k (frontier API models) |
| Setup time | ~3–4 days (Docker + Ollama + GPU) | ~1 day (keys + pip) |

---

## 8. Implementation Timeline (30–45 Days)

| Phase | Name | Days | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | Ingestion + indexing | D 1–8 | Ingest three sources into **ChromaDB**; chunking + metadata schema; baseline retrieval quality on ~20 test queries. |
| 2 | Three-agent core + LangGraph | D 9–16 | All agents wired; **TypedDict** state; **ReAct** with **six-iteration** cap; JSON memo schema; **Demo 1:** flat RAG three-agent system. |
| 3 | Compliance engine | D 17–22 | Clause analyser; SEBI / DPDPA-style gap patterns; remediation drafts; redline-style diff in UI or export. |
| 4 | Advanced retrieval + eval | D 23–30 | Stronger citation and amendment handling via **metadata and chunk graph in Chroma** (or optional external graph if adopted); **RAGAS** harness; **Demo 2:** eval-backed retrieval. |
| 5 | Memory + output layer | D 31–36 | **SQLite** six-message + summary; legal memo schema hardening; Word/PDF export; optional cost-monitoring callbacks for API routes. |
| 6 | UI + hardening | D 37–45 | Streamlit polish; live ReAct trace; load test (~100 queries); token-budget alerts; **Demo 3:** release candidate. |

---

## 9. Feasibility & Risks

| Risk | Level | Mitigation |
| :--- | :--- | :--- |
| Token budget overrun | Medium | Hard caps via callbacks; aggressive summary; MMR-style retrieval (e.g. fetch_k vs return_k) to limit chunk tokens |
| Local LLM quality | Medium | Ground every claim in Chroma chunks; structured JSON with mandatory citation slots; caveat block on memos |
| Three-source corpus gaps | Medium | Architecture allows new sources without redesign |
| Chunking / OCR quality | Medium | Unstructured hi-res path for scans; human-in-loop on low-confidence pages |
| Retrieval hallucination | High | RAG + reranker + RAGAS gates; refuse when similarity is below threshold |
| 45-day timeline | Medium | Each phase ends in a **working** increment; Demo 1 by D16 is independently shippable |

**Recommendation:** For a **privacy-first, low-budget** sprint, ship the **open-source** path first (Chroma + local models + SQLite). For **client-facing demos** where long statutes must sit in context, budget the **proprietary** token line in Section 7.3.

---

## 10. Environment & Configuration

- Store secrets (API keys for optional Gemini/Claude/OpenAI routes) in **`.env`** at project root; application code loads them at startup. Example variable names: `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — use only what the deployed edition requires.
- **`CHROMA_READY`:** single source of truth in code for routing between **RAG agents** and **plain chat**.
- **Model IDs** (e.g. Gemini 2.5 Pro for cloud chat) are configuration, not hard-coded in documentation alone.
