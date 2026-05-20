# 📐 Nyaya Agent — Architecture & Contributor Guidelines

> This document is for contributors and developers who want to understand the codebase, add features, or make UI changes.

---

## 🗂️ Repository Structure

```
/
├── Dockerfile                  # Single-stage Docker build
├── server.py                   # FastAPI entrypoint — all HTTP routes
├── agent_state.py              # Shared LangGraph state TypedDict
├── requirements.txt            # All Python dependencies
├── .env.example                # Template for environment variables
├── frontend/
│   ├── index.html              # The entire UI — single page app
│   ├── style.css               # All styles (dark theme, glassmorphism)
│   └── script.js               # All frontend JavaScript
└── nyaya_agent/
    ├── graph.py                # LangGraph graph wiring
    ├── llm.py                  # LLM factory (Gemini via LangChain)
    ├── settings.py             # Env vars, ChromaDB settings
    ├── retrieval.py            # ChromaDB vector store + search
    ├── evaluate_rag.py         # RAGAS evaluation pipeline
    ├── agents/
    │   ├── research.py         # Legal research agent node
    │   ├── compliance.py       # Compliance audit agent node
    │   └── synthesis.py        # Final synthesis node
    ├── nodes/
    │   └── plain_chat.py       # Default chat node (no RAG)
    └── memory/
        └── sqlite_store.py     # SQLite session memory store
```

---

## 🏗️ Core Architecture

### Request Flow

```
User types message
      │
      ▼
[script.js] POST /api/chat
      │
      ▼
[server.py] augments prompt with persona + session context
      │
      ▼
[LangGraph graph] routes through agent nodes
      │
  ┌───┴──────────────────────┐
  │                          │
  ▼                          ▼
plain_chat node        compliance / research / synthesis nodes
(default)              (triggered by keywords/intent)
  │                          │
  └──────────┬───────────────┘
             │
             ▼
    [sqlite_store.py] persist exchange + prune to 6 sessions
             │
             ▼
    [server.py] return { response, memo, summary }
             │
             ▼
    [script.js] render response in chat UI
```

---

## 📄 Key Files Explained

### `server.py`
The FastAPI application. All API routes live here:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve `frontend/index.html` |
| `/api/chat` | POST | Main chat endpoint |
| `/api/sessions` | GET | List up to 6 recent sessions |
| `/api/session/{id}` | GET | Load a specific session's messages |
| `/api/session/{id}` | DELETE | Delete a session |
| `/api/upload` | POST | Upload PDF/TXT to ChromaDB |
| `/api/evaluate_rag` | POST | Run RAGAS evaluation |
| `/api/status` | GET | Health check |

**Important:** The server injects the user's **persona** into every message before sending to the LLM. This persona context is stored in the `persona` field in the request body.

---

### `nyaya_agent/graph.py`
Defines the LangGraph pipeline. This is where you wire agent nodes together.

**To add a new agent node:**
1. Create your node function in `nyaya_agent/agents/your_agent.py`
2. Import and add it as a node in `graph.py`
3. Add routing logic in the graph's conditional edges

---

### `nyaya_agent/memory/sqlite_store.py`
Handles all session persistence. Key methods:

| Method | Description |
|---|---|
| `ensure_session(session_id)` | Create session if it doesn't exist |
| `append_exchange(session_id, user, assistant)` | Save a message pair |
| `get_context(session_id)` | Get rolling summary + last N messages |
| `get_recent_sessions(limit=6)` | List recent sessions with titles |
| `delete_session(session_id)` | Delete session + all its messages |
| `prune_sessions(limit=6)` | Auto-delete sessions beyond limit |

**Rolling Summary:** When a session exceeds 6 message pairs, older messages are summarized by the LLM and stored as `summary`. The raw messages are deleted. This keeps the context window manageable.

---

### `nyaya_agent/retrieval.py`
ChromaDB-based RAG retriever using `law-ai/InLegalBERT` embeddings.

- Documents are chunked and embedded on upload
- At query time, top-K chunks are retrieved and injected into the LLM prompt
- If no documents are uploaded, RAG is skipped and the LLM answers from training data only

---

### `frontend/script.js`
All UI logic in one file. Key globals and functions:

| Variable/Function | Purpose |
|---|---|
| `sessionId` | Current active session UUID |
| `selectedPersona` | Current user persona |
| `latestMemoText` | Text of the last generated memo |
| `handleSendMessage()` | Sends user message to `/api/chat` |
| `fetchRecentSessions()` | Loads sidebar session list |
| `loadSession(sid)` | Switches to a different session |
| `exportToPdf(text)` | Opens in-page print modal |
| `exportToDoc(text)` | Downloads as .DOC file |
| `addMessage(text, role)` | Renders a chat bubble |

**Cache Busting:** When changing `script.js` or `style.css`, increment the `?v=N` query parameter in `index.html`'s `<script>` and `<link>` tags to force browsers to reload the new file.

```html
<!-- In index.html, update v=12 → v=13 after every JS/CSS change -->
<link rel="stylesheet" href="style.css?v=12">
<script src="script.js?v=12"></script>
```

---

### `frontend/index.html`
The complete single-page app HTML. Key element IDs:

| Element ID | Purpose |
|---|---|
| `chat-messages` | Container for all chat bubbles |
| `user-input` | The textarea for user input |
| `send-btn` | The Execute button |
| `new-session-btn` | Starts a new chat |
| `recent-sessions-list` | Sidebar session list `<ul>` |
| `recents-toggle` | Collapsible Recents header |
| `context-summary-text` | Shows rolling session summary |
| `eval-rag-btn` | Triggers RAG evaluation |
| `view-memo-btn` | Opens the latest memo as PDF modal |
| `persona-select` | Profile switcher dropdown |

---

## 🎨 UI & Styling Guidelines

The app uses a **dark legal theme** with gold accents.

### CSS Variables (defined in `style.css`)
```css
--judicial-gold: #c5a059;    /* Primary gold accent */
--sidebar-black: #0d0d0d;    /* Sidebar background */
--glass-border: rgba(255,255,255,0.08); /* Card borders */
--text-color: rgba(255,255,255,0.85);   /* Main text */
```

### Design Principles
- **Glassmorphism** cards for chat messages
- **Gold gradient** accents for headings and highlights
- **Collapsible sidebar sections** — never hide content below the fold
- **Lucide Icons** — used throughout (loaded via CDN)
- **Marked.js** — renders markdown in chat messages

---

## ➕ How to Add a New Feature

### Adding a New Top-Nav Button
1. In `index.html`, find the `.top-nav-menu` div and add a new `<button class="top-nav-btn">` element
2. In `script.js`, query the button by ID and add an event listener
3. Implement the handler function

### Adding a New API Endpoint
1. In `server.py`, add a new `@app.get/post/delete(...)` route
2. Add the corresponding fetch call in `script.js`

### Adding a New Agent Node
1. Create `nyaya_agent/agents/your_agent.py` with a function `your_agent(state) -> dict`
2. In `graph.py`, add `.add_node("your_agent", your_agent)` and wire it with `.add_edge()`

### Changing the LLM Model
Edit `nyaya_agent/llm.py` — the model name is configured there.

---

## ⚠️ Important Notes for Contributors

1. **Never commit `.env`** — it is in `.gitignore`. Use `.env.example` as a template.
2. **Always increment cache buster** (`?v=N`) in `index.html` when modifying `script.js` or `style.css`.
3. **The `data/` directory** contains ChromaDB vectors and SQLite — do not commit it (it's in `.gitignore`). It is generated at runtime.
4. **Session pruning** — the app auto-prunes to max 6 sessions after every chat. This is intentional.
5. **HF Space secrets** — API keys are stored as Space Secrets in HF, not in the code.
6. **Port** — The app runs on port `7860` (required by Hugging Face Spaces). Do not change this.

---

## 🐛 Debugging Tips

- **Backend logs**: Check HF Space logs tab or run `docker compose logs -f`
- **Frontend errors**: Open browser DevTools → Console
- **RAG not working**: Check if documents are uploaded. The ChromaDB `data/` directory is ephemeral on HF Spaces (resets on restart).
- **Sessions not showing**: Check `/api/sessions` endpoint returns 200. The SQLite DB is also ephemeral on HF Spaces.
- **Cache issue**: Always hard-refresh (`Ctrl+Shift+R`) after deployments.
