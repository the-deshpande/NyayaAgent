# 🔄 Nyaya Agent — Development Workflow

> This guide covers how to set up, run, test, and deploy Nyaya Agent from scratch.

---

## 📋 Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Backend runtime |
| Docker | Latest | Container build & run |
| Git | Any | Version control |
| A browser | Chrome/Firefox | Testing the UI |

---

## 1️⃣ Initial Setup

### Clone the Repository

```bash
git clone https://github.com/the-deshpande/NyayaAgent
cd NyayaAgent

# Switch to the beta branch (this branch has the latest features)
git checkout beta
```

### Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Open .env and fill in your keys:
# GEMINI_API_KEY=<your-gemini-api-key>
# INDIANKANOON_API_KEY=<your-indiankanoon-key>
```

> 💡 Get a Gemini API key at: https://aistudio.google.com/app/apikey

---

## 2️⃣ Running Locally with Docker

```bash
# Build the Docker image
docker build -t nyaya-agent .

# Run the container
docker run -p 7860:7860 --env-file .env nyaya-agent

# Open in browser
# http://localhost:7860
```

---

## 3️⃣ Running Locally without Docker (Dev Mode)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

The `--reload` flag auto-restarts the server when you change Python files.

---

## 4️⃣ Making Code Changes

### Backend Changes (Python)

1. Edit the relevant `.py` file in `nyaya_agent/` or `server.py`
2. The server auto-reloads if you used `--reload` above
3. Test via browser or curl:
   ```bash
   curl http://localhost:7860/api/status
   ```

### Frontend Changes (HTML/CSS/JS)

1. Edit files in `frontend/`
2. **Increment the cache buster** in `frontend/index.html`:
   ```html
   <!-- Change v=12 to v=13 (or whatever the next number is) -->
   <link rel="stylesheet" href="style.css?v=13">
   <script src="script.js?v=13"></script>
   ```
3. Hard-refresh the browser (`Ctrl+Shift+R`)

> ⚠️ **Forgetting to increment the cache buster** is the #1 cause of changes not showing up. Always do this!

---

## 5️⃣ Testing

### Manual Testing Checklist

After any change, verify these core flows:

- [ ] Send a chat message → get a response
- [ ] Switch persona (Lawyer / Student / Explorer / Researcher)
- [ ] Upload a PDF → ask a question related to it
- [ ] View recent sessions in sidebar
- [ ] Delete a session → it disappears from sidebar
- [ ] Click "Evaluate RAG Pipeline" → shows a rating
- [ ] Generate a legal notice → "Download DOC" works
- [ ] Generate a legal notice → "View Latest Memo" shows a formatted modal

### Testing the API Directly

```bash
# Test chat
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Article 21?", "session_id": "test-123", "persona": "student"}'

# List sessions
curl http://localhost:7860/api/sessions

# Health check
curl http://localhost:7860/api/status
```

---

## 6️⃣ Git Workflow

### Branch Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable production code |
| `beta` | Latest features (this branch) |

### Making Changes

```bash
# Always work on the beta branch
git checkout beta

# Make your changes...

# Stage and commit
git add .
git commit -m "feat: describe what you changed"

# Push to GitHub
git push origin beta
```

### Commit Message Convention

Use these prefixes for clarity:

| Prefix | Use For |
|---|---|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `style:` | CSS/UI changes |
| `refactor:` | Code restructuring |
| `docs:` | Documentation only |
| `chore:` | Build/config changes |

---

## 7️⃣ Deploying to Hugging Face Spaces

The `beta` branch automatically maps to the HF Space at:
`https://huggingface.co/spaces/SeriousSam07/nyaya-agent-beta`

To deploy, simply push to the `beta` branch:

```bash
git checkout beta
git add .
git commit -m "deploy: your change description"
git push origin beta
```

Then push to the HF Space remote:
```bash
git push hf-beta beta:main
```

> 💡 The HF Space remote URL is:
> `https://huggingface.co/spaces/SeriousSam07/nyaya-agent-beta`

### Setting Up HF Space Secrets

API keys are not in the code — they must be set in HF Space Settings:

1. Go to: https://huggingface.co/spaces/SeriousSam07/nyaya-agent-beta/settings
2. Scroll to **Repository Secrets**
3. Add `GEMINI_API_KEY` and `INDIANKANOON_API_KEY`

---

## 8️⃣ Common Issues & Fixes

### "No API key was provided"
→ API keys are missing. Add them to `.env` locally, or to HF Space Secrets for deployment.

### "Style/JS changes not showing up"
→ Increment `?v=N` in `index.html` and hard-refresh (`Ctrl+Shift+R`).

### "RAG evaluation shows 0 / 5.0"
→ The vector database has no documents that match the test question. Upload relevant legal PDFs first, or change the test question in `nyaya_agent/evaluate_rag.py` (line ~19).

### "Chat sessions disappear after HF restart"
→ Expected behaviour. HF Space storage is ephemeral — SQLite DB and ChromaDB reset on every restart. This is a known limitation of the free tier.

### "Delete button returns 500"
→ Make sure `sqlite_store.py`'s `delete_session` uses `c.commit()` (not `c.connection.commit()`).

### Container won't start
→ Check `docker logs <container_id>` for the exact traceback. Usually a missing env var or import error.

---

## 9️⃣ Understanding the Codebase in 5 Minutes

1. **`server.py`** is your entry point — read the route handlers top to bottom
2. **`nyaya_agent/graph.py`** shows how AI agents are chained together
3. **`nyaya_agent/nodes/plain_chat.py`** is the simplest agent — good starting point
4. **`frontend/script.js`** — search for `handleSendMessage` to follow the full UI request lifecycle
5. **`nyaya_agent/memory/sqlite_store.py`** — all session persistence logic

For deeper architecture details, see [GUIDELINES.md](GUIDELINES.md).
