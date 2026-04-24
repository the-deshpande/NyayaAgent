from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# When True, ChromaDB is assumed populated and reachable; RAG graph runs.
CHROMA_READY: bool = _bool_env("CHROMA_READY", default=False)

# Persistent Chroma directory (created on first use).
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma")))

# SQLite file for chat sessions (six-message window + rolling summary).
SQLITE_PATH: Path = Path(os.getenv("SQLITE_PATH", str(PROJECT_ROOT / "data" / "nyaya_chat.sqlite3")))

# Primary chat / agent model (Gemma-4-31b-it when using openrouter).
CHAT_MODEL_ID: str = os.getenv("CHAT_MODEL_ID", "google/gemma-4-31b-it:free")

# Chroma collection name for legal corpus.
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "nyaya_kanoon")

# Max tool rounds per agent (proposal: 6).
MAX_REACT_ITERATIONS: int = int(os.getenv("MAX_REACT_ITERATIONS", "6"))

def setup_environment() -> None:
    """Set global environment variables for the agent."""
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
