from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agent_state import RetrievedDoc, SourceType

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RetrieverConfig:
    collection: str = "nyaya"
    top_k: int = 8


class Retriever:
    """No-op retriever when Chroma is disabled or unavailable."""

    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

    def search(self, query: str) -> list[RetrievedDoc]:
        _ = query
        return []


class ChromaRetriever(Retriever):
    """Query ChromaDB persistent store; uses Chroma default embedding for `query_texts`."""

    def __init__(self, *, persist_dir: Path, collection: str, top_k: int = 8):
        super().__init__(RetrieverConfig(collection=collection, top_k=top_k))
        import chromadb

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection)

    def search(self, query: str) -> list[RetrievedDoc]:
        q = query.strip()
        if not q:
            return []
        try:
            n = self._collection.count()
            if n == 0:
                return []
            res = self._collection.query(
                query_texts=[q],
                n_results=min(self.config.top_k, max(1, n)),
            )
        except Exception:
            return []

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out: list[RetrievedDoc] = []
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            st = meta.get("source_type", "unknown")
            if st not in ("case_law", "circular", "legislation", "unknown"):
                st = "unknown"
            out.append(
                {
                    "id": str(doc_id),
                    "source_type": st,  # type: ignore[arg-type]
                    "title": str(meta.get("title", "")),
                    "citation": str(meta.get("citation", "")),
                    "url": meta.get("url"),
                    "text": text or "",
                }
            )
        return out


def get_retriever() -> Retriever:
    """Return Chroma-backed retriever only when `CHROMA_READY` is true in settings."""

    from nyaya_agent.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHROMA_READY

    if CHROMA_READY:
        return ChromaRetriever(
            persist_dir=CHROMA_PERSIST_DIR,
            collection=CHROMA_COLLECTION,
            top_k=RetrieverConfig().top_k,
        )
    return Retriever(RetrieverConfig(collection=CHROMA_COLLECTION))


def make_stub_doc(
    *,
    id: str,
    text: str,
    title: str = "Stub document",
    citation: str = "N/A",
    source_type: SourceType = "unknown",
    url: str | None = None,
) -> RetrievedDoc:
    return {
        "id": id,
        "source_type": source_type,
        "title": title,
        "citation": citation,
        "url": url,
        "text": text,
    }
