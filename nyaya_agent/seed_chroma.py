from __future__ import annotations

"""Optional demo documents for local Chroma smoke tests (not legal advice)."""

from pathlib import Path

from nyaya_agent.retrieval import ChromaRetriever
from nyaya_agent.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR


def seed_demo_corpus(*, persist_dir: Path | None = None, collection: str | None = None) -> int:
    """Insert a few sample chunks into Chroma. Returns number of chunks added."""

    import chromadb

    pd = persist_dir or CHROMA_PERSIST_DIR
    col = collection or CHROMA_COLLECTION
    pd.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(pd))
    c = client.get_or_create_collection(name=col)

    ids = ["demo-sebi-kyc-1", "demo-gazette-def-1", "demo-kanoon-style-1"]
    documents = [
        (
            "SEBI circular excerpt (illustrative): Intermediaries shall carry out KYC of clients "
            "in line with PMLA rules; maintain records; and report suspicious transactions as prescribed."
        ),
        (
            "Gazette-style excerpt (illustrative): Act means the principal legislation as published "
            "in the Official Gazette; 'notification' includes delegated legislation unless context excludes."
        ),
        (
            "Case-law style excerpt (illustrative): Courts examine whether the impugned order is "
            "supported by reasons and material on record; proportionality and natural justice may apply."
        ),
    ]
    metadatas = [
        {
            "source_type": "circular",
            "title": "Illustrative SEBI compliance chunk",
            "citation": "DEMO/SEBI/KYC/1",
            "url": None,
        },
        {
            "source_type": "legislation",
            "title": "Illustrative Gazette / Act chunk",
            "citation": "DEMO/Gazette/Act/1",
            "url": None,
        },
        {
            "source_type": "case_law",
            "title": "Illustrative judicial reasoning chunk",
            "citation": "DEMO/SC/2020/1",
            "url": None,
        },
    ]

    c.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def verify_chroma_query(query: str = "KYC obligations") -> list[str]:
    r = ChromaRetriever(persist_dir=CHROMA_PERSIST_DIR, collection=CHROMA_COLLECTION)
    hits = r.search(query)
    return [h["citation"] for h in hits]


if __name__ == "__main__":
    n = seed_demo_corpus()
    print(f"Seeded {n} demo chunk(s) into {CHROMA_PERSIST_DIR} / {CHROMA_COLLECTION}")
