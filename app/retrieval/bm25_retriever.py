
"""
BM25 keyword search retriever. Rebuilt in-memory from disk each time,
with optional scoping to a single ingested file.
"""
from langchain_classic.retrievers import BM25Retriever

from app.ingestion.loader import load_papers
from app.ingestion.chunker import chunk_documents_multi
from app.config import RETRIEVAL_K


def build_bm25_retriever(source_filter: str | None = None):
    documents = load_papers()
    chunks = chunk_documents_multi(documents)

    if source_filter:
        chunks = [c for c in chunks if c.metadata.get("filename") == source_filter]

    if not chunks:
        chunks = chunk_documents_multi(documents)  # fallback: don't crash on an empty filter match

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = RETRIEVAL_K
    return retriever

if __name__ == "__main__":
    retriever = build_bm25_retriever()
    results = retriever.invoke("BLEU score WMT 2014")
    for doc in results[:5]:
        print(doc.metadata.get("source"), "- page", doc.metadata.get("page"), "-", doc.page_content[:100])