"""
Combines vector search (semantic/fuzzy) and BM25 (exact keyword) retrieval
using LangChain's EnsembleRetriever. Each retriever votes on relevance,
and results are merged/re-ranked by combined score.
"""


"""
Combines vector search (semantic/fuzzy) and BM25 (exact keyword) retrieval
using LangChain's EnsembleRetriever. Supports optional single-document scoping.
"""
from langchain_classic.retrievers import EnsembleRetriever

from app.retrieval.vector_retriever import get_retriever
from app.retrieval.bm25_retriever import build_bm25_retriever


def get_hybrid_retriever(source_filter: str | None = None):
    vector_retriever = get_retriever(source_filter=source_filter)
    bm25_retriever = build_bm25_retriever(source_filter=source_filter)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    results = retriever.invoke("What BLEU score did the base Transformer model achieve?")
    print(f"Retrieved {len(results)} chunks\n")
    for i, doc in enumerate(results[:5]):
        print(f"{i+1}. {doc.metadata.get('source')} - page {doc.metadata.get('page')}")
        print(f"   {doc.page_content[:120]}\n")