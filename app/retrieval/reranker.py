

"""
Reranks a pool of candidate chunks using a cross-encoder model.
Wrapped in @traceable so it shows up as its own step in LangSmith traces,
even though it's a raw sentence-transformers call, not a LangChain object.
"""
from sentence_transformers import CrossEncoder
from langsmith import traceable
from app.config import FINAL_TOP_K

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


@traceable(name="cross_encoder_rerank", run_type="tool")
def rerank(query: str, chunks: list, top_k: int = FINAL_TOP_K):
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in chunks]
    scores = reranker.predict(pairs)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    return [doc for doc, score in scored_chunks[:top_k]]
if __name__ == "__main__":
    from app.retrieval.hybrid import get_hybrid_retriever

    query = "What BLEU score did the base Transformer model achieve?"
    retriever = get_hybrid_retriever()
    candidates = retriever.invoke(query)

    print(f"Before reranking: {len(candidates)} candidates\n")

    top_chunks = rerank(query, candidates)

    print(f"After reranking: kept top {len(top_chunks)}\n")
    for i, doc in enumerate(top_chunks):
        print(f"{i+1}. {doc.metadata.get('source')} - page {doc.metadata.get('page')}")
        print(f"   {doc.page_content[:120]}\n")