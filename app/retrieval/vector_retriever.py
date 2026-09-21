

"""
Step 1 retriever: plain vector similarity search against Pinecone.
Supports optional scoping to a single ingested file via metadata filter.
"""
from app.ingestion.embed_and_store import get_existing_vectorstore
from app.config import RETRIEVAL_K


def get_retriever(source_filter: str | None = None):
    vectorstore = get_existing_vectorstore()
    search_kwargs = {"k": RETRIEVAL_K}
    if source_filter:
        search_kwargs["filter"] = {"filename": source_filter}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("What accuracy did the model achieve?")
    for doc in results[:5]:
        print(doc.metadata.get("source"), "-", doc.page_content[:100])