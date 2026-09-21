"""
Embeds chunks with a local HuggingFace model and upserts them into Pinecone.
Run this once (or whenever you add new papers):
    python -m app.ingestion.embed_and_store
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from app.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def ensure_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


def store_chunks(chunks):
    ensure_pinecone_index()
    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"Stored {len(chunks)} chunks in Pinecone index '{PINECONE_INDEX_NAME}'")
    return vectorstore


def get_existing_vectorstore():
    embeddings = get_embeddings()
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

def store_single_file(file_path: str, filename: str | None = None):
    """
    Loads, chunks, embeds, and stores a single PDF file.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from app.ingestion.chunker import chunk_documents

    documents = PyPDFLoader(file_path).load()
    chunks = chunk_documents(documents, filename=filename or os.path.basename(file_path))
    store_chunks(chunks)
    return len(chunks)

if __name__ == "__main__":
    from app.ingestion.loader import load_papers
    from app.ingestion.chunker import chunk_documents

    docs = load_papers()
    if not docs:
        print("No PDFs found in data/papers/. Add some and re-run.")
    else:
        chunks = chunk_documents(docs)
        store_chunks(chunks)