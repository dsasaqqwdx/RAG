

"""
Splits loaded Documents into smaller chunks for embedding.
Also normalizes a clean 'filename' metadata field (path-independent)
and applies different chunk sizes depending on document type.
"""
import os
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def _guess_chunk_params(filename: str | None):
    """
    Resumes/CVs are short, dense, and bullet-heavy — smaller chunks keep
    sections (like 'Projects') from bleeding into neighboring sections
    (like 'Work Experience'). Papers are long-form prose — bigger chunks
    keep full ideas/paragraphs together.
    """
    if filename:
        lower = filename.lower()
        if "resume" in lower or "cv" in lower:
            return 400, 50
    return CHUNK_SIZE, CHUNK_OVERLAP  


def chunk_documents(documents, filename: str | None = None):
    """
    Splits a single document's pages into chunks. Pass `filename` when you
    know it (single-file ingestion) so chunk size can adapt to document type.
    """
    chunk_size, chunk_overlap = _guess_chunk_params(filename)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Normalize a clean, path-independent filename field on every chunk.
    # (raw metadata["source"] can have inconsistent slashes depending on loader/OS)
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk.metadata["filename"] = os.path.basename(src) if src else (filename or "unknown")

    print(f"Split {len(documents)} pages into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks


def chunk_documents_multi(documents):
    """
    For a folder load spanning multiple files (e.g. BM25 rebuild): groups
    pages by their source file first, so each file gets its own appropriate
    chunk size, then returns one combined flat list of chunks.
    """
    grouped = defaultdict(list)
    for doc in documents:
        grouped[doc.metadata.get("source", "unknown")].append(doc)

    all_chunks = []
    for source, docs in grouped.items():
        filename = os.path.basename(source) if source else "unknown"
        all_chunks.extend(chunk_documents(docs, filename=filename))
    return all_chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_papers

    docs = load_papers()
    chunks = chunk_documents_multi(docs)
    if chunks:
        print("Example chunk metadata:", chunks[0].metadata)