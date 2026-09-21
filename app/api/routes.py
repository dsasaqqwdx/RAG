
import os
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import AskRequest, AskResponse, SourceChunk, IngestResponse, DocumentInfo
from app.retrieval.hybrid import get_hybrid_retriever
from app.retrieval.reranker import rerank
from app.generation.answer_chain import generate_answer
from app.ingestion.embed_and_store import store_single_file
from app.ingestion.manifest import hash_bytes, find_existing, record_ingestion, list_ingested_documents
from app.config import PAPERS_DIR

router = APIRouter()

# Cache only the "no filter" hybrid retriever, since that's the common case.
# A scoped (per-document) retriever is rebuilt fresh each time — a known
# tradeoff documented in the README.
_retriever = None


def get_cached_retriever(source_filter: str | None = None):
    global _retriever
    if source_filter:
        return get_hybrid_retriever(source_filter=source_filter)
    if _retriever is None:
        _retriever = get_hybrid_retriever()
    return _retriever


def safe_page_number(metadata: dict):
    page = metadata.get("page")
    if page is None:
        return None
    try:
        return int(page) + 1
    except (ValueError, TypeError):
        return None


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return list_ingested_documents()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retriever = get_cached_retriever(source_filter=request.source)
    candidates = retriever.invoke(request.question)

    if not candidates:
        return AskResponse(
            answer="I don't have enough information in the provided papers to answer that.",
            sources=[],
        )

    chunks = rerank(request.question, candidates)
    answer = generate_answer(request.question, chunks)

    sources = [
        SourceChunk(
            source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
            page=safe_page_number(doc.metadata),
            snippet=doc.page_content[:150],
        )
        for doc in chunks
    ]

    return AskResponse(answer=answer, sources=sources)


@router.post("/ingest", response_model=List[IngestResponse])
def ingest_pdfs(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append(IngestResponse(
                filename=file.filename,
                chunks_stored=0,
                message="Skipped — only PDF files are supported",
            ))
            continue

        content = file.file.read()
        file_hash = hash_bytes(content)
        existing = find_existing(file_hash)

        if existing:
            results.append(IngestResponse(
                filename=file.filename,
                chunks_stored=existing["chunks"],
                message=f"Skipped — identical content already ingested as '{existing['filename']}'",
            ))
            continue

        os.makedirs(PAPERS_DIR, exist_ok=True)
        save_path = os.path.join(PAPERS_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(content)

        chunk_count = store_single_file(save_path, filename=file.filename)
        record_ingestion(file_hash, file.filename, chunk_count)

        results.append(IngestResponse(
            filename=file.filename,
            chunks_stored=chunk_count,
            message=f"Successfully ingested {file.filename}",
        ))

    global _retriever
    _retriever = None  # invalidate cache so BM25 picks up new files

    return results