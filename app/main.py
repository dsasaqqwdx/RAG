from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="arXiv Research Paper Q&A Bot",
    description="Ask questions about a set of research papers using RAG.",
    version="0.1.0",
)

app.include_router(router)

