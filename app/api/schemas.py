

from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None  


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    message: str


class DocumentInfo(BaseModel):
    filename: str
    chunks: int