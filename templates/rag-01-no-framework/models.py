from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    search_mode: SearchMode = SearchMode.hybrid


class SourceInfo(BaseModel):
    id: str
    chunk_text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class IngestURLRequest(BaseModel):
    url: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    source_type: str
    chunk_count: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str


class DeleteResponse(BaseModel):
    message: str
