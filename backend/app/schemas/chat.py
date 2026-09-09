from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    document_id: int
    chunk_id: int | None = None
    title: str
    page_number: int
    chunk_index: int
    score: float
    content: str


class WebSource(BaseModel):
    title: str
    authors: list[str] = []
    summary: str
    published: str
    entry_url: str
    pdf_url: str | None = None


class ChatRequest(BaseModel):
    question: str
    document_id: int | None = None
    session_id: int | None = None


class MemoryHitRead(BaseModel):
    memory_id: int
    content: str
    memory_type: str
    source_id: int | None = None
    score: float

    model_config = ConfigDict(from_attributes=True)


class TraceEvent(BaseModel):
    type: str
    node: str | None = None
    message: str | None = None
    score: float | None = None
    payload: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    citations: list[Citation]
    web_sources: list[WebSource] = []
    trace: list[str] = []
    memory_hits: list[MemoryHitRead] = []
    trace_events: list[dict] = []
