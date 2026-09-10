"""FastAPI application: ingest documents, ask questions, get cited answers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm import OpenAICompatibleModel
from app.rag import RAGPipeline, Source
from app.store import VectorStore, document_id_for

ALLOWED_TYPES = {"text/plain", "text/markdown", "application/octet-stream"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def build_pipeline(settings: Settings, store: VectorStore | None = None) -> RAGPipeline:
    store = store or VectorStore.persistent(
        settings.chroma_path,
        collection_name=settings.collection_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    model = OpenAICompatibleModel(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
    return RAGPipeline(store, model, top_k=settings.top_k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests inject their own pipeline before startup; otherwise build the real one.
    if not hasattr(app.state, "pipeline"):
        app.state.pipeline = build_pipeline(get_settings())
    yield


app = FastAPI(
    title="rag-docs-api",
    version="0.1.0",
    description="Ask questions about your own documents. Answers cite the passages they came from.",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


Pipeline = Annotated[RAGPipeline, Depends(get_pipeline)]


# ── Schemas ──────────────────────────────────────────────────────────────


class IngestText(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, examples=["handbook.md"])
    text: str = Field(..., min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    name: str
    chunks: int


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    n: int
    document_id: str
    chunk_index: int
    score: float
    excerpt: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_source(cls, s: Source) -> SourceOut:
        return cls(**s.__dict__)


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str | None
    usage: dict[str, int]


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health(pipeline: Pipeline) -> dict:
    store = pipeline.retriever
    docs = store.count_documents() if isinstance(store, VectorStore) else None
    return {"status": "ok", "documents": docs}


@app.post("/documents", response_model=IngestResponse, status_code=201, tags=["documents"])
def ingest_text(body: IngestText, pipeline: Pipeline) -> IngestResponse:
    store = _store(pipeline)
    doc_id = document_id_for(body.name)
    chunks = store.add_document(doc_id, body.text, {"name": body.name, **body.metadata})
    if chunks == 0:
        raise HTTPException(status_code=422, detail="document is empty after trimming")
    return IngestResponse(document_id=doc_id, name=body.name, chunks=chunks)


@app.post("/documents/upload", response_model=IngestResponse, status_code=201, tags=["documents"])
async def ingest_file(file: UploadFile, pipeline: Pipeline) -> IngestResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported content type {file.content_type}; send .txt or .md")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 2 MB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="file must be UTF-8 text") from exc
    name = file.filename or "upload.txt"
    store = _store(pipeline)
    doc_id = document_id_for(name)
    chunks = store.add_document(doc_id, text, {"name": name})
    if chunks == 0:
        raise HTTPException(status_code=422, detail="document is empty")
    return IngestResponse(document_id=doc_id, name=name, chunks=chunks)


@app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: str, pipeline: Pipeline) -> None:
    _store(pipeline).delete_document(document_id)


@app.post("/query", response_model=AnswerOut, tags=["query"])
def query(body: Question, pipeline: Pipeline) -> AnswerOut:
    result = pipeline.ask(body.question, body.top_k)
    return AnswerOut(
        answer=result.answer,
        sources=[SourceOut.from_source(s) for s in result.sources],
        model=result.model,
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    )


def _store(pipeline: RAGPipeline) -> VectorStore:
    if not isinstance(pipeline.retriever, VectorStore):
        raise HTTPException(status_code=500, detail="retriever does not support writes")
    return pipeline.retriever
