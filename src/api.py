"""A small HTTP API around RAGPipeline: ingest, query (blocking and streamed), delete.

Run with: uvicorn src.api:app --reload
Requires the `http-api` extras (fastapi, uvicorn) - see requirements.txt.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from src.config.settings import load_settings
from src.errors import RAGError
from src.pipeline import RAGPipeline, build_pipeline

logger = logging.getLogger("rag_multivec.api")


class IngestRequest(BaseModel):
    records: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    chunks_indexed: int
    chunk_ids: List[str]


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceOut(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceOut]
    cited_source_ids: List[str]


def _sources_out(sources) -> List[SourceOut]:
    return [
        SourceOut(id=s.document.id, text=s.document.text, score=s.score, metadata=s.document.metadata)
        for s in sources
    ]


def create_app(pipeline: Optional[RAGPipeline] = None) -> FastAPI:
    """Build the FastAPI app. Pass `pipeline` to inject one (tests, or a
    pipeline built with non-default settings); omit it to build the default
    one from environment settings lazily at startup (not at import time, so
    importing this module never requires API keys to be set)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(load_settings())
        yield

    app = FastAPI(title="rag-multivec", lifespan=lifespan)

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
        try:
            chunks = pipeline.ingest(body.records)
        except RAGError as exc:
            logger.error("POST /ingest failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(chunks_indexed=len(chunks), chunk_ids=[c.id for c in chunks])

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
        try:
            result = pipeline.query(body.question, top_k=body.top_k)
        except RAGError as exc:
            logger.error("POST /query failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=_sources_out(result.sources),
            cited_source_ids=result.cited_source_ids,
        )

    @app.post("/query/stream")
    async def query_stream(body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
        # query_stream() itself does a blocking retrieval call before returning,
        # and the returned iterator makes blocking generation calls - route both
        # off the event loop via a threadpool rather than freezing other requests.
        try:
            streaming_answer = await run_in_threadpool(pipeline.query_stream, body.question, top_k=body.top_k)
        except RAGError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        async def token_stream():
            # A GenerationError raised mid-stream propagates here, but by then
            # the 200 response has already started - the client sees a
            # truncated body rather than an error status, an inherent
            # limitation of streaming HTTP responses once headers are sent.
            async for chunk in iterate_in_threadpool(iter(streaming_answer)):
                yield chunk

        return StreamingResponse(token_stream(), media_type="text/plain")

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
        pipeline.delete(document_id)
        return {"deleted": document_id}

    return app


app = create_app()
