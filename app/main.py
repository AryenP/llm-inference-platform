import json
import time
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from app.rag import MODES, build_prompt, cite, contexts, retrieve, timed
from app.settings import settings


class Query(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    model: str | None = None


class QueryResult(BaseModel):
    text: str
    model: str
    ttft_ms: float
    total_ms: float
    n_chunks: int


class RagQuery(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid"
    max_tokens: int = 256


class RagResult(BaseModel):
    answer: str
    citations: list[str]
    n_contexts: int
    retrieval_ms: float
    ttft_ms: float
    total_ms: float


def load_embedder():
    # optional: lexical retrieval needs no embedder, and loading bge-m3 competes
    # with vLLM for the same card. Absent means dense and hybrid are unavailable.
    if not settings.embed_model:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embed_model)
    except Exception:  # noqa: BLE001 - absence is a supported state, not a failure
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        base_url=settings.vllm_url, timeout=settings.request_timeout_s
    )
    app.state.embedder = load_embedder()
    yield
    await app.state.http.aclose()


app = FastAPI(title="llm-inference-platform", lifespan=lifespan)


async def stream_completion(prompt: str, model: str, max_tokens: int, temperature: float):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # streamed so ttft is the real first-token latency rather than the whole completion
        "stream": True,
    }
    parts: list[str] = []
    ttft_ms = None
    start = time.perf_counter()

    async with app.state.http.stream("POST", "/completions", json=body) as r:
        if r.status_code >= 400:
            detail = (await r.aread()).decode()[:400]
            raise HTTPException(502, f"vllm returned {r.status_code}: {detail}")
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            text = json.loads(payload)["choices"][0].get("text", "")
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            parts.append(text)

    total_ms = (time.perf_counter() - start) * 1000
    if ttft_ms is None:
        raise HTTPException(502, "vllm returned no tokens")
    return "".join(parts), ttft_ms, total_ms, len(parts)


@app.get("/health")
async def health():
    try:
        r = await app.state.http.get("/models")
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"vllm unreachable at {settings.vllm_url}: {exc}") from exc
    return {
        "vllm": "ok",
        "models": [m["id"] for m in r.json()["data"]],
        "embedder": settings.embed_model if app.state.embedder else None,
    }


@app.post("/query", response_model=QueryResult)
async def query(q: Query):
    model = q.model or settings.model
    text, ttft_ms, total_ms, n = await stream_completion(
        q.prompt, model, q.max_tokens, q.temperature
    )
    return QueryResult(
        text=text, model=model, ttft_ms=round(ttft_ms, 2),
        total_ms=round(total_ms, 2), n_chunks=n,
    )


@app.post("/rag", response_model=RagResult)
async def rag(q: RagQuery):
    if q.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")

    embedding = None
    if q.mode in ("dense", "hybrid"):
        if app.state.embedder is None:
            raise HTTPException(503, "no embedder loaded; use mode=lexical or set EMBED_MODEL")
        embedding = app.state.embedder.encode(q.question, normalize_embeddings=True)

    def fetch():
        with psycopg.connect(settings.database_url) as conn:
            register_vector(conn)
            hits = retrieve(conn, q.mode, q.question, embedding, q.k)
            return hits, contexts(conn, hits)

    (hits, blocks), retrieval_ms = timed(fetch)
    if not blocks:
        raise HTTPException(404, "retrieval returned nothing for that question")

    answer, ttft_ms, total_ms, _ = await stream_completion(
        build_prompt(q.question, blocks), settings.model, q.max_tokens, 0.0
    )
    return RagResult(
        answer=answer,
        citations=cite(hits),
        n_contexts=len(blocks),
        retrieval_ms=round(retrieval_ms, 2),
        ttft_ms=round(ttft_ms, 2),
        total_ms=round(total_ms, 2),
    )
