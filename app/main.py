import json
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        base_url=settings.vllm_url, timeout=settings.request_timeout_s
    )
    yield
    await app.state.http.aclose()


app = FastAPI(title="llm-inference-platform", lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        r = await app.state.http.get("/models")
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"vllm unreachable at {settings.vllm_url}: {exc}") from exc
    return {"vllm": "ok", "models": [m["id"] for m in r.json()["data"]]}


@app.post("/query", response_model=QueryResult)
async def query(q: Query):
    model = q.model or settings.model
    body = {
        "model": model,
        "prompt": q.prompt,
        "max_tokens": q.max_tokens,
        "temperature": q.temperature,
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

    return QueryResult(
        text="".join(parts),
        model=model,
        ttft_ms=round(ttft_ms, 2),
        total_ms=round(total_ms, 2),
        n_chunks=len(parts),
    )
