# llm-inference-platform

Self-hosted LLM serving and evaluation stack: Qwen3-8B on vLLM behind a FastAPI
gateway, hybrid retrieval over PostgreSQL + pgvector, and an evaluation harness
that gates releases in CI. Runs on a single 48GB L40S.

Work in progress. Serving and retrieval land first; the eval harness and the
benchmark suite follow.

## Architecture

```
client → FastAPI gateway
           ├→ retrieval: pgvector + Postgres FTS
           │    BM25 + dense (bge-m3) → reciprocal rank fusion → bge-reranker-v2-m3
           └→ generation: vLLM (OpenAI-compatible), Qwen3-8B

eval harness (CLI) → golden set
           → retrieval: precision@k, recall@k, MRR, nDCG
           → answers: faithfulness, answer relevancy
           → non-zero exit when a metric regresses past threshold
```

## Running it

```
cp .env.example .env      # MODEL, HF_TOKEN
uv sync --extra serve     # vllm is linux/CUDA only, hence the extra
./init.sh                 # postgres, vllm, api
./init.sh eval            # harness against the golden set
./init.sh bench           # benchmark sweep → results.json
```

## Benchmark methodology

Numbers in `results.json` are measured, not estimated. The harness enforces:

- Latency and throughput are separate runs. Latency at a low request rate so
  TTFT reflects compute rather than queueing; throughput as a saturation sweep
- Prefix caching disabled or flushed between runs, recorded in the output
- Warmup requests discarded, count recorded
- TTFT and ITL reported separately, never collapsed into one "latency"
- Every row carries hardware, model, quantization, input length, request rate,
  and run count. FP16 is compared against AWQ-INT4 on identical inputs
