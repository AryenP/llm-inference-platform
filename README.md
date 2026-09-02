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
cp .env.example .env      # set the absolute model paths
uv sync                   # api + eval deps; vllm is installed separately, see below
./scripts/pull_models.sh      # weights into ~/models
./scripts/setup_postgres.sh   # once per host: postgres + pgvector
./init.sh ingest              # arxiv corpus into pgvector
./init.sh                 # postgres, vllm, api
./init.sh test            # unit tests
./init.sh eval            # harness against the golden set
./init.sh bench           # benchmark sweep → results.json
```

Postgres runs natively rather than through docker-compose: the GPU host is itself
a container with no docker daemon. `docker-compose.yml` is the week-7 packaging
target, not the dev path.

vLLM is linux/CUDA only and is installed on the GPU host rather than into the
project venv:

```
uv pip install --system vllm     # pod / root images
uv sync --extra serve            # WSL and anywhere PEP 668 blocks --system
```

`init.sh` uses whichever is present.

`/query` proxies a streamed completion and returns the text with `ttft_ms` and
`total_ms` measured around the stream, so first-token latency is recorded per
request rather than reconstructed afterwards.

## Benchmark methodology

Numbers in `results.json` are measured, not estimated. The harness enforces:

- Latency and throughput are separate runs. Latency at a low request rate so
  TTFT reflects compute rather than queueing; throughput as a saturation sweep
- Prefix caching disabled or flushed between runs, recorded in the output
- Warmup requests discarded, count recorded
- TTFT and ITL reported separately, never collapsed into one "latency"
- Every row carries hardware, model, quantization, input length, request rate,
  and run count. BF16 (`Qwen/Qwen3-8B`) is compared against 4-bit AWQ
  (`Qwen/Qwen3-8B-AWQ`) on identical inputs
