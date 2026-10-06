# llm-inference-platform

Self-hosted LLM serving with retrieval, and an evaluation harness that gates
releases on measured quality rather than a smoke test.

Qwen3-8B on vLLM behind a FastAPI gateway, hybrid retrieval over
PostgreSQL + pgvector across ~10k arXiv abstracts, and a CLI that scores
retrieval and answer quality against a hand-verified question set and exits
non-zero when a metric regresses.

## Status

| | |
|---|---|
| Serving + gateway | working — `/query` returns a completion with per-request TTFT |
| Corpus | 10,000 papers, 11,991 chunks, HNSW + GIN indexed |
| Evaluation set | 150 pairs, **model-generated and automatically screened — not human-verified** |
| Eval harness | built and tested — retrieval metrics, fusion, faithfulness, regression gate. Not yet run against a golden set |
| Benchmark sweep | **done — 16 runs in `results.json`**, BF16 vs AWQ across input length and concurrency |

Numbers below are measured, not estimated. The recorder refuses a row that
arrives without its configuration — hardware, model, quantization, kernel,
sampler, request rate, warmup count, prefix-cache state, and concurrency on a
saturation run. See [Measurement](#measurement).

## Results

One L40S 48GB, vLLM 0.28.0, FlashInfer sampler, prefix caching **disabled**,
10 warmup requests discarded, 120 requests per run.

### Latency at a low arrival rate (4 req/s)

Paced so TTFT reflects compute rather than queueing.

| input tokens | BF16 TTFT p50 | BF16 TTFT p95 | BF16 ITL p50 | AWQ TTFT p50 | AWQ TTFT p95 | AWQ ITL p50 |
|---|---|---|---|---|---|---|
| 128 | 73.5 ms | 85.3 ms | 23.67 ms | 34.1 ms | 41.0 ms | 8.42 ms |
| 512 | 104.7 ms | 160.2 ms | 25.11 ms | 61.8 ms | 96.3 ms | 8.99 ms |
| 1024 | 164.7 ms | 350.1 ms | 28.79 ms | 107.2 ms | 239.9 ms | 10.28 ms |
| 2048 | 761.4 ms | 1745.6 ms | 55.59 ms | 570.3 ms | 1490.7 ms | 29.77 ms |

**TTFT is not linear in prompt length.** It roughly doubles from 128 to 1024
tokens, then jumps 4.6x between 1024 and 2048 on BF16 (164.7 to 761.4 ms).
Prefill is O(n) per layer, so a bend that sharp is a scheduling or chunked-prefill
boundary rather than raw compute, and it is flagged for investigation before any
of it is quoted as a prefill cost.

### Throughput and cost at saturation

A separate experiment: unpaced, with concurrency capped.

| concurrency | BF16 req/s | BF16 TTFT p95 | BF16 $/1k | AWQ req/s | AWQ TTFT p95 | AWQ $/1k |
|---|---|---|---|---|---|---|
| 1 | 0.35 | 98 ms | $0.8712 | 0.87 | 89 ms | $0.3497 |
| 8 | 2.11 | 506 ms | $0.1435 | 4.25 | 557 ms | $0.0713 |
| 32 | 4.91 | 2230 ms | $0.0616 | 6.97 | 2209 ms | $0.0435 |
| 64 | 6.31 | 4946 ms | $0.0480 | 7.62 | 4934 ms | $0.0398 |

**AWQ wins throughput and cost at every concurrency**, reaching 7.62 req/s
against BF16's 6.31 and cutting cost per 1,000 queries from $0.0480 to $0.0398.
Its ITL advantage is the larger effect — 10.28 ms against 28.79 ms at 1024
tokens — which is what 4-bit weights buy on a decode path bound by memory
bandwidth.

**The tail is the real story at high concurrency.** At 64 concurrent requests,
TTFT p95 reaches ~4.9 s on both models while p50 stays near 720 ms. That gap is
queueing, not compute, and it is the reason latency and throughput are run as
separate experiments: a single averaged "latency" number here would describe
neither.

## Architecture

```
client → FastAPI gateway
           ├→ retrieval: pgvector + Postgres FTS
           │    BM25 + dense (bge-m3) → reciprocal rank fusion → bge-reranker-v2-m3
           └→ generation: vLLM (OpenAI-compatible), Qwen3-8B

eval/  → hand-verified question set
           retrieval: precision@k, recall@k, MRR, nDCG
           answers:   faithfulness, answer relevancy
           gate:      non-zero exit when a metric regresses past threshold

bench/ → vllm bench serve sweeps → results.json → cost curve
```

## Measurement

The point of the project is the measurements, so the rules are in the code rather
than in a paragraph of intent:

- **Latency and throughput are separate experiments.** Latency at a low request
  rate, so TTFT reflects compute rather than queueing. Throughput as a saturation
  sweep. One number for both would be meaningless.
- **Prefix caching is disabled or flushed between runs**, and the run records
  that it was. Repeating a sweep against a warm server inflates throughput.
- **Warmup requests are discarded** and the count recorded — the first requests
  after load include CUDA graph capture and compilation.
- **TTFT and ITL are reported separately.** Prefill is compute-bound and decode is
  memory-bandwidth-bound; collapsing them into "latency" hides which one moved.
- **Every row carries its configuration:** hardware, model, quantization, kernel,
  sampler, vLLM version, input length, request rate, run count.

`/query` measures TTFT around the stream rather than reconstructing it after the
fact, so a buffered response cannot masquerade as a fast one.

## Endpoints

| | |
|---|---|
| `POST /query` | raw completion straight to vLLM, no retrieval — this is what the benchmarks drive, so the numbers describe the serving path alone |
| `POST /rag` | retrieve, then answer from the retrieved context; returns the answer, the papers cited, and retrieval time separately from generation time |
| `GET /health` | vLLM reachability, loaded models, and whether an embedder is present |

`/rag` takes `mode` as `dense`, `lexical` or `hybrid`. Lexical needs no embedder;
the other two return 503 when none is loaded, since bge-m3 and vLLM compete for
the same card and running both means giving vLLM a lower memory fraction.
Retrieval time is reported apart from TTFT because they are different costs with
different fixes.

## Quickstart

```
cp .env.example .env            # absolute model paths
uv sync
./scripts/pull_models.sh        # weights into ~/models
./scripts/setup_postgres.sh     # postgres + pgvector, once per host
./init.sh ingest                # arxiv corpus into pgvector
./init.sh up                    # postgres, vllm, gateway
./scripts/verify_day1.sh        # end-to-end check
```

vLLM is Linux/CUDA only and installs outside the project venv on hosts that allow
it:

```
uv pip install --system vllm    # root images
uv sync --extra serve           # where PEP 668 blocks --system
```

`init.sh` uses whichever is present.

## Packaging

```
MODEL=/models/qwen3-8b VLLM_URL=http://<gpu-host>:8000/v1 docker compose up --build
```

Brings up Postgres with pgvector and the gateway. vLLM is deliberately not a
service here — it needs a GPU, and including it would imply the stack deploys
somewhere it cannot actually serve.

## Layout

```
app/      gateway, settings, arxiv client, chunking, ingest
eval/     golden set generation, review tool, harness
bench/    benchmark sweeps
scripts/  host setup, model pulls, acceptance checks
sql/      extensions and schema
```

## Why it is built this way

[DECISIONS.md](DECISIONS.md) records the choices and the reasoning, including a
chunk-size change that the data reversed and an arXiv pagination ceiling that
only shows up three minutes into a run.
