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
| Golden set | generation and review tooling built; 150 pairs not yet verified |
| Eval harness | built and tested — retrieval metrics, fusion, faithfulness, regression gate. Not yet run against a golden set |
| Benchmark sweep | statistics and the results recorder built; the sweep driver needs a live server |

**There are no performance numbers here yet**, and the recorder will not accept
one that arrives without its configuration — hardware, model, quantization,
kernel, sampler, request rate, run count and prefix-cache state are all required
before a row can be written. See [Measurement](#measurement).

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
