# Decisions

Why things are the way they are, including the ones that turned out wrong.

## Qwen3-8B, served with vLLM

An 8B model on one 48 GB card is roughly what a small team actually deploys, so
the numbers mean something outside a benchmark. vLLM for continuous batching,
paged KV cache, and an OpenAI-compatible endpoint that the gateway can proxy
without a client library.

## AWQ checkpoint: `Qwen/Qwen3-8B-AWQ`, not `pytorch/Qwen3-8B-AWQ-INT4`

The PyTorch checkpoint is named AWQ but its `config.json` reports
`quant_method: torchao` — int4 weight-only quantization with an AWQ scale search,
which is a different thing. Serving it needs nightly vLLM *and* nightly torchao
from the cu128 index, with `VLLM_DISABLE_COMPILE_CACHE=1`.

Building on nightly wheels for a week-long project is a bad trade, and calling it
"AWQ" in writing would be wrong. `Qwen/Qwen3-8B-AWQ` is first-party, genuinely
AWQ-GEMM at 4 bits with group size 128, ships safetensors, and loads on stable
vLLM with no flags — it picks `MarlinLinearKernel` off `config.json` unaided.

The torchao checkpoint is still interesting as a third data point (torchao-int4
vs autoawq on the same model), just not on the critical path.

## L40S over an A10

48 GB rather than 24 leaves real KV-cache headroom, which is the difference
between measuring the serving system and measuring the card's memory ceiling
during a concurrency sweep. It is also cheaper per hour than the A10 on the
provider used here.

Availability is the catch: L40S is thin in every datacenter that stocks it. The
volume can only be attached at pod creation and cannot move afterwards, so it
goes in the one datacenter that also stocks A100 SXM — a fallback that exists
beats a cheaper one that doesn't.

## Everything on the rented card

An earlier arrangement split development onto a local RTX 4070 and kept the
rented GPU for measurement only. That ended when the machine stopped being
reachable, and the single-host version is simpler in ways worth keeping: BF16
fits in 48 GB, so the BF16-vs-AWQ comparison is available throughout rather than
only at the end, and the sampler divergence below stops existing.

The data center is **EU-NL-1** — the only one currently offering both L40S and
network volumes. A volume cannot move between data centers, so the pod follows
it. Round-trip latency from the US is around 160 ms; the benchmark driver runs on
the pod against `localhost`, so measurements never cross the network.

**Only the network volume survives a stop.** Apt packages, `/root` and `/tmp` are
wiped between sessions, so the Postgres cluster lives on the volume at
`/workspace/pgdata` and only the binaries are reinstalled each time.

### The reasoning from the two-host period

Kept because it governs any future split. Local development used an RTX 4070. BF16 weights are 16.4 GB, so the bf16 model
cannot load there at all; local serving is the AWQ checkpoint, which leaves about
3 GB for KV cache — 22,352 tokens, a maximum concurrency of 2.73x at 8192 tokens
per request.

That is fine for building and useless for measuring: a throughput-vs-concurrency
curve on that card describes its KV ceiling, not the serving system. **No
performance number measured locally goes into `results.json`.** Quality metrics
are hardware-independent and can be measured anywhere the model is identical.

Two configuration facts differ between the hosts, so every results row records
them:

- **Sampler.** WSL2 ships the CUDA driver but no toolkit, and FlashInfer
  JIT-compiles its sampling kernels against `nvcc`. With no `/usr/local/cuda`
  they cannot build, so local runs use the native torch sampler
  (`VLLM_USE_FLASHINFER_SAMPLER=0`). The rented card has a toolkit and uses
  FlashInfer. Immaterial at temperature 0; still a real divergence.
- **Quantization kernel.** Confirmed as `awq_marlin` locally; re-confirmed on the
  rented card rather than assumed to carry over.

## Postgres installed natively, not through docker-compose

The GPU host is itself a container with no Docker daemon, and Compose is not
available there. The original plan brought Postgres up with `docker compose up`,
which would have failed on the first run. `scripts/setup_postgres.sh` installs
PostgreSQL and pgvector from the distro where it carries them and falls back to
the PGDG repo otherwise — the distro path matters, since PGDG does not publish
for every release codename.

`docker-compose.yml` stays as the packaging target, which is where it was always
going to earn its place.

## arXiv pagination caps at 10,000

`start >= 10000` returns HTTP 500 permanently for any single query, regardless of
what `totalResults` claims — 53,746 papers match the corpus query, and the API
will not paginate past 10k of them. Bisected: 9,800 returns 200, 10,000 returns
500.

The ingest partitions on `submittedDate` into windows and pages each one
separately. Windows are disjoint and the `arxiv_id` primary key dedups anything
that overlaps, so runs are additive and a top-up needs no bookkeeping.

Worth noting how this fails: the query is valid, the first 49 pages return 200,
and the wall arrives after about three minutes of rate-limited fetching. Every
cheap check passes first. A test now asserts no offset ever reaches the cap.

## Ingest writes per page, not at the end

The first version accumulated all 60 pages in memory before writing anything, so
a single transient 503 discarded three minutes of rate-limited work and left the
database empty. Writing per page means a failure costs one page and a re-run
resumes.

Retries are 5xx only, with exponential backoff. A malformed query should fail on
the first attempt rather than the sixteenth.

## Chunking: 1800 characters — and the reversal

The corpus is arXiv abstracts, and abstracts are short. Measured across 10,000
papers at an 1800-character budget: 8,009 produce one chunk and 1,991 produce
two, never more. Mean 1.199.

I briefly raised this to 2400 on the theory that the longest abstracts split into
a real chunk plus a near-empty stub, which would put a contentless vector in the
index. The measured distribution disproved it — the 240-character overlap floor
means a second chunk is never a stub — so it went back to 1800, the value the
corpus was actually built with.

**Retrieval granularity is therefore close to one abstract per vector.** That is
a property of a corpus of abstracts, not a tuning choice: the Atom API returns no
full text. Passage-level retrieval would need PDF or LaTeX source, which is a
different ingest path rather than a parameter change.

## Re-chunking deletes before it inserts

Chunks are keyed `(arxiv_id, ord)`. Re-chunking a paper into *fewer* chunks than
before updates `ord = 0` and never touches `ord = 1`, leaving an orphan carrying
stale text and a stale embedding — invisible to every row count, and still
scoring against queries. `store_page` clears a paper's chunks before writing, in
the same transaction.

## Golden set: generated, then verified by hand

Candidates are drawn under a fixed `setseed`, so the same corpus reproduces the
same sample. Each candidate is auto-dropped before review if the question quotes
five or more consecutive words of the title (answerable by string match, which
flatters retrieval), falls outside 8–45 words, near-duplicates an accepted
question, or carries an answer too short to check.

Everything surviving that is read and kept, edited, or dropped by hand. Automated
judges are calibrated against this set, so it is the one part that cannot be
generated and trusted.

## Recall is scored both per chunk and per paper

Every evaluation reports both. Chunk-level is the stricter measure: only the
exact chunk the question was drawn from counts. Paper-level credits any chunk of
the right paper, which matters because 1,991 of 10,000 papers hold two chunks —
retrieving the second half of the correct abstract is a hit by any reasonable
reading, and scoring it as a miss would understate the retriever by an amount
that varies with how long the abstracts happen to be.

Neither is "the" number. Reporting one without the other invites the obvious
question about which was chosen and why, so both go in the output.

A related note lives in the metrics module: with one relevant document per
question, **precision@k cannot exceed 1/k**. A precision@5 of 0.2 is arithmetic,
not a verdict on the retriever, so recall and MRR carry the signal.

## Pods must pin a CUDA floor

A pod's host driver is whatever the scheduler happens to give you, and the
wheels in `uv.lock` need a recent one. One draw came up with driver 550.163.01
(CUDA 12.4) and torch could not see the GPU at all — `torch.cuda.is_available()`
false, vLLM's engine core dead on arrival with a traceback that names none of
this. An earlier draw on the same GPU type got CUDA 13.0 and worked.

So pod creation passes `gpu.minCudaVersion: "12.8"`. The cost is narrower
capacity — the floor took community cloud from available to empty, forcing the
more expensive secure tier — but a pod that cannot run the stack is worth less
than one that costs more.

Related: host-local persistent storage pins a pod to one machine. When that
machine's GPUs filled up, the stopped pod would not restart at all
("not enough free GPUs on the host machine"), and its disk had to be abandoned.
Network volumes avoid this, but no data center currently offers both L40S and
network volumes. Everything on `/workspace` is therefore treated as rebuildable,
and `scripts/` rebuilds it unattended in about fifteen minutes.

## Open

**The benchmark sweep driver.** `vllm bench serve` produces the raw samples, but
its output schema has not been read against a running server yet, and inventing
the key names would be the kind of untested glue that fails mid-sweep. The
statistics and the recorder underneath it are written and tested; the subprocess
layer waits for a machine.
