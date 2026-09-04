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

## Development on a 12 GB card, measurement on the rented one

Local development uses an RTX 4070. BF16 weights are 16.4 GB, so the bf16 model
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

## Open

**Is recall@k scored per chunk or per paper?** Candidates record both
`arxiv_id` and `chunk_id`. Paper-level looks more honest here — with ~20% of
papers holding two chunks, retrieving the second half of the correct abstract
would otherwise count as a miss — but chunk-level is the stricter measure and
both will be reported.
