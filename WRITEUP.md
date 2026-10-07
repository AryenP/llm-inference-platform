# Measuring a serving system

This project exists to answer four questions about running an 8B model yourself,
with numbers rather than impressions: how fast is it, what does quantization
actually buy, when is self-hosting cheaper than an API, and does better retrieval
produce better answers.

Everything below was measured on one NVIDIA L40S (48GB) running vLLM 0.28.0 with
Qwen3-8B, against a corpus of 12,000 arXiv abstracts. Every figure is in
`results.json` or `eval/results/`, and the scripts that produced them are in the
repo.

## Quantization is the easy win

Serving the 4-bit AWQ checkpoint instead of BF16 improved every metric measured:

| at 1,024 input tokens, 4 req/s | BF16 | AWQ |
|---|---|---|
| TTFT p50 | 164.7 ms | 107.2 ms |
| TTFT p95 | 350.1 ms | 239.9 ms |
| ITL p50 | 28.79 ms | 10.28 ms |

The inter-token latency gap is the larger effect and the more interesting one.
Prefill is compute-bound, so quantization helps it modestly. Decode is bound by
memory bandwidth — every token requires reading the full weight matrix — so
cutting the weights to a quarter of their size cuts the dominant cost directly.
That is why ITL improves by 2.8x while TTFT improves by 1.5x, and it is the
reason reporting a single averaged "latency" number would have hidden the
mechanism entirely.

At saturation AWQ sustained 7.62 req/s against BF16's 6.31.

## Self-hosting has a break-even point

The cost question has a more useful answer than "cheaper". Priced against hosted
Qwen3-8B at published rates (read 6 October 2026), for 1,000 queries of 1,024
input and 128 output tokens:

| | per 1,000 queries |
|---|---|
| Cheapest hosted API | $0.178 |
| Self-hosted AWQ, concurrency 64 | $0.040 |
| Self-hosted AWQ, concurrency 1 | $0.350 |

Self-hosting is 4.5x cheaper at saturation and roughly twice as expensive at one
request at a time, because an idle GPU bills exactly like a busy one. The
break-even is **1.70 requests/sec sustained** at the $1.09/hr rate paid.

That crossover is the number that actually decides anything. Below it, the API is
cheaper and someone else carries the operational burden. The 4.5x figure alone
would be true and misleading.

## The tail, not the average

At 64 concurrent requests, TTFT p95 reached ~4.9 seconds on both models while p50
stayed near 720 ms. That gap is queueing, not computation, and it is why latency
and throughput were run as separate experiments: a low-rate pass measures what
the model costs, a saturation pass measures what the scheduler does under
pressure, and one number averaged across both describes neither.

## A latency curve that was measuring the queue

TTFT appeared to grow superlinearly with prompt length: roughly 4.6x between
1,024 and 2,048 tokens at a 4 req/s arrival rate. Prefill is O(n) per layer, so
that needed explaining rather than publishing.

Two controls settled it. Disabling chunked prefill changed nothing, which ruled
out the scheduler. Dropping the arrival rate to 1 req/s removed the bend
entirely: per-token cost normalised to the 1,024-token baseline ran 1.00, 0.88,
0.87, 0.81, 0.81 across 1,024 to 3,072 tokens — flat, slightly declining, since
longer prefills use the GPU more efficiently. At 4 req/s the same series ran
1.00, 0.99, 1.03, 1.50, 1.83, 2.66.

At 3,072 tokens the identical work took 340 ms unqueued and 1,210 ms at 4 req/s.
The curve was measuring the queue, not the model. This is precisely the failure
the separate latency and throughput passes exist to prevent, and it still
happened — the low-rate pass was low enough for short prompts and not for long
ones.

## Retrieval improved a lot; the standard metric could not see it

Retrieval was evaluated against 150 questions over the 12,000-abstract corpus.
Adding a cross-encoder reranker over reciprocal-rank-fused BM25 and dense
candidates produced a large improvement:

| | dense | reranked |
|---|---|---|
| recall@1 | 0.527 | 0.753 |
| recall@5 | 0.700 | 0.827 |
| MRR | 0.602 | 0.788 |
| answer faithfulness | 0.948 | 0.957 |

A 43% relative gain in recall@1 moved answer faithfulness by 0.009.

The explanation matters more than the result. **Faithfulness measures grounding,
not correctness.** It asks whether the answer follows from the passages the model
was given — so an answer built faithfully from the wrong paper scores just as
well as one built from the right paper. The metric is close to blind to exactly
what reranking fixed.

Scoring the generated answer against the reference answer instead shows the
effect the faithfulness number hides. Answers scored **0.795** against the reference when the correct
paper was retrieved and **0.593** when it was not — a gap of 0.20 across 150
questions. Retrieval quality clearly does
drive answer quality; faithfulness just is not the instrument that detects it.

Running both arms and splitting by whether the gold paper was retrieved shows
where the gain lives. Reranking moved 19 questions into the retrieved bucket
(105 to 124 of 150) and lifted overall similarity from 0.734 to 0.763. But the
conditional means barely changed: 0.795 to 0.797 when the right paper was found,
0.593 to 0.602 when it was not. Decomposed, **89% of the answer-quality gain is
the mix shift and 11% is better answering**. The reranker did not make the model
a better writer. It changed which paper the model wrote from.

This reproduces the finding in *"Retrieval Improvements Do Not Guarantee Better
Answers"* on a different corpus, with a specific mechanism rather than a general
caution: the retrieval metric and the answer metric measure different things, and
optimising one does not move the other unless the second metric is sensitive to
the first.

## What went wrong

Four bugs are worth recording because each produced plausible-looking numbers
before being caught.

**Lexical retrieval scored 0.040, flat across every k.** Flat across k is a bug
signature — a retriever returning nothing, rather than returning wrong things.
`plainto_tsquery` ANDs every term, so a natural-language question required one
abstract to contain all of its words. OR-ing the lexemes took recall@5 from 0.040
to 0.500. Left unnoticed, "hybrid beats lexical by 12x" would have been published
as a finding about retrieval when it was a finding about a broken query.

**The evaluation ran against a corpus missing 82% of its gold documents.** The
golden set keyed on `chunk_id`, a Postgres `bigserial` that does not survive a
re-ingest. The harness reported recall of 0.153 without complaining. The gold
reference is now `(arxiv_id, ord)`, which is stable across rebuilds.

**All 150 faithfulness scores failed silently.** The judge was given a
synchronous OpenAI client where ragas calls `agenerate()`, and 150 concurrent
judge calls then buried the server. The summary reported `mean: null` rather than
averaging failures as zeros — had it done the latter, "faithfulness 0.0" would
have read as a catastrophic result instead of a broken judge.

**A stale vLLM engine worker held 42 of 44 GB through a dozen attempts.**
`pkill -f "vllm serve"` kills the API server but not the engine process, whose
command line does not contain "vllm". Every subsequent start failed with
`Engine core initialization failed`, which names neither the stale process nor
the memory.

The common thread: every one of these produced output that looked like a result.
The defence was not care, it was structure — a validator that refuses to record a
run missing its configuration, a summary that distinguishes "zero" from
"unmeasured", and the habit of treating a suspiciously flat curve as a bug report.

## What this does not show

The 150-question evaluation set is **model-generated and automatically screened,
not human-verified**. A model wrote the questions, a model screened them, and a
model judges faithfulness against them, so correlated blind spots are possible
and unmeasured. The screening removed 23% of candidates for naming their own
paper's method — answerable by string match, which would have flattered lexical
retrieval — but the model's own judgement of answerability and grounding rejected
nothing at all. It approved every question it had written. That is the clearest
available argument for why a human pass matters, and it is a limitation of these
numbers rather than a footnote to them.

The corpus is abstracts, not full text, so retrieval granularity is roughly one
abstract per vector and the results say little about passage-level retrieval over
long documents.

All serving numbers come from a single L40S on one provider. They characterise
that configuration, not the model in general.
