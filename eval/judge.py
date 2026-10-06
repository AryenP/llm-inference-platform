import asyncio
from collections.abc import Sequence

from app.settings import settings


def build_llm(model: str | None = None):
    # ragas speaks OpenAI; vLLM serves an OpenAI-compatible endpoint, so the judge
    # runs on the same model being evaluated rather than a paid external API
    import openai
    from ragas.llms import llm_factory

    # async: ragas scores through ascore(), which calls agenerate() and refuses a
    # synchronous client with a TypeError that surfaces as every question failing
    client = openai.AsyncOpenAI(base_url=settings.vllm_url, api_key="not-used")
    return llm_factory(model or settings.model, provider="openai", client=client)


async def _score_one(metric, item: dict, contexts: Sequence[str], needs_contexts: bool):
    try:
        if needs_contexts:
            result = await metric.ascore(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=list(contexts),
            )
        else:
            result = await metric.ascore(user_input=item["question"], response=item["answer"])
        return float(result.value)
    except Exception as exc:  # noqa: BLE001 - a judge fails in many ways (timeout,
        # unparseable output, rate limit); one bad question must not lose a 150-question run
        return exc


# Each judgement is several LLM calls. Firing 150 at once buries the server and
# every one comes back as a timeout, which the per-item catch then hides as
# "n_failed: 150" with no clue why.
MAX_IN_FLIGHT = 8


async def _score_all(metric, items, contexts_by_id, needs_contexts, limit=MAX_IN_FLIGHT):
    gate = asyncio.Semaphore(limit)

    async def one(it):
        async with gate:
            return await _score_one(metric, it, contexts_by_id.get(it["cid"], []), needs_contexts)

    return await asyncio.gather(*(one(it) for it in items))


def summarise(scores: Sequence) -> dict:
    ok = [s for s in scores if isinstance(s, float)]
    bad = [s for s in scores if not isinstance(s, float)]
    # a mean over the scored subset, with the unscored count beside it — averaging
    # failures in as zeros would understate the metric and hide that they happened.
    # The first error travels with it: "n_failed: 150" alone says nothing about why.
    out = {
        "n_scored": len(ok),
        "n_failed": len(bad),
        "mean": sum(ok) / len(ok) if ok else None,
    }
    if bad:
        out["first_error"] = f"{type(bad[0]).__name__}: {str(bad[0])[:300]}"
    return out


def faithfulness(items, contexts_by_id, llm=None) -> dict:
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=llm or build_llm())
    return summarise(asyncio.run(_score_all(metric, items, contexts_by_id, True)))
