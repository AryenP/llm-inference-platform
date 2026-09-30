import math
from collections.abc import Sequence


def hits_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> int:
    return sum(1 for d in ranked[:k] if d in relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    # With one relevant document per question — the shape of this golden set —
    # precision@k cannot exceed 1/k, so it says more about k than about the
    # retriever. Reported for completeness; recall and MRR carry the signal.
    if k <= 0:
        return 0.0
    return hits_at_k(ranked, relevant, k) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return hits_at_k(ranked, relevant, k) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def summarise(rankings: list[tuple[Sequence[str], set[str]]], ks=(1, 5, 10)) -> dict:
    n = len(rankings)
    if not n:
        return {}
    out = {"n": n, "mrr": sum(reciprocal_rank(r, g) for r, g in rankings) / n}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, g, k) for r, g in rankings) / n
        out[f"precision@{k}"] = sum(precision_at_k(r, g, k) for r, g in rankings) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, g, k) for r, g in rankings) / n
    return out
