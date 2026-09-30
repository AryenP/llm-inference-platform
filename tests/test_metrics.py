import math

from eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_is_a_hit_rate_when_one_document_is_relevant():
    assert recall_at_k(RANKED, {"c"}, 5) == 1.0
    assert recall_at_k(RANKED, {"c"}, 2) == 0.0
    assert recall_at_k(RANKED, {"zzz"}, 5) == 0.0


def test_recall_counts_partial_coverage():
    assert recall_at_k(RANKED, {"a", "c", "zzz"}, 5) == 2 / 3


def test_precision_is_bounded_by_one_over_k_with_a_single_gold():
    # the reason precision@k is reported but not leaned on
    assert precision_at_k(RANKED, {"a"}, 5) == 0.2
    assert precision_at_k(RANKED, {"a"}, 1) == 1.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c"}) == 1 / 3
    assert reciprocal_rank(RANKED, {"c", "e"}) == 1 / 3
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_ndcg_is_one_when_the_gold_is_first():
    assert ndcg_at_k(RANKED, {"a"}, 5) == 1.0


def test_ndcg_discounts_by_position():
    assert ndcg_at_k(RANKED, {"b"}, 5) == 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"c"}, 5) < ndcg_at_k(RANKED, {"b"}, 5)


def test_ndcg_normalises_against_the_best_possible_ordering():
    # two relevant documents, retrieved first and second — a perfect ranking
    assert ndcg_at_k(RANKED, {"a", "b"}, 5) == 1.0


def test_metrics_are_zero_when_nothing_is_relevant():
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0
    assert precision_at_k(RANKED, {"a"}, 0) == 0.0


def test_summarise_averages_across_questions():
    rankings = [(RANKED, {"a"}), (RANKED, {"c"}), (RANKED, {"zzz"})]

    got = summarise(rankings, ks=(1, 5))

    assert got["n"] == 3
    assert got["recall@5"] == 2 / 3
    assert got["recall@1"] == 1 / 3
    assert got["mrr"] == (1.0 + 1 / 3 + 0.0) / 3


def test_summarise_of_nothing_is_empty():
    assert summarise([]) == {}
