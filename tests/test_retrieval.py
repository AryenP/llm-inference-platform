from app.retrieval import Hit, papers, rrf


def h(n, paper=None):
    return Hit(n, paper or f"p{n}")


def test_rrf_ranks_a_document_found_by_both_retrievers_first():
    dense = [h(1), h(2), h(3)]
    lexical = [h(3), h(1), h(9)]

    got = rrf([dense, lexical])

    # 1 is 1st and 2nd; 3 is 3rd and 1st — 1 wins on the sum
    assert got[0] == h(1)
    assert set(got) == {h(1), h(2), h(3), h(9)}


def test_rrf_keeps_a_document_only_one_retriever_found():
    got = rrf([[h(1)], [h(2)]])

    assert set(got) == {h(1), h(2)}


def test_rrf_is_stable_for_equal_scores():
    # same rank in both lists, so the tiebreak is chunk id rather than dict order
    assert rrf([[h(5), h(2)], [h(5), h(2)]]) == [h(5), h(2)]


def test_rrf_of_nothing_is_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_agreement_between_retrievers_beats_a_single_top_rank():
    long_list = [h(i) for i in range(1, 20)]

    got = rrf([long_list, [h(19)]])

    # 19 sits 19th in one list and 1st in the other: 1/79 + 1/61 = 0.029, against
    # 1's single 1/61 = 0.016. Being found twice outranks being found once at the
    # top, which is the whole reason to fuse rather than concatenate
    assert got[0] == h(19)
    assert got[1] == h(1)


def test_rrf_discounts_later_ranks_within_one_list():
    got = rrf([[h(1), h(2), h(3)]])

    assert got == [h(1), h(2), h(3)]


def test_papers_collapses_chunks_and_keeps_the_best_rank():
    hits = [Hit(1, "a"), Hit(2, "b"), Hit(3, "a"), Hit(4, "c")]

    assert papers(hits) == ["a", "b", "c"]


def test_papers_of_nothing_is_empty():
    assert papers([]) == []


class FakeCross:
    """Scores a pair higher the more words the text shares with the query."""

    def predict(self, pairs):
        return [len(set(q.split()) & set(t.split())) for q, t in pairs]


def test_rerank_reorders_by_cross_encoder_score():
    from app.retrieval import rerank

    hits = [h(1), h(2), h(3)]
    texts = {1: "nothing alike", 2: "kv cache waste", 3: "cache"}

    got = rerank(FakeCross(), "kv cache waste", hits, texts, k=3)

    assert got == [h(2), h(3), h(1)]


def test_rerank_truncates_to_k():
    from app.retrieval import rerank

    hits = [h(1), h(2), h(3)]
    texts = {1: "a", 2: "a b", 3: "a b c"}

    assert len(rerank(FakeCross(), "a b c", hits, texts, k=2)) == 2


def test_rerank_skips_hits_with_no_text_available():
    from app.retrieval import rerank

    got = rerank(FakeCross(), "q", [h(1), h(2)], {2: "q"}, k=5)

    assert got == [h(2)]


def test_rerank_falls_back_when_no_text_at_all():
    from app.retrieval import rerank

    # nothing scoreable: return the retrieval order rather than nothing
    assert rerank(FakeCross(), "q", [h(1), h(2)], {}, k=1) == [h(1)]
