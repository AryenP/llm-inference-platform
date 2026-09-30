from eval.run import evaluate

# (chunk_id, arxiv_id) rows, as both retrieval queries return them
DENSE = [(11, "p1"), (22, "p2"), (33, "p3")]
LEXICAL = [(33, "p3"), (44, "p4")]


class FakeConn:
    def __init__(self, dense_rows, lexical_rows):
        self.dense_rows, self.lexical_rows = dense_rows, lexical_rows
        self.queries = []

    def execute(self, sql, params=None):
        self.queries.append(sql)
        rows = self.lexical_rows if "tsv" in sql else self.dense_rows
        return _Result(rows)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def encode(text):
    return [0.0] * 4


def item(chunk_id, arxiv_id):
    return {"question": "what reduces kv cache waste", "chunk_id": chunk_id, "arxiv_id": arxiv_id}


def test_hybrid_scores_both_granularities():
    conn = FakeConn(DENSE, LEXICAL)

    got = evaluate(conn, encode, [item(11, "p1")], "hybrid", k=10)

    assert got["chunk"]["n"] == 1
    assert got["paper"]["n"] == 1
    assert got["chunk"]["recall@5"] == 1.0
    assert got["paper"]["recall@5"] == 1.0


def test_paper_level_credits_the_other_chunk_of_the_right_paper():
    conn = FakeConn(DENSE, [])
    # gold is chunk 99, which was never retrieved — but it belongs to p2, and
    # chunk 22 of p2 was. Chunk-level misses; paper-level counts it.
    got = evaluate(conn, encode, [item(99, "p2")], "dense", k=10)

    assert got["chunk"]["recall@5"] == 0.0
    assert got["paper"]["recall@5"] == 1.0


def test_dense_mode_never_runs_the_lexical_query():
    conn = FakeConn(DENSE, LEXICAL)

    evaluate(conn, encode, [item(11, "p1")], "dense", k=10)

    assert not any("tsv" in q for q in conn.queries)


def test_lexical_mode_never_runs_the_vector_query():
    conn = FakeConn(DENSE, LEXICAL)

    evaluate(conn, encode, [item(33, "p3")], "lexical", k=10)

    assert all("tsv" in q for q in conn.queries)


def test_a_total_miss_scores_zero_everywhere():
    conn = FakeConn(DENSE, LEXICAL)

    got = evaluate(conn, encode, [item(777, "nope")], "hybrid", k=10)

    assert got["chunk"]["recall@5"] == 0.0
    assert got["paper"]["recall@5"] == 0.0
    assert got["chunk"]["mrr"] == 0.0
