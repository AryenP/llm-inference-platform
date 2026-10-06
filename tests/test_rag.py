import pytest

from app.rag import build_prompt, contexts, retrieve
from app.retrieval import Hit


class FakeConn:
    def __init__(self, rows, lexical_rows=None):
        self.rows, self.lexical_rows = rows, lexical_rows or []
        self.queries = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if "where id = any" in sql:
            return _R(self.rows)
        return _R(self.lexical_rows if "tsv" in sql else [(h.chunk_id, h.arxiv_id) for h in HITS])


class _R:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


HITS = [Hit(3, "p3"), Hit(1, "p1"), Hit(2, "p2")]


def test_contexts_restore_retrieval_order():
    # the planner returns rows in its own order; rank is what matters
    conn = FakeConn([(1, "p1", "first"), (2, "p2", "second"), (3, "p3", "third")])

    got = contexts(conn, HITS)

    assert got == [("p3", "third"), ("p1", "first"), ("p2", "second")]


def test_contexts_skip_rows_the_database_no_longer_has():
    conn = FakeConn([(3, "p3", "third")])

    assert contexts(conn, HITS) == [("p3", "third")]


def test_contexts_of_no_hits_queries_nothing():
    conn = FakeConn([])

    assert contexts(conn, []) == []
    assert conn.queries == []


def test_prompt_labels_each_block_with_its_paper():
    prompt = build_prompt("why?", [("2401.1", "alpha"), ("2401.2", "beta")])

    assert "[2401.1] alpha" in prompt
    assert "[2401.2] beta" in prompt
    assert "why?" in prompt
    # the instruction that keeps the model from answering from memory
    assert "only the context" in prompt


def test_retrieve_falls_back_to_lexical_without_an_embedding():
    conn = FakeConn([], lexical_rows=[(9, "p9")])

    got = retrieve(conn, "hybrid", "question", None, 5)

    assert got == [Hit(9, "p9")]
    assert any("tsv" in sql for sql, _ in conn.queries)


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_dense_modes_use_the_embedding_when_present(mode):
    conn = FakeConn([])

    retrieve(conn, mode, "question", [0.0] * 4, 5)

    assert any("embedding <=>" in sql for sql, _ in conn.queries)
