from collections.abc import Iterable, Sequence
from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    arxiv_id: str


def dense(conn, embedding, k: int) -> list[Hit]:
    rows = conn.execute(
        "select id, arxiv_id from chunks order by embedding <=> %s limit %s",
        (embedding, k),
    ).fetchall()
    return [Hit(r[0], r[1]) for r in rows]


def lexical(conn, query: str, k: int) -> list[Hit]:
    # Postgres FTS standing in for BM25: ts_rank_cd with the default weights is
    # close enough in ordering, and keeping it in the database avoids holding a
    # second index of the corpus in the application.
    #
    # The terms are OR'd, not AND'd. plainto_tsquery ANDs everything, so a
    # natural-language question demands that one abstract contain every word in
    # it — which matched almost nothing and made lexical recall look like 0.04.
    # Routing the question through to_tsvector first lexemises and deduplicates
    # it safely, then the lexemes are OR'd and ts_rank_cd does the ordering.
    rows = conn.execute(
        """select c.id, c.arxiv_id
             from chunks c,
                  to_tsquery('english',
                      array_to_string(
                          tsvector_to_array(to_tsvector('english', %s)), ' | ')) q
            where c.tsv @@ q
            order by ts_rank_cd(c.tsv, q) desc
            limit %s""",
        (query, k),
    ).fetchall()
    return [Hit(r[0], r[1]) for r in rows]


def rrf(rankings: Sequence[Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    # reciprocal rank fusion: a document's score is the sum of 1/(k + rank) over
    # the lists it appears in, so a result ranked well by either retriever
    # surfaces without the two scores needing a shared scale
    scores: dict[Hit, float] = {}
    for ranked in rankings:
        for rank, hit in enumerate(ranked, 1):
            scores[hit] = scores.get(hit, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda h: (-scores[h], h.chunk_id))


def papers(hits: Iterable[Hit]) -> list[str]:
    # one paper can hold several chunks; keep its best rank and drop the rest
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h.arxiv_id, None)
    return list(seen)


def rerank(model, query: str, hits: list[Hit], texts: dict[int, str], k: int) -> list[Hit]:
    # A cross-encoder scores the query against each candidate jointly, which the
    # bi-encoder cannot: it never sees the pair. The cost is that it runs over
    # candidates rather than over the index, so it only reorders what retrieval
    # already found — recall@k for k at or above the candidate count cannot improve.
    scored = [(h, texts[h.chunk_id]) for h in hits if h.chunk_id in texts]
    if not scored:
        return hits[:k]
    scores = model.predict([(query, t) for _, t in scored])
    order = sorted(zip(scored, scores, strict=True), key=lambda x: -x[1])
    return [h for (h, _), _ in order][:k]
