import time

from app.retrieval import Hit, dense, lexical, papers, rrf

PROMPT = """Answer the question using only the context below. If the context does
not contain the answer, say so rather than guessing.

CONTEXT
{context}

QUESTION
{question}

ANSWER"""

MODES = ("dense", "lexical", "hybrid")


def retrieve(conn, mode: str, question: str, embedding, k: int) -> list[Hit]:
    if mode == "lexical" or embedding is None:
        return lexical(conn, question, k)
    if mode == "dense":
        return dense(conn, embedding, k)
    return rrf([dense(conn, embedding, k), lexical(conn, question, k)])


def contexts(conn, hits: list[Hit]) -> list[tuple[str, str]]:
    if not hits:
        return []
    ids = [h.chunk_id for h in hits]
    rows = conn.execute(
        "select id, arxiv_id, text from chunks where id = any(%s)", (ids,)
    ).fetchall()
    # the query returns rows in whatever order the planner likes; retrieval rank
    # is the thing that matters, so restore it
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def build_prompt(question: str, blocks: list[tuple[str, str]]) -> str:
    joined = "\n\n".join(f"[{arxiv_id}] {text}" for arxiv_id, text in blocks)
    return PROMPT.format(context=joined, question=question)


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def cite(hits: list[Hit]) -> list[str]:
    return papers(hits)
