import argparse
import pathlib

import psycopg
from pgvector.psycopg import register_vector

from app.arxiv import Paper, pages
from app.chunking import chunk
from app.settings import settings

# cs.LG and cs.DC narrowed to serving-adjacent work; broad enough for a corpus in
# the tens of thousands, narrow enough that the golden set stays answerable
QUERY = (
    "(cat:cs.LG OR cat:cs.DC) AND "
    "(abs:inference OR abs:quantization OR abs:serving OR abs:latency OR abs:throughput)"
)

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "sql" / "002_schema.sql"


def apply_schema(conn):
    conn.execute(SCHEMA.read_text())
    conn.commit()


def upsert_papers(conn, papers: list[Paper]):
    with conn.cursor() as cur:
        cur.executemany(
            """insert into papers (arxiv_id, title, abstract, categories, published, updated)
               values (%s, %s, %s, %s, %s, %s)
               on conflict (arxiv_id) do update set
                 title = excluded.title,
                 abstract = excluded.abstract,
                 updated = excluded.updated""",
            [
                (p.arxiv_id, p.title, p.abstract, p.categories, p.published, p.updated)
                for p in papers
            ],
        )


def store_page(conn, papers: list[Paper], model=None) -> int:
    upsert_papers(conn, papers)

    rows = [
        (p.arxiv_id, i, text)
        for p in papers
        for i, text in enumerate(chunk(f"{p.title}. {p.abstract}"))
    ]
    # re-chunking a paper can produce fewer chunks than last time; without this
    # the old high-ord rows survive as orphans the upsert never touches
    with conn.cursor() as cur:
        cur.execute("delete from chunks where arxiv_id = any(%s)", ([p.arxiv_id for p in papers],))

    if rows:
        # chunks are written even without embeddings, so --skip-embed still
        # exercises the (arxiv_id, ord) conflict — the one carrying the volume
        vectors = (
            model.encode([r[2] for r in rows], normalize_embeddings=True, show_progress_bar=False)
            if model
            else [None] * len(rows)
        )
        with conn.cursor() as cur:
            cur.executemany(
                """insert into chunks (arxiv_id, ord, text, embedding)
                   values (%s, %s, %s, %s)
                   on conflict (arxiv_id, ord) do update set
                     text = excluded.text, embedding = excluded.embedding""",
                [(a, o, t, v) for (a, o, t), v in zip(rows, vectors, strict=True)],
            )

    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--skip-embed", action="store_true", help="store chunks with null embeddings")
    args = ap.parse_args()

    model = None
    if not args.skip_embed:
        if not settings.embed_model:
            raise SystemExit("EMBED_MODEL is unset in .env")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embed_model)

    with psycopg.connect(settings.database_url) as conn:
        apply_schema(conn)
        register_vector(conn)

        papers = chunks = 0
        # written per page: a mid-run failure costs one page, not the whole fetch,
        # and a re-run resumes rather than starting over
        for page in pages(args.query, args.limit, window_days=args.window_days):
            papers += len(page)
            chunks += store_page(conn, page, model)
            print(f"  {papers} papers, {chunks} chunks", flush=True)

        totals = conn.execute(
            "select (select count(*) from papers), (select count(*) from chunks)"
        ).fetchone()
        print(
            f"papers={totals[0]} chunks={totals[1]} embedded_this_run={0 if model is None else chunks}"
        )


if __name__ == "__main__":
    main()
