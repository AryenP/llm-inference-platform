import argparse
import pathlib

import psycopg
from pgvector.psycopg import register_vector

from app.arxiv import Paper, search
from app.chunking import chunk
from app.settings import settings

# cs.LG and cs.DC narrowed to serving-adjacent work; broad enough for 12k papers,
# narrow enough that the golden set can be answered from the corpus
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
    conn.commit()


def embed_and_store(conn, papers: list[Paper], batch: int = 64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)

    rows = [
        (p.arxiv_id, i, text)
        for p in papers
        for i, text in enumerate(chunk(f"{p.title}. {p.abstract}"))
    ]

    for i in range(0, len(rows), batch):
        window = rows[i : i + batch]
        vectors = model.encode(
            [r[2] for r in window], normalize_embeddings=True, show_progress_bar=False
        )
        with conn.cursor() as cur:
            cur.executemany(
                """insert into chunks (arxiv_id, ord, text, embedding)
                   values (%s, %s, %s, %s)
                   on conflict (arxiv_id, ord) do update set
                     text = excluded.text, embedding = excluded.embedding""",
                [(a, o, t, v) for (a, o, t), v in zip(window, vectors, strict=True)],
            )
        conn.commit()
        print(f"  embedded {min(i + batch, len(rows))}/{len(rows)} chunks", flush=True)

    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--skip-embed", action="store_true", help="fetch and store metadata only")
    args = ap.parse_args()

    if not settings.embed_model and not args.skip_embed:
        raise SystemExit("EMBED_MODEL is unset in .env")

    with psycopg.connect(settings.database_url) as conn:
        apply_schema(conn)
        register_vector(conn)

        print(f"fetching up to {args.limit} papers", flush=True)
        papers = search(args.query, args.limit)
        print(f"got {len(papers)}", flush=True)

        upsert_papers(conn, papers)
        n = 0 if args.skip_embed else embed_and_store(conn, papers)

        counts = conn.execute(
            "select (select count(*) from papers), (select count(*) from chunks)"
        ).fetchone()
        print(f"papers={counts[0]} chunks={counts[1]} embedded_this_run={n}")


if __name__ == "__main__":
    main()
