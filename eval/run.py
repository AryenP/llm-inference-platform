import argparse
import json
import pathlib

import psycopg
from pgvector.psycopg import register_vector

from app.retrieval import dense, lexical, papers, rerank, rrf
from app.settings import settings
from eval.gate import BASELINE, regressions
from eval.metrics import summarise
from eval.schema import read_jsonl

MODES = ("dense", "lexical", "hybrid", "rerank")


CANDIDATES = 30


def retrieve(conn, mode: str, embedding, question: str, k: int, reranker=None):
    if mode == "dense":
        return dense(conn, embedding, k)
    if mode == "lexical":
        return lexical(conn, question, k)
    fused = rrf([dense(conn, embedding, CANDIDATES), lexical(conn, question, CANDIDATES)])
    if mode == "hybrid":
        return fused[:k]
    texts = dict(
        conn.execute(
            "select id, text from chunks where id = any(%s)", ([h.chunk_id for h in fused],)
        ).fetchall()
    )
    return rerank(reranker, question, fused, texts, k)


def chunk_key(conn, arxiv_id: str, ord_: int) -> str:
    # (arxiv_id, ord) is stable across re-ingests; the bigserial id is not, so a
    # rebuilt corpus would silently score every question as a miss
    return f"{arxiv_id}#{ord_}"


def evaluate(conn, encode, items: list[dict], mode: str, k: int, reranker=None) -> dict:
    by_chunk, by_paper = [], []
    ords = dict(
        conn.execute("select id, arxiv_id || '#' || ord from chunks").fetchall()
    )
    for it in items:
        hits = retrieve(conn, mode, encode(it["question"]), it["question"], k, reranker)
        # scored both ways: chunk-level is stricter, but ~20% of papers hold two
        # chunks, so retrieving the other half of the right abstract would count
        # as a miss there. Paper-level is the honest headline; both are reported.
        gold = chunk_key(conn, it["arxiv_id"], it.get("ord", 0))
        by_chunk.append(([ords.get(h.chunk_id, "") for h in hits], {gold}))
        by_paper.append((papers(hits), {it["arxiv_id"]}))
    return {"chunk": summarise(by_chunk), "paper": summarise(by_paper)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("eval/golden.jsonl"))
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--baseline", type=pathlib.Path, default=pathlib.Path(BASELINE))
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--gate", action="store_true", help="exit non-zero on regression")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()

    items = read_jsonl(args.golden)
    if not items:
        raise SystemExit(f"no questions in {args.golden} — run ./init.sh review first")
    if not settings.embed_model:
        raise SystemExit("EMBED_MODEL is unset in .env")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)

    def encode(text: str):
        return model.encode(text, normalize_embeddings=True)

    reranker = None
    if args.mode == "rerank":
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(settings.rerank_model)

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        scores = evaluate(conn, encode, items, args.mode, args.k, reranker)

    report = {"mode": args.mode, "k": args.k, **scores}
    print(json.dumps(report, indent=2))

    if args.save_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report, indent=2) + "\n")
        print(f"baseline written to {args.baseline}")
        return

    if args.gate:
        was = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
        if not was:
            raise SystemExit(f"no baseline at {args.baseline} — record one with --save-baseline")
        bad = regressions(scores["paper"], was.get("paper", {}), args.tolerance)
        bad += regressions(scores["chunk"], was.get("chunk", {}), args.tolerance)
        if bad:
            print("\nregressed:")
            for line in bad:
                print(f"  {line}")
            raise SystemExit(1)
        print("\nno regression past threshold")


if __name__ == "__main__":
    main()
