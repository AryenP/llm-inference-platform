import argparse
import json
import pathlib

import httpx
import psycopg
from pgvector.psycopg import register_vector

from app.rag import build_prompt, contexts
from app.settings import settings
from eval.judge import build_llm, faithfulness
from eval.run import MODES, retrieve
from eval.schema import read_jsonl


def generate(client: httpx.Client, model: str, prompt: str, max_tokens: int) -> str:
    r = client.post(
        "/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("eval/golden.jsonl"))
    ap.add_argument("--mode", choices=MODES, default="dense")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    items = read_jsonl(args.golden)
    if args.limit:
        items = items[: args.limit]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(settings.embed_model)
    reranker = None
    if args.mode == "rerank":
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(settings.rerank_model)

    answered, ctx_by_id = [], {}
    with (
        psycopg.connect(settings.database_url) as conn,
        httpx.Client(base_url=settings.vllm_url, timeout=300) as client,
    ):
        register_vector(conn)
        for i, it in enumerate(items, 1):
            emb = embedder.encode(it["question"], normalize_embeddings=True)
            hits = retrieve(conn, args.mode, emb, it["question"], args.k, reranker)
            blocks = contexts(conn, hits)
            if not blocks:
                continue
            answer = generate(
                client, settings.model, build_prompt(it["question"], blocks), args.max_tokens
            )
            # the judge scores the generated answer, not the reference one
            answered.append({"cid": it["cid"], "question": it["question"], "answer": answer})
            ctx_by_id[it["cid"]] = [text for _, text in blocks]
            if i % 25 == 0:
                print(f"  answered {i}/{len(items)}", flush=True)

    # the retrieval models are done; hand their VRAM back before the judge runs
    del embedder, reranker
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    scores = faithfulness(answered, ctx_by_id, llm=build_llm())
    report = {"mode": args.mode, "k": args.k, "n_answered": len(answered), "faithfulness": scores}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
