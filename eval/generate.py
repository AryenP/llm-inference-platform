import argparse
import json
import pathlib
import re
import uuid

import httpx
import psycopg

from app.settings import settings
from eval.quality import reasons
from eval.schema import Candidate, write_jsonl

OUT = pathlib.Path("eval/candidates.jsonl")

PROMPT = """You are helping build an evaluation set for a retrieval system over arXiv abstracts.

Read the abstract below and write ONE question a researcher might ask, plus its answer.

Requirements:
- The question must be answerable from this abstract alone, and specific enough that
  this paper is the right answer rather than any paper on the topic.
- Ask about the contribution, method, or result — not about who wrote it or when.
- Do not quote the title. Do not use the paper's exact phrasing for its method name
  as the whole question.
- The answer must be one or two sentences, drawn only from the abstract.

Return only JSON: {"question": "...", "answer": "..."}

ABSTRACT
%s"""


def sample_chunks(conn, n: int, seed: float):
    conn.execute("select setseed(%s)", (seed,))
    return conn.execute(
        """select c.id, c.arxiv_id, c.text, p.title
             from chunks c join papers p using (arxiv_id)
            where c.ord = 0
            order by random() limit %s""",
        (n,),
    ).fetchall()


def parse(raw: str) -> dict | None:
    # qwen3 can emit a reasoning block even with thinking disabled
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        got = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(got, dict) or not got.get("question") or not got.get("answer"):
        return None
    return got


def ask(client: httpx.Client, model: str, text: str) -> dict | None:
    r = client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT % text}],
            "temperature": 0.7,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    r.raise_for_status()
    return parse(r.json()["choices"][0]["message"]["content"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="chunks to sample")
    ap.add_argument("--seed", type=float, default=0.42, help="postgres setseed, for reproducibility")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    kept: list[Candidate] = []
    seen: set[frozenset] = set()
    dropped = 0

    with (
        psycopg.connect(settings.database_url) as conn,
        httpx.Client(base_url=settings.vllm_url, timeout=180) as client,
    ):
        rows = sample_chunks(conn, args.n, args.seed)
        print(f"sampled {len(rows)} chunks", flush=True)

        for i, (chunk_id, arxiv_id, text, title) in enumerate(rows, 1):
            got = ask(client, settings.model, text)
            if not got:
                dropped += 1
                continue

            why = reasons(got["question"], got["answer"], title, seen)
            if why:
                dropped += 1
                continue

            seen.add(frozenset(re.findall(r"[a-z0-9]+", got["question"].lower())))
            kept.append(
                Candidate(
                    cid=uuid.uuid4().hex[:8],
                    question=got["question"].strip(),
                    answer=got["answer"].strip(),
                    arxiv_id=arxiv_id,
                    chunk_id=chunk_id,
                    source=text,
                    title=title,
                )
            )
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} · {len(kept)} candidates · {dropped} auto-dropped", flush=True)

    write_jsonl(args.out, kept)
    print(f"wrote {len(kept)} candidates to {args.out} ({dropped} auto-dropped)")
    print("now review them by hand: ./init.sh review")


if __name__ == "__main__":
    main()
