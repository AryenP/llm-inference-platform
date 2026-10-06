import argparse
import json
import pathlib
import re

import httpx

from app.settings import settings
from eval.schema import read_jsonl, write_jsonl

VERDICT = """You are screening a question/answer pair built from a single paper abstract.

Judge three things, strictly:
1. answerable — can the question be answered using ONLY this abstract? If it needs
   outside knowledge or refers to content the abstract does not contain, this is false.
2. grounded — is every claim in the answer actually supported by the abstract?
3. specific — does the question describe what it is asking about, rather than
   depending on the reader already knowing this paper's name for its method?

Return only JSON: {"answerable": true/false, "grounded": true/false, "specific": true/false, "why": "one short phrase"}

ABSTRACT
{abstract}

QUESTION
{question}

ANSWER
{answer}"""


def caps(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text))


def distinctive_terms(rows: list[dict], threshold: int = 2) -> dict[str, int]:
    freq: dict[str, int] = {}
    for r in rows:
        for tok in caps(r["title"]):
            freq[tok] = freq.get(tok, 0) + 1
    return {t: n for t, n in freq.items() if n <= threshold}


def leaks_name(row: dict, rare: dict[str, int]) -> set[str]:
    # a question naming its own paper's method is retrievable by exact string
    # match, which flatters lexical retrieval without testing anything
    return {t for t in (caps(row["question"][1:]) & caps(row["title"])) if t in rare}


def judge(client: httpx.Client, model: str, row: dict) -> dict | None:
    prompt = (
        VERDICT.replace("{abstract}", row["source"])
        .replace("{question}", row["question"])
        .replace("{answer}", row["answer"])
    )
    r = client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return parse_verdict(raw)


def parse_verdict(raw: str) -> dict | None:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        got = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not all(k in got for k in ("answerable", "grounded", "specific")):
        return None
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=pathlib.Path, default=pathlib.Path("eval/candidates.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("eval/golden.jsonl"))
    ap.add_argument("--rejected", type=pathlib.Path, default=pathlib.Path("eval/screened_out.jsonl"))
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()

    rows = read_jsonl(args.candidates)
    if not rows:
        raise SystemExit(f"no candidates at {args.candidates}")
    rare = distinctive_terms(rows)

    kept, dropped = [], []
    with httpx.Client(base_url=settings.vllm_url, timeout=180) as client:
        for i, row in enumerate(rows, 1):
            names = leaks_name(row, rare)
            if names:
                dropped.append({**row, "screen": f"names its own method: {sorted(names)}"})
                continue

            verdict = judge(client, settings.model, row)
            if verdict is None:
                dropped.append({**row, "screen": "judge returned no usable verdict"})
                continue
            failed = [k for k in ("answerable", "grounded", "specific") if not verdict.get(k)]
            if failed:
                dropped.append({**row, "screen": f"failed {','.join(failed)}: {verdict.get('why','')}"})
                continue

            kept.append({**row, "screen": "passed"})
            if len(kept) >= args.target:
                break
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} · {len(kept)} kept · {len(dropped)} dropped", flush=True)

    write_jsonl(args.out, kept)
    write_jsonl(args.rejected, dropped)
    print(f"kept {len(kept)} to {args.out}, dropped {len(dropped)} to {args.rejected}")


if __name__ == "__main__":
    main()
