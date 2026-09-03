import argparse
import pathlib
import shutil
import textwrap

from eval.schema import append_jsonl, read_jsonl

CANDIDATES = pathlib.Path("eval/candidates.jsonl")
GOLDEN = pathlib.Path("eval/golden.jsonl")
REJECTED = pathlib.Path("eval/rejected.jsonl")


def wrap(text: str, indent: str = "  ") -> str:
    width = min(shutil.get_terminal_size((100, 24)).columns - len(indent), 100)
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def show(item: dict, n: int, total: int, kept: int, target: int):
    print("\n" + "=" * 78)
    print(f"[{n}/{total}]  kept {kept}/{target}  ·  {item['arxiv_id']}  chunk {item['chunk_id']}")
    print("=" * 78)
    print("\nQ:")
    print(wrap(item["question"]))
    print("\nA:")
    print(wrap(item["answer"]))
    print("\nSOURCE:")
    print(wrap(item["source"]))


def edit(item: dict) -> dict:
    q = input("\n  new question (blank keeps): ").strip()
    a = input("  new answer   (blank keeps): ").strip()
    if q:
        item["question"] = q
    if a:
        item["answer"] = a
    item["edited"] = bool(q or a)
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--candidates", type=pathlib.Path, default=CANDIDATES)
    ap.add_argument("--golden", type=pathlib.Path, default=GOLDEN)
    ap.add_argument("--rejected", type=pathlib.Path, default=REJECTED)
    args = ap.parse_args()

    items = read_jsonl(args.candidates)
    if not items:
        raise SystemExit(f"no candidates at {args.candidates} — run ./init.sh gen first")

    decided = {r["cid"] for r in read_jsonl(args.golden)} | {
        r["cid"] for r in read_jsonl(args.rejected)
    }
    kept = len(read_jsonl(args.golden))
    todo = [i for i in items if i["cid"] not in decided]

    print(f"{len(items)} candidates, {len(decided)} already decided, {kept} kept")
    print("keys: [k]eep  [e]dit then keep  [d]rop  [s]kip  [q]uit — decisions save as you go")

    for n, item in enumerate(todo, 1):
        if kept >= args.target:
            print(f"\nreached {args.target}. stopping.")
            break

        show(item, n, len(todo), kept, args.target)
        while True:
            choice = input("\n  [k/e/d/s/q] > ").strip().lower()
            if choice in {"k", "e", "d", "s", "q"}:
                break

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "d":
            append_jsonl(args.rejected, item)
            continue
        if choice == "e":
            item = edit(item)
        append_jsonl(args.golden, item)
        kept += 1

    print(f"\n{kept} kept in {args.golden}, {len(read_jsonl(args.rejected))} rejected")
    if kept < args.target:
        print(f"{args.target - kept} to go — rerun ./init.sh review to pick up where you left off")


if __name__ == "__main__":
    main()
