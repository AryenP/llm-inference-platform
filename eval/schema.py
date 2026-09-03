import json
import pathlib
from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    cid: str
    question: str
    answer: str
    arxiv_id: str
    chunk_id: int
    source: str
    title: str
    notes: list[str] = field(default_factory=list)


def write_jsonl(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r) if isinstance(r, Candidate) else r) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: pathlib.Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
