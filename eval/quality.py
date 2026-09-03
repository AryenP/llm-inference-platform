import re

MIN_WORDS = 8
MAX_WORDS = 45
LEAK_RUN = 5


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def title_leak(question: str, title: str, run: int = LEAK_RUN) -> bool:
    # a question that quotes the title back is answerable by string match, which
    # would make retrieval look better than it is
    q, t = _words(question), _words(title)
    if len(t) < run:
        return False
    grams = {tuple(t[i : i + run]) for i in range(len(t) - run + 1)}
    return any(tuple(q[i : i + run]) in grams for i in range(len(q) - run + 1))


def bad_length(question: str) -> bool:
    return not (MIN_WORDS <= len(_words(question)) <= MAX_WORDS)


def near_duplicate(question: str, seen: set[frozenset], threshold: float = 0.8) -> bool:
    bag = frozenset(_words(question))
    if not bag:
        return True
    for other in seen:
        overlap = len(bag & other) / max(len(bag | other), 1)
        if overlap >= threshold:
            return True
    return False


def reasons(question: str, answer: str, title: str, seen: set[frozenset]) -> list[str]:
    out = []
    if bad_length(question):
        out.append(f"length {len(_words(question))} outside {MIN_WORDS}-{MAX_WORDS}")
    if title_leak(question, title):
        out.append(f"repeats {LEAK_RUN}+ words of the title")
    if near_duplicate(question, seen):
        out.append("near-duplicate of an earlier question")
    if len(_words(answer)) < 4:
        out.append("answer too short to verify")
    return out
