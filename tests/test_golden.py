import json

from eval.generate import parse
from eval.quality import bad_length, near_duplicate, reasons, title_leak

TITLE = "Efficient Memory Management for Large Language Model Serving with PagedAttention"


def test_parse_strips_a_reasoning_block():
    raw = '<think>the user wants json</think>\n{"question": "q", "answer": "a"}'
    assert parse(raw) == {"question": "q", "answer": "a"}


def test_parse_finds_json_among_prose():
    assert parse('Sure!\n```json\n{"question":"q","answer":"a"}\n```') == {
        "question": "q",
        "answer": "a",
    }


def test_parse_rejects_incomplete_pairs():
    assert parse('{"question": "q"}') is None
    assert parse('{"question": "", "answer": "a"}') is None
    assert parse("not json at all") is None
    assert parse(json.dumps(["q", "a"])) is None


def test_title_leak_catches_a_quoted_run():
    # answerable by string match rather than by retrieval
    assert title_leak("What is efficient memory management for large language model serving?", TITLE)


def test_title_leak_allows_a_paraphrase():
    assert not title_leak("How does the paging scheme cut wasted key-value storage?", TITLE)


def test_title_leak_ignores_short_titles():
    assert not title_leak("what does this short one do exactly here", "Attention")


def test_bad_length_bounds():
    assert bad_length("too short")
    assert not bad_length(" ".join(["word"] * 20))
    assert bad_length(" ".join(["word"] * 60))


def test_near_duplicate_matches_a_reworded_question():
    seen = {frozenset(["how", "does", "paged", "attention", "reduce", "kv", "cache", "waste"])}
    assert near_duplicate("how does paged attention reduce kv cache waste really", seen)
    assert not near_duplicate("what throughput gain does continuous batching give", seen)


def test_reasons_reports_every_problem_it_finds():
    why = reasons("short", "ok", TITLE, set())
    assert any("length" in r for r in why)
    assert any("answer too short" in r for r in why)
