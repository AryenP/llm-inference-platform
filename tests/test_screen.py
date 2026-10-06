from eval.screen import distinctive_terms, leaks_name, parse_verdict

ROWS = [
    {"title": "RadixMLP for Causal Transformers", "question": "x"},
    {"title": "A Survey of Transformers", "question": "x"},
    {"title": "Transformers and Attention", "question": "x"},
]


def test_distinctive_terms_exclude_common_words():
    rare = distinctive_terms(ROWS, threshold=2)

    assert "RadixMLP" in rare
    # Transformers appears in all three titles, so it carries no identity
    assert "Transformers" not in rare


def test_leaks_name_catches_a_method_the_question_names():
    rare = distinctive_terms(ROWS, threshold=2)
    row = {"title": "RadixMLP for Causal Transformers",
           "question": "What does RadixMLP deduplicate across a batch?"}

    assert leaks_name(row, rare) == {"RadixMLP"}


def test_leaks_name_allows_a_description_of_the_method():
    rare = distinctive_terms(ROWS, threshold=2)
    row = {"title": "RadixMLP for Causal Transformers",
           "question": "What is deduplicated across a batch sharing a prefix?"}

    assert leaks_name(row, rare) == set()


def test_leaks_name_ignores_a_sentence_initial_capital():
    rare = distinctive_terms(ROWS, threshold=2)
    row = {"title": "RadixMLP for Causal Transformers",
           "question": "RadixMLP aside, what is deduplicated?"}

    # the first token is skipped: sentence-initial capitals carry no signal
    assert leaks_name(row, rare) == set()


def test_parse_verdict_reads_a_complete_judgement():
    got = parse_verdict('{"answerable": true, "grounded": false, "specific": true, "why": "x"}')

    assert got["answerable"] is True
    assert got["grounded"] is False


def test_parse_verdict_strips_reasoning_and_fences():
    raw = '<think>hmm</think>\n```json\n{"answerable":true,"grounded":true,"specific":true}\n```'

    assert parse_verdict(raw)["specific"] is True


def test_parse_verdict_rejects_a_partial_judgement():
    # a missing key must not read as a pass
    assert parse_verdict('{"answerable": true}') is None
    assert parse_verdict("no json here") is None
