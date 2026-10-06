from eval.judge import summarise


def test_mean_over_the_scored_subset():
    assert summarise([1.0, 0.5, 0.0]) == {"n_scored": 3, "n_failed": 0, "mean": 0.5}


def test_failures_are_counted_not_averaged_in_as_zero():
    # counting a failure as 0.0 would understate faithfulness and hide the failure
    got = summarise([1.0, RuntimeError("judge timed out"), 1.0])

    assert got["n_scored"] == 2
    assert got["n_failed"] == 1
    assert got["mean"] == 1.0


def test_all_failed_reports_no_mean_rather_than_zero():
    got = summarise([RuntimeError("x"), RuntimeError("y")])

    assert got["mean"] is None
    assert got["n_failed"] == 2


def test_nothing_scored_is_not_a_perfect_score():
    assert summarise([])["mean"] is None


def test_scoring_never_runs_more_than_the_limit_at_once():
    import asyncio

    from eval import judge

    peak = 0
    live = 0

    async def fake_score(metric, item, contexts, needs):
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return 1.0

    original = judge._score_one
    judge._score_one = fake_score
    try:
        items = [{"cid": str(i), "question": "q", "answer": "a"} for i in range(40)]
        out = asyncio.run(judge._score_all(None, items, {}, True, limit=4))
    finally:
        judge._score_one = original

    assert len(out) == 40
    assert peak <= 4


def test_summarise_carries_the_first_error():
    got = summarise([RuntimeError("judge timed out"), 1.0])

    assert "RuntimeError" in got["first_error"]
    assert "timed out" in got["first_error"]


def test_summarise_has_no_error_field_when_all_scored():
    assert "first_error" not in summarise([1.0, 0.5])
