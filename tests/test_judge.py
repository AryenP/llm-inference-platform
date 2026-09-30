from eval.judge import summarise


def test_mean_over_the_scored_subset():
    assert summarise([1.0, 0.5, 0.0]) == {"n_scored": 3, "n_failed": 0, "mean": 0.5}


def test_failures_are_counted_not_averaged_in_as_zero():
    # counting a failure as 0.0 would understate faithfulness and hide the failure
    got = summarise([1.0, RuntimeError("judge timed out"), 1.0])

    assert got == {"n_scored": 2, "n_failed": 1, "mean": 1.0}


def test_all_failed_reports_no_mean_rather_than_zero():
    got = summarise([RuntimeError("x"), RuntimeError("y")])

    assert got["mean"] is None
    assert got["n_failed"] == 2


def test_nothing_scored_is_not_a_perfect_score():
    assert summarise([])["mean"] is None
