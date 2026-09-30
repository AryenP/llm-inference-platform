from eval.gate import regressions

BASE = {"n": 150, "recall@5": 0.80, "mrr": 0.65, "ndcg@10": 0.72}


def test_no_regression_when_everything_holds():
    assert regressions(BASE, BASE) == []


def test_improvement_is_not_a_regression():
    assert regressions({**BASE, "recall@5": 0.91}, BASE) == []


def test_a_real_drop_is_reported():
    got = regressions({**BASE, "recall@5": 0.70}, BASE)

    assert len(got) == 1
    assert got[0].startswith("recall@5")


def test_noise_inside_the_tolerance_passes():
    assert regressions({**BASE, "recall@5": 0.795}, BASE, tolerance=0.01) == []
    assert regressions({**BASE, "recall@5": 0.785}, BASE, tolerance=0.01) != []


def test_every_regression_is_listed_not_just_the_first():
    got = regressions({"recall@5": 0.1, "mrr": 0.1, "ndcg@10": 0.1}, BASE)

    assert len(got) == 3


def test_sample_size_is_not_treated_as_a_metric():
    # a smaller golden set is a change to explain, not a build failure
    assert regressions({**BASE, "n": 10}, BASE) == []


def test_a_new_metric_has_nothing_to_regress_against():
    assert regressions({**BASE, "recall@20": 0.99}, BASE) == []


def test_a_missing_metric_does_not_silently_pass_as_zero():
    # dropping a metric should not read as a catastrophic regression either
    assert regressions({"mrr": 0.65}, BASE) == []
