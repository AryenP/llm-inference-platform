import pytest

from bench.stats import cost_per_1k, discard_warmup, itl_ms, percentile


def test_percentile_of_a_single_sample_is_that_sample():
    assert percentile([42.0], 95) == 42.0


def test_percentile_endpoints_are_min_and_max():
    xs = [10.0, 20.0, 30.0, 40.0]
    assert percentile(xs, 0) == 10.0
    assert percentile(xs, 100) == 40.0


def test_percentile_interpolates_between_neighbours():
    # p50 of four points sits between the 2nd and 3rd
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0


def test_percentile_ignores_input_order():
    assert percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0


def test_percentile_of_nothing_raises_rather_than_returning_zero():
    # a zero here would look like a very fast run
    with pytest.raises(ValueError):
        percentile([], 50)


def test_itl_is_the_gaps_after_the_first_token():
    # first token at 100ms is TTFT, not an inter-token latency
    assert itl_ms([100.0, 110.0, 125.0, 130.0]) == [10.0, 15.0, 5.0]


def test_itl_of_a_single_token_is_empty():
    assert itl_ms([100.0]) == []
    assert itl_ms([]) == []


def test_discard_warmup_removes_the_first_n_and_reports_the_count():
    kept, dropped = discard_warmup([1, 2, 3, 4, 5], 2)

    assert kept == [3, 4, 5]
    assert dropped == 2


def test_discard_warmup_reports_what_it_actually_dropped():
    # asking to drop more than exists must not claim it dropped them all
    kept, dropped = discard_warmup([1, 2], 5)

    assert kept == []
    assert dropped == 2


def test_discard_warmup_of_zero_keeps_everything():
    kept, dropped = discard_warmup([1, 2, 3], 0)

    assert kept == [1, 2, 3]
    assert dropped == 0


def test_cost_per_1k_from_hourly_rate_and_throughput():
    # $0.79/hr at 10 req/s → 36,000 queries an hour → about 2.2 cents per 1000
    assert cost_per_1k(0.79, 10.0) == pytest.approx(0.02194, abs=1e-5)


def test_cost_falls_as_throughput_rises():
    assert cost_per_1k(0.79, 20.0) < cost_per_1k(0.79, 10.0)


def test_cost_at_zero_throughput_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        cost_per_1k(0.79, 0.0)


def test_vllm_is_resolved_next_to_the_interpreter(tmp_path, monkeypatch):
    import sys

    from bench import sweep

    fake = tmp_path / "python"
    fake.write_text("")
    (tmp_path / "vllm").write_text("")
    monkeypatch.setattr(sys, "executable", str(fake))

    # a detached run has no useful PATH, so the venv sibling must win
    assert sweep.vllm_bin() == str(tmp_path / "vllm")


def test_vllm_resolution_falls_back_to_path(tmp_path, monkeypatch):
    import sys

    from bench import sweep

    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(sweep.shutil, "which", lambda _: "/usr/bin/vllm")

    assert sweep.vllm_bin() == "/usr/bin/vllm"
