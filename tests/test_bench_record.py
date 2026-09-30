import json

import pytest

from bench.record import append, problems

GOOD = {
    "run_id": "r1",
    "timestamp": "2026-09-30T12:00:00Z",
    "hardware": "RunPod L40S 48GB EU-NL-1",
    "model": "Qwen/Qwen3-8B",
    "quantization": "awq",
    "quant_kernel": "awq_marlin",
    "vllm_version": "0.28.0",
    "sampler": "flashinfer",
    "input_len_tokens": 1024,
    "request_rate": 4.0,
    "n_requests": 200,
    "n_warmup_discarded": 10,
    "prefix_cache": "disabled",
    "ttft_ms": {"p50": 120.0, "p95": 180.0},
    "itl_ms": {"p50": 12.0, "p95": 19.0},
    "throughput_rps": 9.4,
    "cost_per_1k_usd": 0.0233,
}


def test_a_complete_row_has_no_problems():
    assert problems(GOOD) == []


def test_every_missing_field_is_named():
    got = problems({k: v for k, v in GOOD.items() if k not in {"hardware", "sampler"}})

    assert sorted(got) == ["missing hardware", "missing sampler"]


def test_a_single_latency_number_is_rejected():
    got = problems({**GOOD, "ttft_ms": 140.0})

    assert any("percentiles" in p for p in got)


def test_a_half_filled_percentile_block_is_rejected():
    got = problems({**GOOD, "itl_ms": {"p50": 12.0}})

    assert got == ["itl_ms missing p95"]


def test_a_warm_cache_cannot_be_recorded():
    # the trap the whole benchmark section exists to avoid
    got = problems({**GOOD, "prefix_cache": "enabled"})

    assert any("prefix_cache" in p for p in got)


def test_both_documented_cache_states_pass():
    assert problems({**GOOD, "prefix_cache": "flushed"}) == []
    assert problems({**GOOD, "prefix_cache": "disabled"}) == []


def test_zero_warmup_is_allowed_but_negative_is_not():
    assert problems({**GOOD, "n_warmup_discarded": 0}) == []
    assert problems({**GOOD, "n_warmup_discarded": -1}) != []


def test_append_writes_the_row(tmp_path):
    path = tmp_path / "results.json"

    append(GOOD, path)
    doc = json.loads(path.read_text())

    assert len(doc["runs"]) == 1
    assert doc["runs"][0]["run_id"] == "r1"


def test_append_accumulates_across_runs(tmp_path):
    path = tmp_path / "results.json"

    append(GOOD, path)
    append({**GOOD, "run_id": "r2"}, path)

    assert [r["run_id"] for r in json.loads(path.read_text())["runs"]] == ["r1", "r2"]


def test_append_refuses_an_underspecified_row_and_writes_nothing(tmp_path):
    path = tmp_path / "results.json"

    with pytest.raises(ValueError, match="underspecified"):
        append({**GOOD, "hardware": None}, path)

    assert not path.exists()
