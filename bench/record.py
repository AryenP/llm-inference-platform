import json
import pathlib

RESULTS = pathlib.Path("results.json")

# Every one of these has to be present on a row before it is written. The point
# of the project is that a number can be defended, and a number missing its
# hardware, its quantization or its cache state cannot be.
REQUIRED = (
    "run_id",
    "timestamp",
    "hardware",
    "model",
    "quantization",
    "quant_kernel",
    "vllm_version",
    "sampler",
    "input_len_tokens",
    "request_rate",
    "experiment",
    "n_requests",
    "n_warmup_discarded",
    "prefix_cache",
    "ttft_ms",
    "itl_ms",
    "throughput_rps",
    "cost_per_1k_usd",
)

# max_concurrency is deliberately not in REQUIRED: a paced run has none. But a
# saturation run that omits it cannot be told apart from another, so it is
# checked below against the request rate instead.

# Which experiment produced the row. These are not interchangeable and must not
# be read off the other fields: a latency run bounded to one in-flight request
# looks exactly like a saturation run at concurrency 1, so without this the two
# can only be told apart by their input length, which is a coincidence, not a
# distinction.
EXPERIMENTS = ("latency", "saturation")
PERCENTILES = ("p50", "p95")

# A sweep run against a warm server measures the cache, not the engine. There is
# no third option here on purpose: "enabled" is not a state a recorded run may be
# in, and leaving it unset is how that happens by accident.
CACHE_STATES = ("disabled", "flushed")


def problems(row: dict) -> list[str]:
    out = [f"missing {f}" for f in REQUIRED if row.get(f) is None]

    for field in ("ttft_ms", "itl_ms"):
        value = row.get(field)
        if isinstance(value, dict):
            out += [f"{field} missing {p}" for p in PERCENTILES if value.get(p) is None]
        elif value is not None:
            out.append(f"{field} must carry percentiles, not a single number")

    experiment = row.get("experiment")
    if experiment is not None and experiment not in EXPERIMENTS:
        out.append(f"experiment is {experiment!r}, expected one of {EXPERIMENTS}")

    cache = row.get("prefix_cache")
    if cache is not None and cache not in CACHE_STATES:
        out.append(f"prefix_cache is {cache!r}, expected one of {CACHE_STATES}")

    if isinstance(row.get("n_warmup_discarded"), int) and row["n_warmup_discarded"] < 0:
        out.append("n_warmup_discarded is negative")

    # a saturation run is identified by its concurrency; without it two rows at
    # different concurrencies are indistinguishable
    if row.get("request_rate") == -1 and row.get("max_concurrency") is None:
        out.append("saturation run (request_rate -1) needs max_concurrency")

    return out


def append(row: dict, path: pathlib.Path = RESULTS) -> dict:
    bad = problems(row)
    if bad:
        raise ValueError("refusing to record an underspecified run:\n  " + "\n  ".join(bad))

    doc = json.loads(path.read_text()) if path.exists() else {"runs": []}
    doc.setdefault("runs", []).append(row)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc
