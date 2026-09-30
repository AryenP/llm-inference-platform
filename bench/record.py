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
    "n_requests",
    "n_warmup_discarded",
    "prefix_cache",
    "ttft_ms",
    "itl_ms",
    "throughput_rps",
    "cost_per_1k_usd",
)

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

    cache = row.get("prefix_cache")
    if cache is not None and cache not in CACHE_STATES:
        out.append(f"prefix_cache is {cache!r}, expected one of {CACHE_STATES}")

    if isinstance(row.get("n_warmup_discarded"), int) and row["n_warmup_discarded"] < 0:
        out.append("n_warmup_discarded is negative")

    return out


def append(row: dict, path: pathlib.Path = RESULTS) -> dict:
    bad = problems(row)
    if bad:
        raise ValueError("refusing to record an underspecified run:\n  " + "\n  ".join(bad))

    doc = json.loads(path.read_text()) if path.exists() else {"runs": []}
    doc.setdefault("runs", []).append(row)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc
