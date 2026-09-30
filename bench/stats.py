from collections.abc import Sequence
from itertools import pairwise


def percentile(values: Sequence[float], p: float) -> float:
    # linear interpolation between neighbours, matching numpy's default so the
    # numbers line up with anything recomputed from the raw samples later
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (p / 100)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def itl_ms(token_times_ms: Sequence[float]) -> list[float]:
    # inter-token latency is the gaps *after* the first token; the first gap is
    # TTFT and belongs to a different measurement
    return [b - a for a, b in pairwise(token_times_ms)]


def discard_warmup(samples: Sequence, n: int) -> tuple[list, int]:
    # the first requests after load include CUDA graph capture and compilation,
    # so they describe startup rather than steady state
    if n <= 0:
        return list(samples), 0
    kept = list(samples[n:])
    return kept, min(n, len(samples))


def cost_per_1k(hourly_rate: float, throughput_rps: float) -> float:
    if throughput_rps <= 0:
        raise ValueError("throughput must be positive")
    return hourly_rate * 1000 / (throughput_rps * 3600)
