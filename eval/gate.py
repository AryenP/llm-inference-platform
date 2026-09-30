BASELINE = "eval/baseline.json"


def regressions(current: dict, baseline: dict, tolerance: float = 0.01) -> list[str]:
    # Only metrics present in both are compared: a newly added metric has nothing
    # to regress against, and a dropped one should not fail the build silently.
    out = []
    for name, was in baseline.items():
        if name == "n" or not isinstance(was, int | float):
            continue
        now = current.get(name)
        if not isinstance(now, int | float):
            continue
        if now < was - tolerance:
            out.append(f"{name}: {now:.4f} < {was:.4f} - {tolerance}")
    return sorted(out)
