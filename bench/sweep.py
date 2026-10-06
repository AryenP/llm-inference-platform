import argparse
import json
import pathlib
import subprocess
import time
import uuid
from datetime import UTC, datetime

import httpx

from app.settings import settings
from bench.record import append
from bench.stats import cost_per_1k

RESULT_DIR = pathlib.Path("/tmp/bench")


def wait_for(url: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/models", timeout=3).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return False


def start_server(model: str, port: int, gpu_util: float):
    # The sweep owns the server so it can guarantee prefix caching is off. A warm
    # cache across repeated runs inflates throughput, and that is the single
    # easiest way to publish a number that cannot be reproduced.
    proc = subprocess.Popen(
        [
            "vllm", "serve", model,
            "--port", str(port),
            "--max-model-len", "8192",
            "--gpu-memory-utilization", str(gpu_util),
            "--no-enable-prefix-caching",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for(f"http://localhost:{port}/v1"):
        proc.terminate()
        raise SystemExit(f"vllm did not come up for {model}")
    return proc


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(5)


def run_bench(model: str, port: int, input_len: int, output_len: int, n: int,
              warmups: int, rate, max_concurrency=None, label="") -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{label}-{uuid.uuid4().hex[:6]}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--base-url", f"http://localhost:{port}",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(n),
        "--num-warmups", str(warmups),
        "--ignore-eos",
        "--seed", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95",
        "--save-result",
        "--result-dir", str(RESULT_DIR),
        "--result-filename", name,
        "--request-rate", str(rate),
    ]
    if max_concurrency:
        cmd += ["--max-concurrency", str(max_concurrency)]
    subprocess.run(cmd, check=True)
    return json.loads((RESULT_DIR / name).read_text())


def pick(raw: dict, *names, default=None):
    for n in names:
        if raw.get(n) is not None:
            return raw[n]
    return default


def to_row(raw: dict, *, model: str, quantization: str, kernel: str, vllm_version: str,
           sampler: str, hardware: str, input_len: int, rate, warmups: int,
           hourly: float) -> dict:
    throughput = float(pick(raw, "request_throughput", default=0.0))
    return {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": hardware,
        "model": model,
        "quantization": quantization,
        "quant_kernel": kernel,
        "vllm_version": vllm_version,
        "sampler": sampler,
        "input_len_tokens": input_len,
        "request_rate": float(rate) if rate != "inf" else -1.0,
        "n_requests": int(pick(raw, "completed", "num_prompts", default=0)),
        "n_warmup_discarded": warmups,
        "prefix_cache": "disabled",
        "ttft_ms": {
            "p50": float(pick(raw, "p50_ttft_ms", "median_ttft_ms", default=0.0)),
            "p95": float(pick(raw, "p95_ttft_ms", default=0.0)),
        },
        "itl_ms": {
            "p50": float(pick(raw, "p50_itl_ms", "median_itl_ms", default=0.0)),
            "p95": float(pick(raw, "p95_itl_ms", default=0.0)),
        },
        "throughput_rps": throughput,
        "cost_per_1k_usd": round(cost_per_1k(hourly, throughput), 6) if throughput > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results.json"))
    ap.add_argument("--hardware", default="RunPod L40S 48GB")
    ap.add_argument("--hourly", type=float, default=1.09)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--n", type=int, default=120, help="prompts per run")
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--rates", type=float, nargs="+", default=[4.0])
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vllm_version = subprocess.run(
        ["vllm", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip().splitlines()[-1]

    targets = args.models or [
        (settings.model, "bf16", "none"),
        (settings.model_awq, "awq", "awq_marlin"),
    ]
    if args.models:
        targets = [(m, "unknown", "unknown") for m in args.models]

    for model, quant, kernel in targets:
        if not model:
            continue
        if args.dry_run:
            print(f"would sweep {model} ({quant})")
            continue

        proc = start_server(model, args.port, args.gpu_util)
        try:
            # latency: low arrival rate, so ttft reflects compute not queueing
            for input_len in args.input_lens:
                for rate in args.rates:
                    raw = run_bench(model, args.port, input_len, 128, args.n,
                                    args.warmups, rate, label=f"lat-{quant}-{input_len}")
                    append(to_row(raw, model=model, quantization=quant, kernel=kernel,
                                  vllm_version=vllm_version, sampler="flashinfer",
                                  hardware=args.hardware, input_len=input_len, rate=rate,
                                  warmups=args.warmups, hourly=args.hourly), args.out)
                    print(f"  recorded latency {quant} in={input_len} rate={rate}", flush=True)

            # throughput: a separate experiment, saturated rather than paced
            for conc in args.concurrencies:
                raw = run_bench(model, args.port, 1024, 128, args.n, args.warmups,
                                "inf", max_concurrency=conc, label=f"tput-{quant}-{conc}")
                append(to_row(raw, model=model, quantization=quant, kernel=kernel,
                              vllm_version=vllm_version, sampler="flashinfer",
                              hardware=args.hardware, input_len=1024, rate="inf",
                              warmups=args.warmups, hourly=args.hourly), args.out)
                print(f"  recorded throughput {quant} concurrency={conc}", flush=True)
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
