#!/usr/bin/env python3
"""
Sheaf benchmark suite.
  --save       Save results as new baseline (baseline.json)
  --device     metal (default), cuda
  --runs N     Runs per benchmark (default: 7)
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASELINE_FILE = SCRIPT_DIR / "baseline.json"
SHEAF = os.environ.get("SHEAF", str(SCRIPT_DIR.parent / "sheaf/target/release/sheaf"))
DEFAULT_DEVICE = "metal"
RUNS = 7
MACRO_RUNS = 3

MICRO = [
    ("matmul [128x128]",
     "(let [a (random-normal (random-key 0) [128 128]) b (random-normal (random-key 1) [128 128])] {repeat})",
     "(@ a b)", "@", 50),
    ("matmul [512x512]",
     "(@ (random-normal (random-key 0) [512 512]) (random-normal (random-key 1) [512 512]))",
     None, "@", 1),
    ("matmul [1024x1024]",
     "(@ (random-normal (random-key 0) [1024 1024]) (random-normal (random-key 1) [1024 1024]))",
     None, "@", 1),
    ("gelu [100000]",
     "(gelu (random-normal (random-key 0) [100000]))",
     None, "gelu", 1),
    ("softmax [32x1000]",
     "(softmax (random-normal (random-key 0) [32 1000]) :axis -1)",
     None, "softmax", 1),
    ("softmax [32x50000]",
     "(softmax (random-normal (random-key 0) [32 50000]) :axis -1)",
     None, "softmax", 1),
    ("layer-norm [4x256x768]",
     "(layer-norm (random-normal (random-key 0) [4 256 768]) {:gamma (ones [768]) :beta (zeros [768])} -1)",
     None, "layer-norm", 1),
    ("sum [1000000]",
     "(let [x (random-normal (random-key 0) [1000000])] {repeat})",
     "(sum x)", "sum", 50),
    ("cross-entropy [1x256x50257]",
     "(cross-entropy-loss (random-normal (random-key 0) [1 256 50257]) (random-randint (random-key 1) [1 256] 0 50257))",
     None, "cross-entropy-loss", 1),
    ("value-and-grad [4x512]->[512x512]",
     "((value-and-grad (fn [p] (mean (gelu (+ (@ (random-normal (random-key 0) [4 512]) (get p :W)) (get p :b)))))) {:W (random-normal (random-key 1) [512 512]) :b (zeros [512])})",
     None, None, 1),
]

MACRO = [
    ("GPT-2 124M forward", "bench_forward.shf"),
    ("GPT-2 124M value-and-grad", "bench_value_and_grad.shf"),
]


def parse_duration(s: str) -> float:
    s = s.strip()
    if s.endswith("μs"):
        return float(s[:-2]) / 1000.0
    elif s.endswith("ms"):
        return float(s[:-2])
    elif s.endswith("ns"):
        return float(s[:-2]) / 1_000_000.0
    elif s.endswith("s"):
        return float(s[:-1]) * 1000.0
    raise ValueError(f"unknown duration: {s}")


def build_expr(setup: str, body: str | None, repeat: int) -> str:
    if body is None:
        return setup
    calls = " ".join([body] * repeat)
    return setup.replace("{repeat}", f"(do {calls})")


def run_blame(cmd: list[str], discard_stdout: bool = False) -> str:
    stdout = subprocess.DEVNULL if discard_stdout else subprocess.PIPE
    r = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, text=True, timeout=600)
    if r.returncode != 0:
        print(f"FAIL: {r.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return r.stderr


def parse_wall(stderr: str) -> float:
    m = re.search(r"Profiler:\s+([\d.]+)(ms|s|μs|ns)\s+wall", stderr)
    if not m:
        print(f"No profiler output:\n{stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return parse_duration(m.group(1) + m.group(2))


def parse_op_self(stderr: str, op: str) -> float:
    for line in stderr.splitlines():
        cols = line.split()
        if len(cols) >= 4 and cols[0] == op:
            return parse_duration(cols[3])
    print(f"Op '{op}' not found in --blame output:\n{stderr}", file=sys.stderr)
    sys.exit(1)


def run_micro_once(expr: str, device: str, op: str | None) -> float:
    stderr = run_blame([SHEAF, "-c", expr, "--device", device, "--blame"])
    return parse_wall(stderr) if op is None else parse_op_self(stderr, op)


def bench_micro(name, setup, body, op, repeat, device, runs) -> float:
    expr = build_expr(setup, body, repeat)
    run_micro_once(expr, device, op)  # warmup
    times = [run_micro_once(expr, device, op) for _ in range(runs)]
    med = statistics.median(times)
    if repeat > 1:
        med /= repeat
    return med


def bench_macro_one(name: str, script: str, device: str, runs: int) -> float:
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        return None
    cmd = [SHEAF, str(script_path), "--device", device, "--blame"]
    run_blame(cmd, discard_stdout=True)  # warmup
    times = [parse_wall(run_blame(cmd, discard_stdout=True)) for _ in range(runs)]
    return statistics.median(times)


def load_baseline():
    if not BASELINE_FILE.exists():
        return None, None
    with open(BASELINE_FILE) as f:
        data = json.load(f)
    return data.get("sheaf"), data.get("git")


def print_table(results: dict, baseline_sheaf: dict | None):
    has_baseline = baseline_sheaf and any(baseline_sheaf.get(n) for n in results)

    header = f"{'Benchmark':<42} {'ms':>8}"
    if has_baseline:
        header += f" {'baseline':>8} {'delta':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    fmt = lambda v: f"{v:.3f}" if v else "-"
    for name, ms in results.items():
        line = f"  {name:<40} {fmt(ms):>8}"
        if has_baseline:
            b = baseline_sheaf.get(name)
            line += f" {fmt(b):>8}"
            if ms and b:
                change = (ms - b) / b * 100
                sign = "+" if change > 0 else ""
                line += f" {sign}{change:>5.0f}%"
            else:
                line += f" {'':>7}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Sheaf benchmark suite")
    parser.add_argument("--save", action="store_true", help="Save results as new baseline")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["metal", "cuda", "cpu"])
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    iree_compile = os.environ.get("IREE_COMPILE", "default toolchain")
    print(f"Sheaf benchmarks: device={args.device}, {args.runs} runs (median)")
    print(f"  binary: {SHEAF}")
    print(f"  iree-compile: {iree_compile}\n")

    results = {}

    print("Micro:")
    for name, setup, body, op, repeat in MICRO:
        ms = bench_micro(name, setup, body, op, repeat, args.device, args.runs)
        print(f"  {name:<40} {ms:>10.3f} ms", flush=True)
        results[name] = ms

    print("\nMacro:")
    for name, script in MACRO:
        macro_runs = min(args.runs, MACRO_RUNS)
        ms = bench_macro_one(name, script, args.device, macro_runs)
        if ms is not None:
            print(f"  {name:<40} {ms:>10.3f} ms", flush=True)
            results[name] = ms

    baseline_sheaf, baseline_git = load_baseline()
    if baseline_sheaf:
        print_table(results, baseline_sheaf)
        commit_date = subprocess.run(
            ["git", "log", "-1", "--format=%ci", baseline_git],
            capture_output=True, text=True
        ).stdout.strip().split(" ")[0]
        print(f"\nBaseline: commit {baseline_git} ({commit_date})")
    else:
        print("\nNo baseline. Run with --save to record one.")

    if args.save:
        git_rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        with open(BASELINE_FILE, "w") as f:
            json.dump({"git": git_rev, "device": args.device, "sheaf": results}, f, indent=2)
        print(f"Saved to {BASELINE_FILE}")


if __name__ == "__main__":
    main()
