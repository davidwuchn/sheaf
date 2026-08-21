"""
Run the Sheaf CUDA benchmark suite on a remote instance.

Run with:
  modal run benchmarks/modal_runner.py

Set SHEAF to benchmark a different binary.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO / "benchmarks"


def resolve_sheaf() -> Path:
    binary = Path(
        os.environ.get("SHEAF", REPO / "bazel-bin/sheaf/sheaf")
    ).expanduser()
    if binary.is_file():
        return binary.resolve()
    raise SystemExit(
        f"Sheaf binary not found: {binary}\n"
        "Build it with:\n"
        "  bazel build --config=release //sheaf:bin\n"
        "or set SHEAF explicitly."
    )


def source_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


SHEAF = resolve_sheaf()
SOURCE_COMMIT = source_commit()
app = modal.App("sheaf-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .add_local_file(str(SHEAF), remote_path="/root/sheaf")
    .add_local_file(
        str(BENCHMARKS / "run_all.py"),
        remote_path="/root/benchmarks/run_all.py",
    )
    .add_local_file(
        str(BENCHMARKS / "bench_forward.shf"),
        remote_path="/root/benchmarks/bench_forward.shf",
    )
    .add_local_file(
        str(BENCHMARKS / "bench_vag_sheaf.shf"),
        remote_path="/root/benchmarks/bench_vag_sheaf.shf",
    )
)


@app.function(image=image, gpu="A10G", timeout=1800)
def bench_cuda():
    import subprocess

    output = "/tmp/results.json"
    env = {**os.environ, "SHEAF": "/root/sheaf"}
    if SOURCE_COMMIT:
        env["SHEAF_GIT_COMMIT"] = SOURCE_COMMIT
    result = subprocess.run(
        [
            "python3",
            "/root/benchmarks/run_all.py",
            "--device",
            "cuda",
            "--runs",
            "7",
            "--save",
            output,
        ],
        capture_output=True,
        text=True,
        timeout=1500,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Sheaf benchmark failed (exit {result.returncode}):\n{result.stderr}"
        )
    with open(output) as file:
        return json.load(file)


@app.local_entrypoint()
def main():
    results = bench_cuda.remote()
    print("RESULTS:" + json.dumps(results))
