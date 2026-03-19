"""
Run benchmarks on remote A10G instance (CUDA).
Run with: modal run modal_runner.py
"""

import json
import modal
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

app = modal.App("sheaf-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "jax[cuda12]")
    .add_local_file(
        str(REPO / "sheaf/target/release/sheaf"),
        remote_path="/root/sheaf",
    )
    .add_local_dir(
        str(REPO / "benchmarks"),
        remote_path="/root/benchmarks",
    )
)


def parse_bench_output(output: str) -> dict:
    results = {}
    for line in output.splitlines():
        m = re.match(r"\s{2}(\S.+?)\s+([\d.]+)\s+ms/iter", line)
        if m:
            results[m.group(1).strip()] = float(m.group(2))
    return results


@app.function(image=image, gpu="A10G", timeout=600)
def bench_cuda():
    import subprocess

    subprocess.run(["chmod", "+x", "/root/sheaf"], check=True)

    def run(label, cmd):
        print(f"{label}...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode}):\n{r.stderr}")
        results = parse_bench_output(r.stdout + r.stderr)
        print(f"  {len(results)} benchmarks", flush=True)
        return results

    sheaf_micro = run("Sheaf micro", ["/root/sheaf", "/root/benchmarks/bench.shf", "--device", "cuda"])
    sheaf_transformer = run("Sheaf transformer", ["/root/sheaf", "/root/benchmarks/bench_transformer.shf", "--device", "cuda"])
    sheaf_all = {**sheaf_micro, **sheaf_transformer}

    pytorch_all = run("PyTorch", ["python3", "/root/benchmarks/baseline_pytorch.py", "--device", "cuda"])
    jax_all = run("JAX", ["python3", "/root/benchmarks/baseline_jax.py"])

    return sheaf_all, pytorch_all, jax_all


@app.local_entrypoint()
def main():
    sheaf_all, pytorch_all, jax_all = bench_cuda.remote()
    # JSON on last line for run_all.py to parse
    print("RESULTS:" + json.dumps({"sheaf": sheaf_all, "pytorch": pytorch_all, "jax": jax_all}))
