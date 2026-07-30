"""
Run the Sheaf example test suite on a remote CUDA instance using Modal.
"""

import os
import sys
from pathlib import Path

import modal

# Paths
HERE = Path(__file__).resolve().parent           # sheaf/tests/
SHEAF_DIR = HERE.parent                          # sheaf/
REPO_ROOT = SHEAF_DIR.parent                     # repo root
EXAMPLES_DIR = REPO_ROOT / "examples"

# Sources (remote)
#   SHEAF_SOURCES    = remote | local   (default remote)
#   SHEAF_BINARY_URL = <github release url>  (default: rolling nightly)
#   SHEAF_LOCAL_BINARY = <path to a local sheaf-linux-x86_64.tar.gz>
# When set (and SHEAF_SOURCES=local), upload the local tarball to Modal for
# try-builds instead of downloading it from SHEAF_BINARY_URL.
DEFAULT_BINARY_URL = (
    "https://github.com/sheaf-lang/sheaf/releases/download/nightly/"
    "sheaf-linux-x86_64.tar.gz"
)
BINARY_URL = os.environ.get("SHEAF_BINARY_URL", DEFAULT_BINARY_URL)
LOCAL_BINARY = os.environ.get("SHEAF_LOCAL_BINARY") or ""
if LOCAL_BINARY:
    LOCAL_BINARY = str(Path(LOCAL_BINARY).resolve())
    if not Path(LOCAL_BINARY).is_file():
        raise SystemExit(
            f"examples_modal: SHEAF_LOCAL_BINARY={LOCAL_BINARY!r} not found"
        )
GIT_URL = "https://github.com/sheaf-lang/sheaf.git"

# Remote layout
REMOTE_ROOT = "/work"
REMOTE_SHEAF_DIR = f"{REMOTE_ROOT}/sheaf"
REMOTE_TARBALL = "/tmp/sheaf.tar.gz"
REMOTE_TESTS_DIR = f"{REMOTE_SHEAF_DIR}/tests"
REMOTE_BIN = f"{REMOTE_SHEAF_DIR}/target/release/sheaf"

SPARSE_PATHS = ["examples", "sheaf/tests"]

# Local env could be dirty with stale VMFB cache, random macOS noise...
# Perform some clean up first...
def _local_ignore(p) -> bool:
    s = str(p)
    if "__sheaf__" in s or ".DS_Store" in s:
        return True
    # Drop the heavy checkpoint weights (123MB + 41MB) but keep
    # out-shakespeare-char/config.json which train.shf reads.
    if s.endswith(".safetensors"):
        return True
    return False



def _download_binary_step(binary_url: str) -> str:
    return (
        f"mkdir -p {REMOTE_SHEAF_DIR}/target/release && "
        f"curl -fsSL -o {REMOTE_TARBALL} {binary_url} && "
        f"tar xzf {REMOTE_TARBALL} -C /tmp && "
        f"mv /tmp/sheaf {REMOTE_BIN} && "
        f"chmod +x {REMOTE_BIN} && "
        f"rm -f {REMOTE_TARBALL} && "
        f"test -x {REMOTE_BIN}"
    )


def _extract_local_binary_step() -> str:
    """Extract a tarball already staged in the image at REMOTE_TARBALL."""
    return (
        f"mkdir -p {REMOTE_SHEAF_DIR}/target/release && "
        f"tar xzf {REMOTE_TARBALL} -C /tmp && "
        f"mv /tmp/sheaf {REMOTE_BIN} && "
        f"chmod +x {REMOTE_BIN} && "
        f"rm -f {REMOTE_TARBALL} && "
        f"test -x {REMOTE_BIN}"
    )


def _clone_sources_step() -> str:
    return (
        f"mkdir -p {REMOTE_ROOT} && cd {REMOTE_ROOT} && "
        f"git clone --quiet --depth 1 --filter=blob:none "
        f"--no-checkout {GIT_URL} repo && "
        "cd repo && "
        "git sparse-checkout init --cone && "
        f"git sparse-checkout set {' '.join(SPARSE_PATHS)} && "
        # `--track origin/HEAD` is unreliable across git versions on
        # shallow clones; a detached HEAD checkout is robust and we
        # only need the tree, not a branch.
        "git checkout --quiet HEAD && "
        "cd .. && "
        f"mkdir -p {REMOTE_SHEAF_DIR} && "
        f"cp -R repo/examples {REMOTE_ROOT}/ && "
        f"cp -R repo/sheaf/tests {REMOTE_SHEAF_DIR}/ && "
        "rm -rf repo && "
        # Sanity: the runner refuses to operate without these.
        f"test -f {REMOTE_TESTS_DIR}/examples-run.sh && "
        f"test -f {REMOTE_TESTS_DIR}/examples-manifest.txt && "
        f"test -f {REMOTE_TESTS_DIR}/examples-checks.shf && "
        f"test -d {REMOTE_ROOT}/examples && "
        "echo 'examples_modal: clean-room sources OK'"
    )


def _base_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "ca-certificates", "unzip")
    )


def _warmup_toolchain_step() -> str:
    """Run-command that pre-downloads the IREE compiler toolchain."""
    return (
        f"{REMOTE_BIN} -c '(sum (ones [2]))' --device cpu "
        f">/tmp/warmup.out 2>/tmp/warmup.err || true"
    )


def _image_remote(binary_url: str) -> modal.Image:
    """Clean-room: clone sources + download binary + warm toolchain."""
    return (
        _base_image()
        .run_commands(_clone_sources_step())
        .run_commands(_download_binary_step(binary_url))
        .run_commands(_warmup_toolchain_step())
    )


def _image_local(binary_url: str) -> modal.Image:
    """Debug: upload local examples + harness.

    Binary origin:
      - SHEAF_LOCAL_BINARY set: upload the local tarball (try-build path).
      - otherwise: download from binary_url (release/nightly URL).

    Modal layers image instructions in order, so a locally uploaded tarball
    (add_local_file) must be staged before the run_command that extracts it.
    """
    use_local = bool(LOCAL_BINARY)
    image = (
        _base_image()
        .run_commands(
            f"mkdir -p {REMOTE_TESTS_DIR} {REMOTE_SHEAF_DIR}/target/release "
            f"{REMOTE_ROOT}/examples"
        )
    )
    if use_local:
        image = (
            image
            .add_local_file(LOCAL_BINARY, remote_path=REMOTE_TARBALL, copy=True)
            .run_commands(
                _extract_local_binary_step() + " && "
                + _warmup_toolchain_step()
            )
        )
    else:
        image = image.run_commands(
            _download_binary_step(binary_url) + " && "
            + _warmup_toolchain_step()
        )
    return (
        image
        .add_local_dir(
            str(EXAMPLES_DIR),
            remote_path=f"{REMOTE_ROOT}/examples",
            ignore=_local_ignore,
        )
        .add_local_file(str(HERE / "examples-run.sh"),
                        remote_path=f"{REMOTE_TESTS_DIR}/examples-run.sh")
        .add_local_file(str(HERE / "examples-manifest.txt"),
                        remote_path=f"{REMOTE_TESTS_DIR}/examples-manifest.txt")
        .add_local_file(str(HERE / "examples-checks.shf"),
                        remote_path=f"{REMOTE_TESTS_DIR}/examples-checks.shf")
    )


def _image(binary_url: str) -> modal.Image:
    mode = os.environ.get("SHEAF_SOURCES", "remote").lower()
    if mode == "remote":
        return _image_remote(binary_url)
    if mode == "local":
        return _image_local(binary_url)
    raise SystemExit(
        f"examples_modal: SHEAF_SOURCES={mode!r} unknown (use remote|local)"
    )


app = modal.App("sheaf-examples")


def _run(device: str, seconds: int) -> int:
    """Execute examples-run.sh on the remote and stream its output."""
    import subprocess
    env = {
        "SHEAF_DEVICE": device,
        "HOME": "/root",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    proc = subprocess.run(
        ["bash", f"{REMOTE_TESTS_DIR}/examples-run.sh"],
        cwd=REMOTE_TESTS_DIR,
        env=env,
        capture_output=False,
        timeout=seconds,
    )
    return proc.returncode


@app.function(image=_image(BINARY_URL), gpu="A10G", timeout=3600)
def run_cuda():
    """Run the examples suite on CUDA (NVIDIA A10G)."""
    return _run("cuda", seconds=3300)


@app.function(image=_image(BINARY_URL), timeout=3600)
def run_cpu():
    return _run("cpu", seconds=3300)


@app.local_entrypoint()
def main(device: str = "cuda"):
    print(
        f"examples_modal: device={device} sources="
        f"{os.environ.get('SHEAF_SOURCES','remote')} binary="
        f"{LOCAL_BINARY or BINARY_URL}",
        file=sys.stderr,
    )
    if device == "cuda":
        rc = run_cuda.remote()
    elif device == "cpu":
        rc = run_cpu.remote()
    else:
        print(
            f"examples_modal: unknown device {device!r} (use cuda|cpu)",
            file=sys.stderr,
        )
        sys.exit(2)

    if rc != 0:
        raise RuntimeError(f"examples_modal: FAILED (exit {rc})")
    print("\nexamples_modal: ALL EXAMPLES PASSED")
