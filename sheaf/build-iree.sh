#!/usr/bin/env bash
#
# build-iree.sh - Build the IREE runtime static libraries that Sheaf links
# against, and drop them where build.rs expects them.
#
# Usage:
#   sheaf/build-iree.sh          # native build + install to sheaf/iree-runtime/
#
# Environment variables:
#   IREE_INSTALL_DIR   where the .a files are copied
#                      (default: <repo>/sheaf/iree-runtime/ for cargo build)
#   IREE_SRC_DIR       where to clone the IREE source (default: <repo>/sheaf/build/iree-src)
#   IREE_BUILD_DIR     CMake build tree (default: <repo>/sheaf/build/iree-build)
#   ENABLE_CUDA        set to 1 to build the CUDA HAL driver (Linux, needs CUDA toolkit)
#   ENABLE_VULKAN      set to 1 to build the Vulkan HAL driver (off by default)
#   IREE_JOBS          parallel jobs passed to ninja (default: all cores)
#   SKIP_CLONE         set to 1 to reuse an existing IREE_SRC_DIR without re-cloning
#

set -euo pipefail

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && usage

# Locate the repo root (this script lives in <repo>/sheaf/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CARGO_TOML="$REPO_ROOT/sheaf/Cargo.toml"
if [[ ! -f "$CARGO_TOML" ]]; then
  echo "error: cannot find sheaf/Cargo.toml relative to $SCRIPT_DIR" >&2
  echo "       (expected repo layout: <repo>/sheaf/build-iree.sh and <repo>/sheaf/Cargo.toml)" >&2
  exit 1
fi

# Read the pinned IREE version from Cargo.toml ([package.metadata] iree-version).
IREE_VERSION="$(grep '^iree-version' "$CARGO_TOML" | sed 's/.*"\(.*\)"/\1/')"
if [[ -z "$IREE_VERSION" ]]; then
  echo "error: could not read iree-version from $CARGO_TOML" >&2
  exit 1
fi

INSTALL_DIR="${IREE_INSTALL_DIR:-$REPO_ROOT/sheaf/iree-runtime}"
SRC_DIR="${IREE_SRC_DIR:-$REPO_ROOT/sheaf/build/iree-src}"
BUILD_DIR="${IREE_BUILD_DIR:-$REPO_ROOT/sheaf/build/iree-build}"
JOBS="${IREE_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

echo "==> Sheaf IREE runtime build"
echo "    IREE version : v$IREE_VERSION"
echo "    source       : $SRC_DIR"
echo "    build tree   : $BUILD_DIR"
echo "    install dir  : $INSTALL_DIR"
echo "    jobs         : $JOBS"
echo

# --- prerequisites -----------------------------------------------------------
missing=()
for tool in cmake ninja git g++; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done

# Check we have clang ready on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    [[ " ${missing[*]} " != *" g++ "* ]] && missing+=("g++")
  fi
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "error: missing required tools: ${missing[*]}" >&2
  echo "       install them first:" >&2
  echo "         macOS:  xcode-select --install ; brew install cmake ninja git" >&2
  echo "         Linux:  sudo apt-get install cmake ninja-build git g++" >&2
  exit 1
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- driver flags -----------------------------------------------------------
# CPU task drivers are always on. GPU drivers are opt-in except Metal on macOS
# (which only needs system frameworks).
CMAKE_FLAGS=(
  -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_TESTS=OFF
  -DIREE_BUILD_SAMPLES=OFF
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON
  -DIREE_HAL_DRIVER_VULKAN=OFF
)

case "$OS" in
  Darwin)
    CMAKE_FLAGS+=(-DIREE_HAL_DRIVER_METAL=ON)
    echo "    platform    : macOS ($ARCH) -> Metal driver enabled"
    ;;
  Linux)
    [[ "${ENABLE_CUDA:-0}" == "1" ]]   && CMAKE_FLAGS+=(-DIREE_HAL_DRIVER_CUDA=ON)
    [[ "${ENABLE_VULKAN:-0}" == "1" ]] && CMAKE_FLAGS+=(-DIREE_HAL_DRIVER_VULKAN=ON)
    echo "    platform    : Linux ($ARCH); CUDA=${ENABLE_CUDA:-0} Vulkan=${ENABLE_VULKAN:-0}"
    ;;
  *)
    echo "error: unsupported OS '$OS' (expected Darwin or Linux)" >&2
    exit 1
    ;;
esac
echo

# --- clone IREE at the pinned tag -------------------------------------------
if [[ "${SKIP_CLONE:-0}" != "1" ]]; then
  if [[ -d "$SRC_DIR/.git" ]]; then
    echo "==> Reusing existing clone at $SRC_DIR (set SKIP_CLONE=0 and rm -rf it to refetch)"
  else
    echo "==> Cloning IREE v$IREE_VERSION (shallow) into $SRC_DIR"
    git clone --depth 1 --branch "v$IREE_VERSION" \
      https://github.com/iree-org/iree.git "$SRC_DIR"
    echo "==> Initializing submodules (shallow)"
    git -C "$SRC_DIR" submodule update --init --depth 1
  fi
else
  echo "==> SKIP_CLONE=1: assuming $SRC_DIR is already populated"
  [[ -d "$SRC_DIR" ]] || { echo "error: SKIP_CLONE=1 but $SRC_DIR does not exist" >&2; exit 1; }
fi
echo

# --- configure with CMake / Ninja -------------------------------------------
# CMake does not reliably detect host changes on its own, so discard the cache
# when running on a different OS.
PLATFORM_MARKER="$BUILD_DIR/.sheaf-iree-platform"
PLATFORM_ID="$OS-$ARCH"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cached_platform=""
  [[ -f "$PLATFORM_MARKER" ]] && cached_platform="$(<"$PLATFORM_MARKER")"
  if [[ "$cached_platform" != "$PLATFORM_ID" ]]; then
    echo "==> Removing stale CMake build tree (cached platform: ${cached_platform:-unknown}; current: $PLATFORM_ID)"
    rm -rf "$BUILD_DIR"
  fi
fi

mkdir -p "$BUILD_DIR"
echo "==> Configuring CMake"
cmake -G Ninja -B "$BUILD_DIR" -S "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  "${CMAKE_FLAGS[@]}"
printf '%s\n' "$PLATFORM_ID" > "$PLATFORM_MARKER"

# --- build the runtime static libs ------------------------------------------
echo "==> Building iree_runtime_unified + flatcc (this takes a while)"
# flatcc targets may not exist in all IREE layouts; fall back to the unified
# target only, exactly like the CI workflow.
cmake --build "$BUILD_DIR" --target iree_runtime_unified flatcc_parsing flatcc_runtime -- -j"$JOBS" \
  || cmake --build "$BUILD_DIR" --target iree_runtime_unified -- -j"$JOBS"

# --- collect the .a files ----------------------------------------------------
echo "==> Collecting static libraries into $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
found_unified=0
while IFS= read -r -d '' f; do
  cp "$f" "$INSTALL_DIR/"
  found_unified=1
  echo "    + $(basename "$f")"
done < <(find "$BUILD_DIR" -name 'libiree_runtime_unified.a' -print0)

if [[ "$found_unified" != "1" ]]; then
  echo "error: libiree_runtime_unified.a was not produced; check the CMake output above" >&2
  exit 1
fi

# flatcc libs are optional (some builds vendor them into the unified lib).
flatcc_count=0
while IFS= read -r -d '' f; do
  cp "$f" "$INSTALL_DIR/"
  echo "    + $(basename "$f")"
  flatcc_count=$((flatcc_count + 1))
done < <(find "$BUILD_DIR" -name 'libflatcc*.a' -type f -print0 2>/dev/null || true)
echo "    (flatcc libs: $flatcc_count)"
echo

echo "==> Done. Libraries installed in: $INSTALL_DIR"
echo
echo "    Required by build.rs:"
ls -1 "$INSTALL_DIR" | sed 's/^/      /'
echo
echo "    Next step:"
echo "      Install rustc and cargo"
if [[ "$INSTALL_DIR" == "$REPO_ROOT/sheaf/iree-runtime" ]]; then
  echo "      cd \"$REPO_ROOT/sheaf\" && cargo build --release"
else
  echo "      IREE_RUNTIME_LIB_DIR=\"$INSTALL_DIR\" cargo build --release  (from sheaf/)"
fi
