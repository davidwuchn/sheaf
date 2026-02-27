#!/usr/bin/env bash
# Run all examples and report pass/fail.
# Usage: sheaf/tests/run-examples.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHEAF_DIR="$SCRIPT_DIR/.."
REPO_ROOT="$SHEAF_DIR/.."
EXAMPLES_DIR="$REPO_ROOT/examples"

# Find sheaf binary
SHEAF=""
for candidate in "$SHEAF_DIR/target/release/sheaf" "$SHEAF_DIR/target/debug/sheaf"; do
    if [[ -x "$candidate" ]]; then
        SHEAF="$candidate"
        break
    fi
done
if [[ -z "$SHEAF" ]]; then
    SHEAF="$(command -v sheaf 2>/dev/null || true)"
fi
if [[ -z "$SHEAF" ]]; then
    echo "ERROR: sheaf binary not found. Run: cd sheaf && cargo build"
    exit 1
fi

echo "Run examples"
echo "  binary: $SHEAF"
echo "  examples: $EXAMPLES_DIR"
echo

passed=0
failed=0
errors=""

for dir in "$EXAMPLES_DIR"/*/; do
    name="$(basename "$dir")"
    run_file="$dir/run.shf"
    if [[ ! -f "$run_file" ]]; then
        continue
    fi
    if "$SHEAF" "$run_file" > /dev/null 2>&1; then
        echo "  $name OK"
        ((passed++))
    else
        echo "  $name FAIL"
        ((failed++))
        errors="$errors $name"
    fi
done

echo
echo "Results: $passed passed, $failed failed"
if [[ $failed -gt 0 ]]; then
    echo "Failed:$errors"
    exit 1
fi
