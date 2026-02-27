#!/usr/bin/env bash
# Sheaf test suite — runs everything in one pass.
# Usage: sheaf/tests/run_all.sh
#
# Phase 1: cargo test (unit + integration tests)
# Phase 2: REPL regression tests (core-tests.yaml via sheaf -c)
# Phase 3: Example smoke tests (sheaf run on each example)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHEAF_DIR="$SCRIPT_DIR/.."

failed=0

echo "=== Phase 1: cargo test ==="
if (cd "$SHEAF_DIR" && cargo test 2>&1); then
    echo "Phase 1: OK"
else
    echo "Phase 1: FAIL"
    ((failed++))
fi
echo

echo "=== Phase 2: REPL regression tests ==="
if python3 "$SCRIPT_DIR/regression_repl.py"; then
    echo "Phase 2: OK"
else
    echo "Phase 2: FAIL"
    ((failed++))
fi
echo

echo "=== Phase 3: Example smoke tests ==="
if "$SCRIPT_DIR/run-examples.sh"; then
    echo "Phase 3: OK"
else
    echo "Phase 3: FAIL"
    ((failed++))
fi
echo

if [[ $failed -eq 0 ]]; then
    echo "All tests passed."
else
    echo "$failed phase(s) failed."
    exit 1
fi
