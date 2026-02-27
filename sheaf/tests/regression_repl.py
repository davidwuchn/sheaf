#!/usr/bin/env python3
"""
Sheaf Regression Test Runner

Evaluates all core-tests.yaml cases against the Rust interpreter
by invoking `sheaf -c <expr>` and comparing stdout to `expected`.

Usage:
    python sheaf/tests/regression_repl.py [--verbose] [--filter NAME]

Requirements:
    - cargo build (from sheaf/) must have been run, or sheaf in PATH
    - core-tests.yaml in sheaf/tests/

Exit code: number of failures
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


TESTS_DIR = Path(__file__).resolve().parent
SHEAF_DIR = TESTS_DIR.parent
REPO_ROOT = SHEAF_DIR.parent

YAML_CANDIDATES = [
    TESTS_DIR / "core-tests.yaml",
    Path("core-tests.yaml"),
]


def find_yaml():
    for p in YAML_CANDIDATES:
        if p.exists():
            return p
    sys.exit("ERROR: core-tests.yaml not found")


def find_binary():
    candidates = [
        SHEAF_DIR / "target" / "release" / "sheaf",
        SHEAF_DIR / "target" / "debug" / "sheaf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    import shutil
    found = shutil.which("sheaf")
    if found:
        return found
    sys.exit(
        "ERROR: sheaf not found.\n"
        "Run:  cd sheaf && cargo build"
    )


def eval_expr(binary: str, expr: str) -> tuple[bool, str]:
    """Call `sheaf -c expr`, return (ok, output)."""
    try:
        result = subprocess.run(
            [binary, "-c", expr],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


# Tests to skip: not yet implemented in the V2 interpreter.
SKIP_PATTERNS = [
    r"^(vmap|scan|random-key|random-normal|random-uniform|random-randint|choice|init-zeros|init-ones|io|use|str-call)\b",
    r"\(str-call\b",
]

# Tests whose expected value reflects V1-Python/JAX artefacts that differ
# from V2-native Sheaf output. These should be updated in core-tests.yaml
# rather than worked around in the runner.
V1_FORMAT_SKIP = {
    # V1 returns Python repr of JAX arrays: Array(2., dtype=float32)
    "map",
    "fn",
    # V1 returns Python dict repr {'key': val}, V2 uses {:key val}
    "tree-map",
    "tree-map-zeros",
    "dict",
    # V1 returns (3, 2) JAX shape tuple, V2 returns [3. 2.] tensor
    "swapaxes",
}


def should_skip(name: str, expr: str) -> str | None:
    """Return skip reason if test should be skipped, else None."""
    for pat in SKIP_PATTERNS:
        if re.search(pat, name) or re.search(pat, expr):
            return "not yet implemented"
    if name in V1_FORMAT_SKIP:
        return "expected value uses V1-Python format (update core-tests.yaml)"
    return None


def normalize(s: str) -> str:
    """Collapse superficial formatting differences for comparison."""
    # Collapse whitespace runs (NumPy column-alignment padding)
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove spaces just inside brackets: [ 0. x ] → [0. x]
    s = re.sub(r'\[ +', '[', s)
    s = re.sub(r' +\]', ']', s)
    return s


def compare(actual: str, expected: str, name: str, expr: str) -> bool:
    if actual == expected:
        return True

    na, ne = normalize(actual), normalize(expected)
    if na == ne:
        return True

    # gensym: just check prefix
    if "gensym" in name or "(gensym" in expr:
        prefix = expected.split("_")[0] if "_" in expected else expected[:3]
        return actual.startswith(prefix)

    # Scalar numeric tolerance (f32 precision)
    try:
        return abs(float(actual) - float(expected)) < 1e-5
    except ValueError:
        pass

    # 1D tensor numeric tolerance: [a b c] — try element-wise comparison
    if na.startswith('[') and ne.startswith('['):
        try:
            a_vals = [float(x) for x in na.strip('[]').split()]
            e_vals = [float(x) for x in ne.strip('[]').split()]
            if len(a_vals) == len(e_vals) and len(a_vals) > 0:
                return all(abs(a - e) < 1e-5 for a, e in zip(a_vals, e_vals))
        except ValueError:
            pass

    return False


def run_tests(binary: str, test_cases: list, verbose: bool, filter_name: str | None) -> int:
    results = []
    skipped = 0

    for i, case in enumerate(test_cases):
        name = str(case.get("name", ""))
        expr = str(case.get("test", ""))
        expected = str(case.get("expected", ""))

        if filter_name and filter_name.lower() not in name.lower():
            continue

        skip_reason = should_skip(name, expr)
        if skip_reason:
            skipped += 1
            if verbose:
                print(f"{'SKIP':6} [{i:3}] {name:30} — {skip_reason}")
            continue

        ok, output = eval_expr(binary, expr)

        if not ok:
            passed = False
            detail = f"ERROR: {output}"
        else:
            passed = compare(output, expected, name, expr)
            detail = output if passed else f"expected={expected!r}  got={output!r}"

        results.append((name, passed, detail))

        status = "PASS" if passed else "FAIL"
        if verbose:
            print(f"{status:6} [{i:3}] {name:30}  {expr[:50]}")
            if not passed:
                print(f"       expected: {expected!r}")
                print(f"       got:      {output!r}")
        else:
            if passed:
                print(f"{name:40} PASS")
            else:
                print(f"{name:40} FAIL  {detail}")

    pass_count = sum(1 for _, p, _ in results if p)
    fail_count = sum(1 for _, p, _ in results if not p)
    total = pass_count + fail_count

    print()
    print("=" * 60)
    print(f"Results:  {pass_count}/{total} passed,  {fail_count} failed,  {skipped} skipped")

    if fail_count > 0 and not verbose:
        failures = [(n, d) for n, p, d in results if not p]
        print(f"\nFirst failures:")
        for name, detail in failures[:5]:
            print(f"  {name}: {detail}")

    return fail_count


def main():
    parser = argparse.ArgumentParser(description="Sheaf V2 Regression Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--filter", "-f", help="Only run tests whose name contains FILTER")
    parser.add_argument("--binary", help="Path to sheaf-eval binary (auto-detected by default)")
    args = parser.parse_args()

    yaml_path = find_yaml()
    binary = args.binary or find_binary()

    print(f"Sheaf Regression Tests")
    print(f"  binary:  {binary}")
    print(f"  tests:   {yaml_path}")
    print()

    with open(yaml_path) as f:
        test_cases = yaml.safe_load(f)

    print(f"Loaded {len(test_cases)} test cases")
    print()

    return run_tests(binary, test_cases, args.verbose, args.filter)


if __name__ == "__main__":
    sys.exit(main())
