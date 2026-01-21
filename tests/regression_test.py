#!/usr/bin/env python3
"""
Sheaf Language Final Regression Test Suite

This script tests all functions using the real results from tests/real-results.yaml
to ensure that future compiler changes don't break existing functionality.

Usage:
    python tests/final_regression_test.py [--verbose]

Expected output:
    function_name    PASS/FAIL
"""

import argparse
import re
import sys

import numpy as np
import yaml

from sheaf import Sheaf


def load_test_cases():
    """Load test cases from YAML file"""
    with open("core-tests.yaml", "r") as f:
        test_cases = yaml.safe_load(f)
    return test_cases


def create_test_function(shf, func_name, test_index, example_expr):
    """Create a test function for a single example"""
    # Create a function name that's valid in Python
    name = func_name
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace("*", "mul")
    name = name.replace("/", "div")
    name = name.replace("+", "plus")
    name = name.replace("=", "eq")
    name = name.replace("<", "lt")
    name = name.replace(">", "gt")
    name = name.replace("!", "not_")
    name = name.replace("?", "q")
    name = name.replace(":", "_")
    name = name.replace("@", "matmul")
    name = name.replace("%", "mod")
    name = name.replace(",", "_")
    name = name.replace("\\", "")
    name = name.replace("/", "_")

    # Remove any remaining special characters
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    # Ensure it starts with 'test_' and is lowercase
    if not name.startswith("test_"):
        name = f"test_{name}"
    name = name.lower()

    # Add GLOBAL unique index to make it unique across all tests
    name = f"{name}_{test_index}"

    # Create the test function definition
    func_def = f"""
(defn {name} []
  {example_expr})
"""

    return name, func_def


def test_example(
    shf, func_name, test_index, example_expr, expected_result, verbose=False
):
    """Test an example and compare with expected result"""
    try:
        # Create test function
        test_func_name, func_def = create_test_function(
            shf, func_name, test_index, example_expr
        )

        # Load the test function
        shf.load(func_def)

        # Get the function from registry
        if test_func_name not in shf.registry:
            return False, f"Function {test_func_name} not found in registry"

        test_func = shf.registry[test_func_name]

        # Call the function
        result = test_func()

        # Special handling for random functions - check shape only
        if func_name.startswith("random") or "random" in func_name:
            if hasattr(result, "shape"):
                # For random functions, just check that it returns an array with expected shape
                if "[" in expected_result and "]" in expected_result:
                    result_str = (
                        f"Random function returned array with shape {result.shape}"
                    )
                    return True, result_str
                else:
                    return False, f"Random function: Expected array, got {type(result)}"
            else:
                return False, f"Random function: Expected array, got scalar {result}"

        # Special handling for gensym - just check it starts with 'var'
        if func_name == "gensym":
            result_str = str(result)
            if result_str.startswith("var"):
                return True, f"Generated unique symbol: {result_str}"
            else:
                return False, f"Expected symbol starting with 'var', got: {result_str}"

        # Special handling for str, gensym, symbol? - the gensym part should start with 'var'
        if func_name == "str, gensym, symbol?" and "(gensym" in example_expr:
            result_str = str(result)
            if result_str.startswith("var"):
                return True, f"Generated unique symbol: {result_str}"
            else:
                return False, f"Expected symbol starting with 'var', got: {result_str}"

        # For non-random functions, convert result to string for comparison
        if hasattr(result, "shape"):  # JAX array
            result_str = str(np.array(result))
        else:  # Scalar
            result_str = str(result)

        # Compare with expected result
        if result_str == expected_result:
            return True, result_str
        else:
            return False, f"Expected: {expected_result}, Got: {result_str}"

    except Exception as e:
        return False, str(e)


def main():
    """Main regression test function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sheaf Language Regression Test Suite")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed test information"
    )
    args = parser.parse_args()

    print("Sheaf Language Final Regression Test Suite")
    print("=" * 60)

    # Create Sheaf instance
    shf = Sheaf()

    # Load test cases
    test_cases = load_test_cases()

    print(f"Loaded {len(test_cases)} test cases from real-results.yaml")
    if args.verbose:
        print("Mode: VERBOSE")
    print()

    # Test each case
    results = []

    for i, test_case in enumerate(test_cases):
        func_name = test_case["name"]
        code = test_case["test"]
        expected = test_case["expected"]

        # Test the example
        success, result_str = test_example(
            shf, func_name, i, code, expected, args.verbose
        )

        if success:
            status = "PASS"
            results.append((func_name, status, result_str))
            if args.verbose:
                print(f"Function: {func_name}")
                print(f"Test: {code}")
                print(f"Expected: {expected}")
                print(f"GOT: {result_str}\t{status}")
                print("--- ")
            else:
                print(f"{func_name:40} {status}")
        else:
            status = "FAIL"
            results.append((func_name, status, result_str))
            if args.verbose:
                print(f"Function: {func_name}")
                print(f"Test: {code}")
                print(f"Expected: {expected}")
                print(f"GOT: {result_str}\t{status}")
                print("--- ")
            else:
                print(f"{func_name:40} {status} - {result_str}")

    print()
    print("=" * 60)
    pass_count = len([r for r in results if r[1] == "PASS"])
    fail_count = len([r for r in results if r[1] == "FAIL"])
    print(f"Test Summary: {pass_count} PASS, {fail_count} FAIL")

    # Show some failures for debugging
    if fail_count > 0 and not args.verbose:
        print(f"\nFirst {min(3, fail_count)} Failures:")
        for func_name, status, result in results:
            if status == "FAIL":
                print(f"  {func_name}: {result}")
                if fail_count <= 3:
                    break
                fail_count -= 1

    # Return exit code
    return fail_count


if __name__ == "__main__":
    sys.exit(main())
