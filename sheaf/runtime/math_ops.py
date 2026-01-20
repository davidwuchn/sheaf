# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Implements variadic arithmetic and logical operators for Sheaf.
Translates Lisp-style functional math into vectorized JAX computations.
"""

from functools import reduce

import jax.numpy as jnp


def _sheaf_and(*args):
    """
    Lisp-style logical AND: returns last truthy value or false.

    Examples:
        (and true false) -> false
        (and true (> 2 1)) -> true
        (and 1 2 3) -> 3
    """
    if not args:
        return True

    for arg in args[:-1]:
        # Check if arg is falsy
        if arg is False or arg is None:
            return arg
        # For JAX arrays, check if all elements are truthy
        if hasattr(arg, "__iter__") and not isinstance(arg, str):
            try:
                if not jnp.all(arg):
                    return False
            except (TypeError, ValueError):
                pass

    # Return the last argument
    return args[-1]


def _sheaf_or(*args):
    """
    Lisp-style logical OR: returns first truthy value or last value.

    Examples:
        (or false true) -> true
        (or false false nil) -> nil
        (or nil 42) -> 42
    """
    if not args:
        return False

    for arg in args[:-1]:
        # Check if arg is truthy
        if arg is not False and arg is not None:
            return arg
        # For JAX arrays, check if any element is truthy
        if hasattr(arg, "__iter__") and not isinstance(arg, str):
            try:
                if jnp.any(arg):
                    return arg
            except (TypeError, ValueError):
                pass

    # Return the last argument
    return args[-1]


def get_math_env():
    return {
        # Variadic addition: (+ 1 2 3) -> 6
        "+": lambda *args: reduce(jnp.add, args) if args else 0.0,
        # Variadic multiplication: (* 2 3 4) -> 24
        "*": lambda *args: reduce(jnp.multiply, args) if args else 1.0,
        # Variadic subtraction: (- 10 2 1) -> 7. Handles unary (- 5) -> -5.
        "-": lambda *args: (-args[0] if len(args) == 1 else reduce(jnp.subtract, args)),
        # Division is usually kept binary to avoid ambiguity
        "/": lambda a, b: a / b,
        "**": jnp.power,
        "//": lambda a, b: a // b,
        "mod": lambda a, b: a % b,
        "%": lambda a, b: a % b,  # Alias for mod
        "@": jnp.matmul,
        # Boolean Logic (Using JAX-friendly reduction)
        "=": lambda a, b: jnp.array_equal(a, b),
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "abs": jnp.abs,
        "and": _sheaf_and,
        "or": _sheaf_or,
        "not": jnp.logical_not,
        "exp": jnp.exp,
        "log": jnp.log,
        "mean": jnp.mean,
        "min": jnp.min,  # JAX-compatible min
        "max": jnp.max,  # JAX-compatible max
        "sum": jnp.sum,
        "sqrt": jnp.sqrt,
    }
