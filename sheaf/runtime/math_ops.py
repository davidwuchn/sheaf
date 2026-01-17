# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Implements variadic arithmetic and logical operators for Sheaf.
Translates Lisp-style functional math into vectorized JAX computations.
"""

from functools import reduce

import jax.numpy as jnp


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
        "@": jnp.matmul,
        # Boolean Logic (Using JAX-friendly reduction)
        "=": lambda a, b: jnp.array_equal(a, b),
        "==": lambda a, b: a == b,
        "!=": lambda a, b: jnp.logical_not(jnp.array_equal(a, b)),
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        "abs": jnp.abs,
        "and": lambda *args: reduce(jnp.logical_and, args),
        "or": lambda *args: reduce(jnp.logical_or, args),
        "not": jnp.logical_not,
        "exp": jnp.exp,
        "log": jnp.log,
        "mean": jnp.mean,
        "min": jnp.min,  # JAX-compatible min
        "max": jnp.max,  # JAX-compatible max
        "sum": jnp.sum,
        "sqrt": jnp.sqrt,
    }
