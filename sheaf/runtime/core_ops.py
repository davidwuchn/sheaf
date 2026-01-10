# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Provides fundamental data structure manipulation primitives.
"""

from functools import reduce


def _sheaf_get_in(obj, path):
    """Navigate through nested dicts/lists using a path."""
    # If path is not a list (single level), convert it
    if not isinstance(path, (list, tuple)):
        path = [path]

    res = obj
    for key in path:
        # Auto-clean Lisp keywords: ':token' -> 'token'
        k = key[1:] if isinstance(key, str) and key.startswith(":") else key
        res = res[k]
    return res


def create_dict(*args):
    # Process arguments in pairs: key-value
    # args[i] will be the key (e.g., ":Wq"), args[i+1] the value
    d = {}
    for i in range(0, len(args), 2):
        key = args[i]
        if isinstance(key, str) and key.startswith(":"):
            key = key[1:]
        d[key] = args[i + 1]
    return d


def generic_apply(func, *args):
    """
    Apply a function to arguments.

    This enables dynamic function application, useful for:
    - Higher-order programming
    - Dynamic dispatch
    - Metaprogramming

    Examples:
        (apply + 1 2 3)        -> 6
        (apply max [1 5 3])    -> 5 (if list is unpacked)

    Args:
        func: callable function
        *args: arguments to pass to the function

    Returns:
        result of func(*args)
    """
    return func(*args)


def generic_slice(obj, start, end=None):
    """
    Generic slice operation for strings, lists, and tensors.

    Examples:
        (slice "hello" 1)     -> "ello"
        (slice "hello" 1 3)   -> "el"
        (slice [1 2 3 4] 1 3) -> [2 3]
        (slice tensor 0 10)   -> tensor[0:10]

    Args:
        obj: sequence to slice (string, list, tuple, or tensor)
        start: start index
        end: end index (optional, None means to end)

    Returns:
        sliced object
    """
    if end is None:
        return obj[start:]
    return obj[start:end]


def get_core_env():
    return {
        "apply": generic_apply,
        "dict": create_dict,
        "first": lambda x: x[0],
        # "get" is now a special form in compiler.py to avoid keyword argument issues
        # "get": lambda obj, *keys: obj[...],
        "get-in": _sheaf_get_in,
        "last": lambda x: x[-1],
        "list": lambda *args: list(args),
        "map": lambda f, lst: [f(x) for x in lst],
        "nth": lambda x, i: x[i],
        "reduce": lambda f, acc, lst: reduce(f, lst, acc),
        "slice": generic_slice,
    }
