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


def cons(head, tail):
    """
    Construct a new list by prepending head to tail.

    Examples:
        (cons 1 [2 3])    -> [1 2 3]
        (cons 'a [])      -> ['a]
        (cons 'x ['y 'z]) -> ['x 'y 'z]

    Args:
        head: element to prepend
        tail: list to prepend to

    Returns:
        new list with head prepended to tail
    """
    if not isinstance(tail, list):
        raise TypeError(f"cons: second argument must be a list, got {type(tail)}")
    return [head] + tail


def count(lst):
    """
    Return the number of elements in a list.

    Examples:
        (count [1 2 3])  -> 3
        (count [])       -> 0

    Returns:
        number of elements
    """
    return len(lst) if isinstance(lst, (list, tuple, str)) else 0


def empty_q(lst):
    """
    Check if a list is empty.

    Examples:
        (empty? [])     -> True
        (empty? [1 2])  -> False

    Args:
        lst: list to check

    Returns:
        True if list is empty, False otherwise
    """
    return len(lst) == 0 if isinstance(lst, (list, tuple)) else False


def rest(lst):
    """
    Return all elements of a list except the first.

    Examples:
        (rest [1 2 3])  -> [2 3]
        (rest ['a])     -> []
        (rest [])       -> []

    Returns:
        list without the first element
    """
    if not isinstance(lst, (list, tuple)):
        raise TypeError(f"rest: argument must be a list, got {type(lst)}")
    return list(lst[1:]) if len(lst) > 0 else []


def symbol_q(obj):
    """
    Check if object is a symbol.

    In Sheaf, symbols are represented as strings.

    Examples:
        (symbol? 'foo)   -> True
        (symbol? "foo")  -> True
        (symbol? 42)     -> False

    Returns:
        True if object is a symbol/string, False otherwise
    """
    return isinstance(obj, str)


def gensym(prefix="G__"):
    """
    Generate a unique symbol.

    Useful for creating unique variable names in macros.

    Examples:
        (gensym)       -> "G__1"
        (gensym "tmp") -> "tmp2"

    Args:
        prefix: prefix for the generated symbol

    Returns:
        unique symbol string
    """
    import uuid

    return f"{prefix}{uuid.uuid4().hex[:8]}"


def get_core_env():
    return {
        "apply": generic_apply,
        "cons": cons,
        "count": count,
        "dict": create_dict,
        "empty?": empty_q,
        "first": lambda x: x[0] if x else None,
        "gensym": gensym,
        # "get" is now a special form in compiler.py to avoid keyword argument issues
        # "get": lambda obj, *keys: obj[...],
        "get-in": _sheaf_get_in,
        "last": lambda x: x[-1] if x else None,
        "list": lambda *args: list(args),
        "nth": lambda x, i: x[i],
        "reduce": lambda f, acc, lst: reduce(f, lst, acc),
        "rest": rest,
        "slice": generic_slice,
        "symbol?": symbol_q,
    }
