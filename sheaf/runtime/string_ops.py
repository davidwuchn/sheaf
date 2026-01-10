# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
String operations for Sheaf.

Provides a generic dispatch mechanism for string methods.
Only imported when needed to avoid runtime bloat.
"""


def str_call(method_name, s, *args):
    """
    We provide only one generic operation instead of reimplementing Python in Sheaf...

    Examples:
        (str-call "upper" "hello")           -> "HELLO"
        (str-call "startswith" "hello" "he") -> True
        (str-call "replace" "foo" "o" "a")   -> "faa"
        (str-call "split" "a,b,c" ",")       -> ["a", "b", "c"]

    Args:
        method_name: name of the string method to call
        s: the string to operate on
        *args: additional arguments to pass to the method

    Returns:
        result of calling the method

    Raises:
        AttributeError: if the method doesn't exist
    """
    s_str = str(s)
    method = getattr(s_str, method_name, None)

    if method is None:
        raise AttributeError(f"String has no method '{method_name}'")

    if not callable(method):
        raise AttributeError(f"String attribute '{method_name}' is not callable")

    return method(*args)


def get_string_env():
    return {
        "str-call": str_call,
    }
