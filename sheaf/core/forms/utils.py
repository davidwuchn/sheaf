# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Utility special forms: get, dict, last, use, quote
"""

import os

from ..error_handler import set_source
from ..parser import SheafRuntimeError, SheafVector, parse_full
from .base import SpecialForm


class GetForm(SpecialForm):
    """get indexing: (get obj key1 key2 ...) or (get obj key default)"""

    def __init__(self):
        super().__init__("get")

    def compile(self, compiler, args, local_vars):
        # (get obj key1 key2 ...) or (get obj key [default])
        obj = compiler.compile(args[0], local_vars)

        # Special case: if 3 args and obj is a dict, the 3rd might be a default value
        # We need to check if key exists first before deciding
        if len(args) == 3 and isinstance(obj, dict):
            # Compile first key
            raw_key = compiler.compile(args[1], local_vars)
            clean_key = (
                raw_key[1:]
                if isinstance(raw_key, str) and raw_key.startswith(":")
                else raw_key
            )

            if clean_key in obj:
                # Key exists - check if it's a nested dict access
                val = obj[clean_key]
                if isinstance(val, dict):
                    # Might be nested access like (get {:attn {...}} :attn :Wq)
                    # Check if 3rd arg is a key-like thing (string starting with :)
                    raw_third = compiler.compile(args[2], local_vars)
                    if isinstance(raw_third, str) and raw_third.startswith(":"):
                        # It's a nested key access
                        clean_third = raw_third[1:]
                        if clean_third in val:
                            return val[clean_third]
                # Simple value found, return it
                return val
            else:
                # Key doesn't exist, treat 3rd arg as default value
                default_val = compiler.compile(args[2], local_vars)
                return default_val

        # Standard multi-key access (arrays or nested dicts)
        raw_keys = [compiler.compile(k, local_vars) for k in args[1:]]

        # Strip ':' prefix from keyword symbols
        clean_keys = [
            k[1:] if isinstance(k, str) and k.startswith(":") else k for k in raw_keys
        ]

        # Multi-dimensional access: arr[k1, k2] or dict[k1][k2]
        if len(clean_keys) > 1:
            if isinstance(obj, dict):
                # Nested dict access: iterate through keys
                res = obj
                for k in clean_keys:
                    res = res[k]
                return res
            else:
                # JAX array indexing: arr[k1, k2]
                return obj[tuple(clean_keys)]
        else:
            # Simple access: obj[key]
            return obj[clean_keys[0]]


class DictForm(SpecialForm):
    """dict literal: {:key1 val1 :key2 val2} or (dict :key1 val1 ...)"""

    def __init__(self):
        super().__init__("dict")

    def compile(self, compiler, args, local_vars):
        # (dict :key1 val1 :key2 val2 ...)
        # Args come in pairs: key, value, key, value, ...
        if len(args) % 2 != 0:
            raise SheafRuntimeError(
                "dict requires an even number of arguments (key-value pairs)", args
            )

        result = {}
        for i in range(0, len(args), 2):
            key = args[i]
            val = args[i + 1]

            # Keys are literal strings or keywords, don't compile them
            if isinstance(key, str):
                # Handle keyword keys (:key -> "key")
                if key.startswith(":"):
                    key = key[1:]
                # Handle string keys ("key" -> key)
                elif key.startswith('"') and key.endswith('"'):
                    key = key[1:-1]
                # key is now a string literal
            else:
                # Non-string keys are not supported
                raise SheafRuntimeError(
                    f"dict keys must be strings or keywords, got {type(key).__name__}: {key}",
                    args,
                )

            compiled_val = compiler.compile(val, local_vars)
            result[key] = compiled_val

        return result


class LastForm(SpecialForm):
    """last: get last element of a sequence"""

    def __init__(self):
        super().__init__("last")

    def compile(self, compiler, args, local_vars):
        return compiler.compile(args[0], local_vars)[-1]


class UseForm(SpecialForm):
    """use module import: (use module-name)"""

    def __init__(self):
        super().__init__("use")

    def compile(self, compiler, args, local_vars):
        # Clean the input name
        raw_name = str(args[0]).strip('"')

        file_path = None
        extensions = ["", ".shf"]

        # Build search roots: stdlib + cwd + current file's directory
        search_roots = []
        if not os.path.isabs(raw_name) and "/" not in raw_name:
            search_roots = list(compiler.load_path)  # stdlib + cwd
            # Add directory of current file being loaded
            if compiler.current_file and compiler.current_file != "<sheaf>":
                current_dir = os.path.dirname(os.path.abspath(compiler.current_file))
                if current_dir not in search_roots:
                    search_roots.append(current_dir)
        else:
            search_roots = [""]

        for root in search_roots:
            for ext in extensions:
                potential_path = os.path.join(root, raw_name + ext)
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    file_path = potential_path
                    break
            if file_path:
                break

        if file_path is None:
            raise SheafRuntimeError(
                f"Module '{raw_name}' not found. Searched in: {compiler.load_path}",
                args,
            )

        # Get absolute path to avoid duplicate loads
        abs_file_path = os.path.abspath(file_path)

        # Skip if already loaded
        if abs_file_path in compiler.loaded_modules:
            return None

        try:
            with open(file_path, "r") as f:
                module_code = f.read()

            # Register source code for error formatting
            set_source(module_code, file_path)

            expressions = parse_full(module_code, file_path)
            for expr in expressions:
                compiler.compile(expr, {})

            # Mark module as loaded
            compiler.loaded_modules.add(abs_file_path)

            return None
        except Exception as e:
            raise SheafRuntimeError(f"Error loading module {file_path}: {str(e)}", args)


class QuoteForm(SpecialForm):
    """quote: prevent evaluation and return data as-is"""

    def __init__(self):
        super().__init__("quote")

    def compile(self, compiler, args, local_vars):
        """
        Return the argument without evaluating it.

        Syntax: (quote expr) or 'expr

        Example:
            (quote (+ 1 2))  ; => (+ 1 2) (not evaluated)
            '(+ 1 2)         ; => (+ 1 2) (same)
            'symbol          ; => symbol (not looked up)
            '[1 2 3]         ; => (1, 2, 3) (raw tuple for shapes)
        """
        if len(args) != 1:
            raise ValueError("quote requires exactly one argument")

        expr = args[0]

        # For vectors, return as raw Python tuple (useful for shapes)
        if isinstance(expr, SheafVector):
            return self._vector_to_tuple(expr)

        # For other expressions, return as-is
        return expr

    def _vector_to_tuple(self, vec):
        """Convert a SheafVector to a raw Python tuple, recursively."""
        result = []
        for item in vec:
            if isinstance(item, SheafVector):
                result.append(self._vector_to_tuple(item))
            elif isinstance(item, str):
                # Keep symbols as strings for now
                result.append(item)
            else:
                result.append(item)
        return tuple(result)
