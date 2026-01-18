# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Base class and utilities for special forms.
"""

import sys

from ..parser import SheafVector


def _warn_parens_in_binding(context_name, expr):
    """Emit a warning if parentheses () are used instead of brackets [] in binding context."""
    # SheafVector is correct, SheafList with _bracket_type="(" is wrong
    if isinstance(expr, SheafVector):
        return  # Correct syntax
    if hasattr(expr, "_bracket_type") and expr._bracket_type == "(":
        line_info = (
            f" (line {expr.line})" if hasattr(expr, "line") and expr.line else ""
        )
        msg = (
            f"Syntax warning{line_info}: Use [] instead of () for {context_name}. "
            f"Example: (defn foo [x y] ...) or (let [a 1 b 2] ...)"
        )
        print(msg, file=sys.stderr)


class SpecialForm:
    """Base class for special forms."""

    def __init__(self, name):
        self.name = name

    def compile(self, compiler, args, local_vars):
        """
        Compile this special form.

        Args:
            compiler: The Sheaf compiler instance
            args: Arguments to the special form (everything after the operator)
            local_vars: Current local variable scope

        Returns:
            The compiled result
        """
        raise NotImplementedError(f"Special form '{self.name}' not implemented")
