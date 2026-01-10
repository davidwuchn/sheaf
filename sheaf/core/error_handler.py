# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Sheaf's error formatting logic
"""

import sys
from typing import Optional


class SheafErrorFormatter:
    def __init__(self):
        self.source_code = None
        self.source_lines = []
        self.filename = "<sheaf>"

    def set_source(self, code: str, filename: str = "<sheaf>"):
        # Store the original Sheaf source code for error context
        self.source_code = code
        self.source_lines = code.splitlines()
        self.filename = filename

    def get_code_context(self, line_num: int, context_lines: int = 2) -> str:
        # Get lines of code around the error with line numbers
        if not self.source_lines:
            return ""

        lines = []
        start = max(1, line_num - context_lines)
        end = min(len(self.source_lines), line_num + context_lines)

        for i in range(start, end + 1):
            line_text = self.source_lines[i - 1] if i <= len(self.source_lines) else ""
            prefix = "→ " if i == line_num else "  "
            lines.append(f"{prefix}{i:4d} | {line_text}")

        return "\n".join(lines)

    def format_error(
        self, error: Exception, expression=None, func_name: str = "top-level"
    ) -> str:
        # Extract line number if available
        line_num = None
        if hasattr(expression, "line"):
            line_num = expression.line

        # Build the error message
        parts = []

        # Error type and message
        error_type = type(error).__name__
        error_msg = str(error)

        # Clean up common Python error messages
        if error_type == "TypeError":
            if "positional argument" in error_msg:
                error_msg = "wrong number of arguments"
            elif "got an unexpected keyword argument" in error_msg:
                error_msg = error_msg.replace(
                    "() got an unexpected keyword argument",
                    "received unexpected parameter",
                )
        elif error_type == "KeyError":
            error_msg = f"key not found: {error_msg}"
        elif error_type == "IndexError":
            error_msg = f"index out of range: {error_msg}"

        # Header with location
        location = f"{self.filename}"
        if line_num:
            location += f":{line_num}"
        if func_name != "top-level":
            location += f" in `{func_name}`"

        parts.append(f"\nerror: {error_msg}")
        parts.append(f" --> {location}")

        # Show code context if we have line number
        if line_num and self.source_lines:
            parts.append("  |")
            # Get context lines
            context_lines = 2
            start = max(1, line_num - context_lines)
            end = min(len(self.source_lines), line_num + context_lines)

            for i in range(start, end + 1):
                line_text = (
                    self.source_lines[i - 1] if i <= len(self.source_lines) else ""
                )
                if i == line_num:
                    parts.append(f"{i:3} | {line_text}")
                    # Add caret line pointing to error
                    if expression and str(expression) != "<syntax error>":
                        # Try to find the expression in the line
                        expr_str = str(expression)
                        if expr_str in line_text:
                            col = line_text.index(expr_str)
                            parts.append(f"    | {' ' * col}{'^' * len(expr_str)}")
                        else:
                            parts.append(f"    | ^")
                    else:
                        parts.append(f"    | ^")
                else:
                    parts.append(f"{i:3} | {line_text}")

        parts.append("  |")

        # Suggestions if we have some...
        suggestion = self.get_suggestion(error, expression)
        if suggestion:
            parts.append(f"  = note: {suggestion}")

        parts.append("")

        return "\n".join(parts)

    def get_suggestion(self, error: Exception, expression) -> Optional[str]:
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # JAX TracerBoolConversionError - very common in JIT functions
        if (
            error_type == "TracerBoolConversionError"
            or "boolean conversion of traced array" in error_msg
        ):
            return (
                "Cannot use control flow (if/and/or) with traced values in JIT functions.\n"
                "  = hint: Use 'where' instead of 'if' for differentiable branching:\n"
                "         Replace: (if condition then-expr else-expr)\n"
                "         With:    (where condition then-expr else-expr)"
            )

        if error_type == "TypeError":
            if "not callable" in error_msg:
                return "Make sure you're calling a function, not a value."
            if "argument" in error_msg:
                return "Check the number of arguments you're passing to the function."

        elif error_type == "NameError" or "not found" in error_msg:
            return "Check for typos in function or variable names."

        elif error_type == "KeyError":
            return "Verify that the key exists in the dictionary."

        elif "Shapes must be" in error_msg:
            return "Tensor shape mismatch. Check your array dimensions."

        elif "broadcasting" in error_msg:
            return "Arrays have incompatible shapes for broadcasting."

        return None


# Global formatter instance
_formatter = SheafErrorFormatter()


def set_source(code: str, filename: str = "<sheaf>"):
    _formatter.set_source(code, filename)


def format_error(
    error: Exception, expression=None, func_name: str = "top-level"
) -> str:
    return _formatter.format_error(error, expression, func_name)


def install_exception_handler():
    """
    Install a custom exception handler that catches Sheaf errors
    and displays them formatted + without Python traces.
    """
    original_excepthook = sys.excepthook

    def sheaf_excepthook(exc_type, exc_value, exc_traceback):
        # Check if this is a Sheaf error
        if hasattr(exc_value, "is_sheaf_error") and exc_value.is_sheaf_error:
            # This is already a formatted Sheaf error, just print it
            print(str(exc_value), file=sys.stderr)
        elif exc_traceback and "sheaf/core/compiler.py" in str(
            exc_traceback.tb_frame.f_code.co_filename
        ):
            # This is an error that originated in Sheaf but wasn't caught
            # Format it nicely
            formatted = format_error(exc_value)
            print(formatted, file=sys.stderr)
        else:
            # Not a Sheaf error, use default handler
            original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = sheaf_excepthook
