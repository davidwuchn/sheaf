# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Transform S-expressions into executable abstract syntax trees (AST) for the Sheaf compiler.
"""

import re


class SheafRuntimeError(Exception):
    """Custom exception to carry Lisp context."""

    def __init__(self, message, expression=None):
        super().__init__(message)
        self.expression = expression
        self.is_sheaf_error = True


class SheafSyntaxError(SheafRuntimeError):
    """Syntax error in Sheaf code"""

    def __init__(self, message, line_num=None):
        super().__init__(message)
        self.line_num = line_num
        self.is_sheaf_error = True


class SheafList(list):
    def __init__(self, *args, line=None):
        super().__init__(*args)
        self.line = line

    def __format__(self, format_spec):
        if "f" in format_spec:
            raise TypeError(
                f"Attempted to format a SheafList (line {self.line}) as a number. "
                f"Check the return values in your Sheaf function. "
                f"Content snippet: {str(self)[:50]}..."
            )
        return super().__format__(format_spec)


class SheafSymbol(str):
    def __new__(cls, content, line=None):
        obj = str.__new__(cls, content)
        obj.line = line
        return obj


def tokenize(chars):
    # Remove comments: both ;; and single ; until end of line
    chars = re.sub(r";.*", "", chars)
    # Updated pattern to capture backtick (`), tilde (~), and quote (') as separate tokens
    # ~@ must be captured as a single token
    token_pattern = r'"[^"]*"|~@|[()\[\]`~\']|[^\s()\[\]`~\']+'
    lines = chars.splitlines()
    tokens_with_meta = []
    for line_num, line in enumerate(lines, 1):
        for match in re.finditer(token_pattern, line):
            token = match.group()
            tokens_with_meta.append((token, line_num))
    return tokens_with_meta


def atom(token, line_num):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return SheafSymbol(token, line=line_num)


def parse(tokens, last_func=None):
    if not tokens:
        raise SheafSyntaxError(
            "Unexpected end of file - missing closing parenthesis or bracket"
        )

    token_text, line_num = tokens.pop(0)

    # Contextual help: find the function name if we see 'defn'
    if token_text == "defn" and tokens:
        last_func = tokens[0][0]

    # Reader macros: ' ` ~ ~@
    if token_text == "'":
        # Quote: prevent evaluation
        # 'expr => (quote expr)
        return SheafList(["quote", parse(tokens, last_func)], line=line_num)

    if token_text == "`":
        # Backtick: quasiquote
        # `expr => (quasiquote expr)
        return SheafList(["quasiquote", parse(tokens, last_func)], line=line_num)

    if token_text == "~":
        # Tilde: unquote
        # ~expr => (unquote expr)
        return SheafList(["unquote", parse(tokens, last_func)], line=line_num)

    if token_text == "~@":
        # Tilde-at: unquote-splicing
        # ~@expr => (unquote-splicing expr)
        return SheafList(["unquote-splicing", parse(tokens, last_func)], line=line_num)

    if token_text in ("(", "["):
        L = SheafList(line=line_num)
        while tokens and tokens[0][0] not in (")", "]"):
            # Pass the function name down the recursion
            L.append(parse(tokens, last_func=last_func))

        if not tokens:
            ctx = f" in function `{last_func}`" if last_func else ""
            raise SheafSyntaxError(f"Unclosed parenthesis or bracket{ctx}", line_num)
        tokens.pop(0)
        return L
    elif token_text in (")", "]"):
        ctx = f" in function `{last_func}`" if last_func else ""
        raise SheafSyntaxError(
            f"Unexpected closing character '{token_text}'{ctx}", line_num
        )
    else:
        return atom(token_text, line_num)


def parse_full(code):
    """Takes raw code and returns a list of expressions (AST)."""
    tokens = tokenize(code)
    expressions = []
    while tokens:
        expressions.append(parse(tokens))
    return expressions
