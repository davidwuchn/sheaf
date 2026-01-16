# Copyright (c) 2026 Damien Boureille
# Licensed under the MIT License.


class Macro:
    # Represents a compiled macro

    def __init__(self, name, params, body_template, expander_func):
        self.name = name
        self.params = params
        self.body_template = body_template
        self.expander_func = expander_func

    def expand(self, args):
        # Expand the macro with the given arguments
        return self.expander_func(args)


class MacroEngine:
    """
    Macro expansion engine.
    Handles macro definition (defmacro) and expansion.
    """

    def __init__(self, compiler=None):
        self.macros = {}  # name -> Macro
        self.compiler = compiler  # Reference to compiler for eval-at-compile-time
        self._expansion_depth = 0
        self._max_expansion_depth = 100  # Prevent infinite recursion

    def defmacro(self, name, params, body_template):
        """
        Define a new macro.

        Args:
            name: Macro name (string)
            params: List of parameter names
            body_template: Template S-expression with parameter substitutions

        Example:
            (defmacro when [cond & body]
              (list 'if cond (cons 'do body) nil))
        """

        # Create an expander function
        def expander(args):
            # Bind macro arguments to parameters
            bindings = self._bind_params(params, args)
            # Substitute parameters in body template
            return self._substitute(body_template, bindings)

        macro = Macro(name, params, body_template, expander)
        self.macros[name] = macro

    def defmacro_native(self, name, expander_func):
        """
        Define a native macro with a Python function.

        Args:
            name: Macro name
            expander_func: Python function that takes args and returns expanded S-expr

        This is useful for implementing complex macros in Python.
        """
        macro = Macro(name, [], None, expander_func)
        self.macros[name] = macro

    def expand(self, exp, recursive=True):
        """
        Expand macros in an S-expression.

        Args:
            exp: S-expression to expand
            recursive: If True, recursively expand nested macros

        Returns:
            Expanded S-expression
        """
        if self._expansion_depth > self._max_expansion_depth:
            raise RecursionError(
                f"Macro expansion depth exceeded {self._max_expansion_depth}. "
                f"Possible infinite macro recursion."
            )

        # Not a list? Nothing to expand
        if not isinstance(exp, list) or len(exp) == 0:
            return exp

        op = exp[0]

        # Check if it's a macro call
        if isinstance(op, str) and op in self.macros:
            self._expansion_depth += 1
            try:
                # Expand the macro
                expanded = self.macros[op].expand(exp[1:])
                # Recursively expand the result if requested
                if recursive:
                    expanded = self.expand(expanded, recursive=True)
                return expanded
            finally:
                self._expansion_depth -= 1

        # Not a macro? Recursively expand elements
        if recursive:
            result = []
            for x in exp:
                expanded = self.expand(x, recursive=True)
                result.append(expanded)

            # Preserve SheafVector structure for proper compilation
            # This ensures vectors remain evaluable expressions, not function calls
            if hasattr(exp, "_is_vector"):
                from .parser import SheafVector

                vector_result = SheafVector()
                vector_result.extend(result)
                return vector_result

            return result

        return exp

    def _bind_params(self, params, args):
        """
        Bind macro arguments to parameters.

        Supports:
        - Simple params: [a b c]
        - Rest params: [a & rest]
        """
        bindings = {}

        # Check for rest parameter (&)
        if "&" in params:
            rest_idx = params.index("&")
            if rest_idx + 1 >= len(params):
                raise ValueError("& must be followed by a parameter name")

            # Bind positional params
            for i in range(rest_idx):
                if i < len(args):
                    bindings[params[i]] = args[i]
                else:
                    bindings[params[i]] = None

            # Bind rest params
            rest_name = params[rest_idx + 1]
            bindings[rest_name] = args[rest_idx:]
        else:
            # Simple positional binding
            for i, param in enumerate(params):
                if i < len(args):
                    bindings[param] = args[i]
                else:
                    bindings[param] = None

        return bindings

    def _substitute(self, template, bindings):
        """
        Substitute parameter bindings in a template S-expression.

        Supports quasiquote syntax:
        - `(backtick): quasiquote - create template
        - ~ (tilde): unquote - evaluate expression
        - ~@ (tilde-at): unquote-splicing - evaluate and splice list

        Example:
            template: `(if ~cond (do ~@body) nil)
            bindings: {'cond': ['>', 'x', 0], 'body': [['print', '"yes"'], ['return', 'true']]}
            result: ['if', ['>', 'x', 0], ['do', ['print', '"yes"'], ['return', 'true']], 'nil']
        """
        # Check if this is a quasiquoted expression
        if isinstance(template, list) and len(template) > 0:
            if template[0] == "quasiquote":
                # Process quasiquote
                return self._expand_quasiquote(template[1], bindings, depth=0)

        # Old-style simple substitution (for backward compatibility)
        if isinstance(template, str):
            # It's a symbol, check if it's a parameter
            if template in bindings:
                return bindings[template]
            return template

        if isinstance(template, list):
            # Check if this is a SheafVector (vector literal)
            if hasattr(template, "_is_vector"):
                # Preserve vector structure but substitute elements
                # Create a new SheafVector to maintain vector semantics
                from .parser import SheafVector

                result = SheafVector()
                for item in template:
                    substituted = self._substitute(item, bindings)
                    result.append(substituted)
                return result

            # Regular list: recursively substitute in list
            result = []
            for item in template:
                substituted = self._substitute(item, bindings)
                # Handle splicing for rest parameters
                if isinstance(substituted, list) and self._is_splice_marker(
                    item, bindings
                ):
                    result.extend(substituted)
                else:
                    result.append(substituted)
            return result

        # Other types (int, float...) pass through
        return template

    def _expand_quasiquote(self, template, bindings, depth):
        # Handle unquote: ~expr
        if isinstance(template, list) and len(template) > 0:
            if template[0] == "unquote":
                if depth == 0:
                    # Evaluate the unquoted expression
                    unquoted_expr = template[1]

                    # Substitute variables in the expression
                    substituted = self._substitute(unquoted_expr, bindings)

                    # If it's a function call (list), evaluate at compile-time
                    if isinstance(substituted, list) and len(substituted) > 0:
                        if self.compiler is not None:
                            return self.eval_at_compile_time(substituted, bindings)
                        return substituted

                    # If it's a symbol or literal, return it AS-IS (as data, not a variable)
                    # Careful: symbols are data in macros, not variables to resolve
                    return substituted
                else:
                    # Nested quasiquote: just decrease depth
                    return [
                        "unquote",
                        self._expand_quasiquote(template[1], bindings, depth - 1),
                    ]

            # Handle unquote-splicing: ~@expr
            if template[0] == "unquote-splicing":
                if depth == 0:
                    # Evaluate and mark for splicing
                    unquoted_expr = template[1]

                    # If it's a function call, evaluate at compile-time
                    if isinstance(unquoted_expr, list) and len(unquoted_expr) > 0:
                        substituted = self._substitute(unquoted_expr, bindings)
                        if self.compiler is not None:
                            val = self.eval_at_compile_time(substituted, bindings)
                        else:
                            val = substituted
                    else:
                        val = self._substitute(unquoted_expr, bindings)

                    return ("__splice__", val)
                else:
                    return [
                        "unquote-splicing",
                        self._expand_quasiquote(template[1], bindings, depth - 1),
                    ]

            # Handle nested quasiquote
            if template[0] == "quasiquote":
                return [
                    "quasiquote",
                    self._expand_quasiquote(template[1], bindings, depth + 1),
                ]

            # Regular list: process elements
            result = []
            for item in template:
                expanded = self._expand_quasiquote(item, bindings, depth)

                # Handle splicing
                if (
                    isinstance(expanded, tuple)
                    and len(expanded) == 2
                    and expanded[0] == "__splice__"
                ):
                    # Splice the list
                    splice_val = expanded[1]
                    if isinstance(splice_val, list):
                        result.extend(splice_val)
                    else:
                        raise ValueError(f"Cannot splice non-list value: {splice_val}")
                else:
                    result.append(expanded)

            # Check if original template was a SheafVector
            if hasattr(template, "_is_vector"):
                # Preserve as SheafVector to maintain vector semantics
                from .parser import SheafVector

                vector_result = SheafVector()
                vector_result.extend(result)
                return vector_result

            return result

        # Simple symbol or literal
        if isinstance(template, str):
            # Check if it's a parameter reference
            if template in bindings:
                return bindings[template]
            return template

        # Literal value (int, float, etc.)
        return template

    def _is_splice_marker(self, item, bindings):
        # Check if an item should be spliced
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__splice__":
            return True
        return False

    def eval_at_compile_time(self, expr, bindings):
        """
        Evaluate an expression at macro expansion time.

        This is a "shallow" evaluation that executes Lisp functions (map, first, etc.)
        but treats all results as AST data, not compiled JAX code.

        The key difference from compile():
        - compile() resolves symbols as JAX variables and produces executable code
        - eval_at_compile_time() treats everything as data/AST manipulation

        Example:
            `~(first params)` where params=['x']
            - Returns the symbol 'x' as data
            - Does NOT try to resolve x as a JAX variable

        Args:
            expr: Expression to evaluate (e.g., (first params), (map transform-layer layers))
            bindings: Macro parameter bindings (already-resolved S-expressions)

        Returns:
            Result as raw data (symbols, lists, etc.) not compiled code
        """
        if self.compiler is None:
            raise RuntimeError(
                "Cannot eval-at-compile-time: MacroEngine has no compiler reference"
            )

        # Simple literal values
        if not isinstance(expr, list):
            # Could be a symbol that's in bindings
            if isinstance(expr, str) and expr in bindings:
                return bindings[expr]
            # Check if it's a function in the environment (for use as first-class value)
            if isinstance(expr, str) and expr in self.compiler.env:
                return self.compiler.env[expr]
            return expr

        if len(expr) == 0:
            return expr

        # Function call: (func arg1 arg2 ...)
        func_name = expr[0]
        args = expr[1:]

        # If func_name is not a string/symbol, this is data (e.g., nested list)
        # Return the list as-is with recursively evaluated elements
        if not isinstance(func_name, str):
            return [self.eval_at_compile_time(item, bindings) for item in expr]

        # Handle special forms
        if func_name == "quote":
            # Quote returns its argument unevaluated
            return args[0] if args else None

        # Get the function from environment or bindings
        if func_name in bindings:
            func = bindings[func_name]
        elif func_name in self.compiler.env:
            func = self.compiler.env[func_name]
        else:
            # Function not found: this is *data*, not code to execute
            # Examples: ['x'] (a vector), ['layer', ':l1', ...] (AST node)
            # Return the list as-is with recursively evaluated elements
            return [self.eval_at_compile_time(item, bindings) for item in expr]

        # Evaluate arguments recursively (but still as AST data)
        evaluated_args = [self.eval_at_compile_time(arg, bindings) for arg in args]

        # Call the function with evaluated arguments
        # This allows map, first, rest, etc. to work on AST data
        return func(*evaluated_args)


# Macro engine factory
def create_macro_engine():
    """
    Create a new MacroEngine instance.

    Standard macros are defined in lib/macros.shf.
    Users can load them with: (use macros)
    """
    return MacroEngine()
