# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Special forms registry for Sheaf compiler.

Each special form is a class that handles the compilation logic
for a specific S-expression operator (defn, let, if, etc.).
"""

from .error_handler import set_source
from .parser import SheafRuntimeError
from .tracer import shf_tracer


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


class ThreadFirstForm(SpecialForm):
    """-> threading macro: (-> x (f1) (f2 a)) becomes (f2 (f1 x) a)"""

    def __init__(self):
        super().__init__("->")

    def compile(self, compiler, args, local_vars):
        # (-> x (layer1) (layer2))
        # x becomes the first argument of the following functions
        current_expr = args[0]
        for step in args[1:]:
            if isinstance(step, list):
                # (func arg1) -> becomes (func current_expr arg1)
                func_name = step[0]
                rest_args = step[1:]
                current_expr = [func_name, current_expr] + rest_args
            else:
                # If 'func' alone (ex: relu) -> becomes (func current_expr)
                current_expr = [step, current_expr]
        # Compile final expression
        return compiler.compile(current_expr, local_vars)


class ThreadAsForm(SpecialForm):
    """as-> threading macro: (as-> init name step1 step2) binds value at each step"""

    def __init__(self):
        super().__init__("as->")

    def compile(self, compiler, args, local_vars):
        # (as-> initial_val name step1 step2 ...)
        val_exp = args[0]
        var_name = args[1]
        steps = args[2:]

        # Eval initial value
        current_val = compiler.compile(val_exp, local_vars)

        # Create a local context to avoid polluting
        context = dict(local_vars)

        for step in steps:
            context[var_name] = current_val
            current_val = compiler.compile(step, context)

        return current_val


class CaseForm(SpecialForm):
    """case pattern matching: (case target val1 result1 val2 result2 ... default)"""

    def __init__(self):
        super().__init__("case")

    def compile(self, compiler, args, local_vars):
        # (case target val1 result1 val2 result2 ... default)
        target_val = compiler.compile(args[0], local_vars)

        # Iterate through pairs
        for i in range(1, len(args) - 1, 2):
            case_val = compiler.compile(args[i], local_vars)
            if target_val == case_val:
                return compiler.compile(args[i + 1], local_vars)

        # If odd number of arguments, the last one is the default
        if len(args) % 2 == 0:
            return compiler.compile(args[-1], local_vars)
        return None


class DefnForm(SpecialForm):
    """defn function definition: (defn name [params] body)"""

    def __init__(self):
        super().__init__("defn")

    def _expr_to_source(self, expr, indent=0):
        """Convert a parsed expression back to readable source code."""
        if isinstance(expr, list):
            if not expr:
                return "[]"

            # Special handling for 'let' bindings
            if len(expr) > 0 and expr[0] == "let":
                # (let [bindings...] body...)
                lines = ["(let ("]
                bindings = expr[1]
                # Group bindings in pairs
                for i in range(0, len(bindings), 2):
                    if i + 1 < len(bindings):
                        var_name = bindings[i]
                        var_val = self._expr_to_source(bindings[i + 1], indent + 4)
                        lines.append(" " * (indent + 6) + f"{var_name} {var_val}")
                lines.append(" " * (indent + 4) + ")")
                # Body expressions
                for body_expr in expr[2:]:
                    lines.append(
                        " " * (indent + 2) + self._expr_to_source(body_expr, indent + 2)
                    )
                lines.append(" " * indent + ")")
                return "\n".join(lines)

            # Format as multi-line if contains nested lists
            has_nested = any(isinstance(e, list) for e in expr)
            if has_nested and len(expr) > 2:
                lines = ["("]
                for i, e in enumerate(expr):
                    if i == 0:
                        lines[0] += self._expr_to_source(e, indent)
                    else:
                        lines.append(
                            " " * (indent + 2) + self._expr_to_source(e, indent + 2)
                        )
                lines.append(" " * indent + ")")
                return "\n".join(lines)
            else:
                # Simple one-liner
                items = " ".join(self._expr_to_source(e, indent) for e in expr)
                return f"({items})"
        elif isinstance(expr, str):
            # Keep strings as-is
            if expr.startswith(":") or expr.startswith('"'):
                return expr
            return expr
        else:
            # Numbers, etc.
            return str(expr)

    def compile(self, compiler, args, local_vars):
        is_jit = args[0] == ":jit"
        offset = 1 if is_jit else 0

        name = args[offset]
        params = args[offset + 1]
        body = args[offset + 2 :]

        def generated_func(*input_args, **kwargs):
            # 1. Check if 'trace' was passed in this specific call
            trace_call = kwargs.pop("trace", False)
            log_call = kwargs.pop("log", "console")
            scope_call = kwargs.pop("scope", None)

            # 2. Setup tracing if needed
            original_trace_state = getattr(compiler, "trace", False)
            if trace_call:
                compiler.trace = True
                shf_tracer.enabled = True
                shf_tracer.monitoring = True
                shf_tracer.level = 0

                shf_tracer.mode = (
                    trace_call if isinstance(trace_call, str) else "normal"
                )
                shf_tracer.log_format = log_call
                shf_tracer.scope_filter = scope_call

                if scope_call:
                    print(
                        f"--- Selective Tracing: {scope_call} (Mode: {shf_tracer.mode}) ---"
                    )
                else:
                    print(
                        f"--- Tracing Sheaf Function: {name} [Mode: {shf_tracer.mode}] ---"
                    )

            try:
                # 3. Standard execution logic
                context = dict(local_vars)
                arg_bindings = dict(zip(params, input_args))
                context.update(arg_bindings)
                context["__current_func__"] = name

                res = None
                for expression in body:
                    res = compiler.compile(expression, context)

                return res

            finally:
                # 4. Clean up trace state
                if trace_call:
                    shf_tracer.enabled = False
                    compiler.trace = original_trace_state

        # Apply JAX JIT if requested
        if is_jit:
            import jax

            from .compiler import HashableDict

            static_argnums = []
            if "config" in params:
                static_argnums.append(params.index("config"))

            # Create a wrapper to ensure dictionaries are hashable for JAX
            base_func = generated_func

            def jitted_wrapper(*args, **kwargs):
                # Extract trace/log/scope kwargs BEFORE passing to jax.jit
                trace_kwarg = kwargs.pop("trace", False)
                kwargs.pop("log", None)  # Remove but ignore
                kwargs.pop("scope", None)  # Remove but ignore

                # Warn user if they try to trace a JIT function
                if trace_kwarg:
                    print(
                        f"Warning: Cannot trace JIT-compiled function '{name}'. Tracing disabled."
                    )

                new_args = list(args)
                for idx in static_argnums:
                    if isinstance(new_args[idx], dict):
                        new_args[idx] = HashableDict(new_args[idx])

                # Call JIT function WITHOUT trace kwargs
                return jax.jit(base_func, static_argnums=tuple(static_argnums))(
                    *new_args, **kwargs
                )

            generated_func = jitted_wrapper
            generated_func._sheaf_is_jit = True

        # Check for redefinition
        if name in compiler.registry or name in compiler.env:
            from .parser import SheafRuntimeError

            # Determine where it's defined
            location = "user code" if name in compiler.registry else "standard library"

            raise SheafRuntimeError(
                f"Error:\nFunction '{name}' is already defined in {location}. "
                f"Redefinition is not allowed to prevent shadowing bugs.",
                args,
            )

        # Store source code for inspection in REPL
        params_str = "[" + " ".join(str(p) for p in params) + "]"
        source_lines = [f"(defn{' :jit' if is_jit else ''} {name} {params_str}"]
        for expr in body:
            source_lines.append("  " + self._expr_to_source(expr, 2))
        source_lines.append(")")
        generated_func.__sheaf_source__ = "\n".join(source_lines)
        generated_func.__sheaf_name__ = name
        generated_func.__sheaf_params__ = params

        # Register the function
        compiler.registry[name] = generated_func
        compiler.env[name] = generated_func
        return generated_func


class GetForm(SpecialForm):
    """get indexing: (get obj key1 key2 ...)"""

    def __init__(self):
        super().__init__("get")

    def compile(self, compiler, args, local_vars):
        # (get obj key1 key2 ...)
        obj = compiler.compile(args[0], local_vars)
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
            # Simple access: obj[k1]
            return obj[clean_keys[0]]


class GuardForm(SpecialForm):
    """guard runtime assertions: (guard :no-nan x) or (guard :shape [64 256] x)"""

    def __init__(self):
        super().__init__("guard")

    def compile(self, compiler, args, local_vars):
        # Format: (guard :type x) or (guard :type expected x)
        shf_tracer.monitoring = True

        guard_type = args[0]

        if guard_type == ":no-nan":
            # (guard :no-nan x)
            val_expr = args[1]
            val = compiler.compile(val_expr, local_vars)
            return shf_tracer.trigger_guard(":no-nan", val)

        elif guard_type in (":shape", ":range"):
            # (guard :shape expected x) or (guard :range expected x)
            # expected must be a literal list, not compiled (we need Python values, not JAX tracers)
            expected_expr = args[1]
            if not isinstance(expected_expr, list):
                raise SheafRuntimeError(
                    f"guard {guard_type} expects a literal list, got {expected_expr}",
                    args,
                )
            # Convert to Python list of concrete values
            expected = [
                float(x) if isinstance(x, (int, float)) else x for x in expected_expr
            ]
            val_expr = args[2]
            val = compiler.compile(val_expr, local_vars)
            return shf_tracer.trigger_guard(guard_type, val, expected)

        raise SheafRuntimeError(f"Unknown guard type: {guard_type}", args)


class IfForm(SpecialForm):
    """if conditional: (if cond then else)"""

    def __init__(self):
        super().__init__("if")

    def compile(self, compiler, args, local_vars):
        cond = compiler.compile(args[0], local_vars)
        return (
            compiler.compile(args[1], local_vars)
            if cond
            else compiler.compile(args[2], local_vars)
        )


class LambdaForm(SpecialForm):
    """lambda anonymous function: (lambda [params] body)"""

    def __init__(self):
        super().__init__("lambda")

    def compile(self, compiler, args, local_vars):
        # Format: (lambda [params] body)
        l_params, *l_body = args

        # Capture the current local_vars at definition time
        def anonymous_func(*l_args, closure_env=dict(local_vars)):
            # Merge closure_env with current lambda arguments
            l_context = dict(closure_env)
            l_context.update(dict(zip(l_params, l_args)))

            res = None
            for expr in l_body:
                res = compiler.compile(expr, l_context)
            return res

        return anonymous_func


class LastForm(SpecialForm):
    """last: get last element of a sequence"""

    def __init__(self):
        super().__init__("last")

    def compile(self, compiler, args, local_vars):
        return compiler.compile(args[0], local_vars)[-1]


class LetForm(SpecialForm):
    """let local bindings: (let [var1 val1 var2 val2] body)"""

    def __init__(self):
        super().__init__("let")

    def compile(self, compiler, args, local_vars):
        # args is [bindings_list, body_exp1, body_exp2, ...]
        bindings, *body = args

        # Copy context to avoid polluting parent scope
        current_context = dict(local_vars)

        # Process pairs: (var1 val1 var2 val2 ...)
        for i in range(0, len(bindings), 2):
            target = bindings[i]
            # Compile the value using the context updated by previous pairs
            val = compiler.compile(bindings[i + 1], current_context)

            if isinstance(target, list):  # Support for [a b] (split key)
                for name, v in zip(target, val):
                    current_context[name] = v
            else:
                current_context[target] = val

        # Execute the body with the final context
        res = None
        for expression in body:
            res = compiler.compile(expression, current_context)
        return res


class RepeatForm(SpecialForm):
    """repeat loop: (repeat [i n] [acc init] body)"""

    def __init__(self):
        super().__init__("repeat")

    def compile(self, compiler, args, local_vars):
        # Syntax: (repeat [i 6] [acc_name init_val] body)
        binding_iter = args[0]  # [i, 6]
        idx_name, count_expr = binding_iter[0], binding_iter[1]
        count = compiler.compile(count_expr, local_vars)

        binding_acc = args[1]  # [h, x]
        acc_name, init_expr = binding_acc[0], binding_acc[1]
        current_val = compiler.compile(init_expr, local_vars)

        body = args[2]

        for i in range(int(count)):
            # Context for this iteration
            loop_ctx = dict(local_vars)
            loop_ctx[idx_name] = i
            loop_ctx[acc_name] = current_val  # Inject previous value

            # Evaluate body
            current_val = compiler.compile(body, loop_ctx)

        return current_val


class StaticForm(SpecialForm):
    """static: force static evaluation in JIT context"""

    def __init__(self):
        super().__init__("static")

    def compile(self, compiler, args, local_vars):
        val = compiler.compile(args[0], local_vars)
        try:
            if hasattr(val, "item") and callable(getattr(val, "item")):
                return val.item()
            # Fallback to int conversion for standard tracers
            if isinstance(val, (float, int)):
                return val
            return int(val)
        except Exception as e:
            current_func = local_vars.get("__current_func__", "unknown")
            msg = (
                f"The 'static' form failed in function '{current_func}'.\n"
                f"Reason: JAX Tracer detected. Inside :jit, you cannot use 'static' on "
                f"values that depend on function inputs.\n"
                f"Value: {val}\n"
                f"Sub-expression: (static {args[0]})"
            )
            raise SheafRuntimeError(msg, args) from None


class UseForm(SpecialForm):
    """use module import: (use module-name)"""

    def __init__(self):
        super().__init__("use")

    def compile(self, compiler, args, local_vars):
        import os

        from .parser import parse_full

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


class WithParamsForm(SpecialForm):
    """with-params parameter destructuring: (with-params p body)"""

    def __init__(self):
        super().__init__("with-params")

    def compile(self, compiler, args, local_vars):
        p_expr, *body = args
        p_val = compiler.compile(p_expr, local_vars)
        context = dict(local_vars)
        if isinstance(p_val, dict):
            for k, v in p_val.items():
                # Normalize the key in case it starts with ":"
                clean_k = k[1:] if isinstance(k, str) and k.startswith(":") else k
                context[clean_k] = v

        res = None
        for e in body:
            res = compiler.compile(e, context)
        return res


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
        """
        if len(args) != 1:
            raise ValueError("quote requires exactly one argument")

        # Return the expression as-is, without evaluation
        return args[0]


class DefmacroForm(SpecialForm):
    """defmacro macro definition: (defmacro name [params] body)"""

    def __init__(self):
        super().__init__("defmacro")

    def compile(self, compiler, args, local_vars):
        """
        Define a macro at compile-time.

        Syntax: (defmacro name [params] body-template)

        Example:
            (defmacro when [cond body]
              `(if ~cond ~body nil))
        """
        if len(args) < 3:
            raise ValueError("defmacro requires name, params, and body")

        name = args[0]
        params = args[1]
        body_template = args[2]  # Usually a quasiquoted expression

        # Create an expander function
        def expander(macro_args):
            # Bind macro arguments to parameters
            bindings = compiler.macro_engine._bind_params(params, macro_args)
            # Substitute in the template
            return compiler.macro_engine._substitute(body_template, bindings)

        # Register the macro in the macro engine
        compiler.macro_engine.defmacro_native(name, expander)

        # defmacro doesn't return a runtime value
        return None


# Registry of all special forms
special_forms = {
    "->": ThreadFirstForm(),
    "as->": ThreadAsForm(),
    "case": CaseForm(),
    "defmacro": DefmacroForm(),
    "defn": DefnForm(),
    "get": GetForm(),
    "guard": GuardForm(),
    "if": IfForm(),
    "lambda": LambdaForm(),
    "last": LastForm(),
    "let": LetForm(),
    "quote": QuoteForm(),
    "repeat": RepeatForm(),
    "static": StaticForm(),
    "use": UseForm(),
    "with-params": WithParamsForm(),
}
