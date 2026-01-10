# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Parses Sheaf S-expressions and lowers them into JAX-compatible computation graphs.
"""

import builtins
import os

from ..runtime import core_ops, jax_ops, math_ops, nn_ops
from .error_handler import format_error, set_source
from .parser import SheafRuntimeError, SheafSyntaxError, parse_full
from .tracer import sheaf_probe, shf_tracer


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class Sheaf:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.lib_dir = os.path.join(self.base_dir, "lib")
        self.load_path = [self.lib_dir, "."]

        self.env = {}
        # Runtime load
        self.env.update(core_ops.get_core_env())
        self.env.update(jax_ops.get_jax_env())
        self.env.update(math_ops.get_math_env())
        self.env.update(nn_ops.get_nn_env())

        self.env.update(
            {
                "...": Ellipsis,
                "False": False,
                "True": True,
                "concat": lambda *args: "".join(map(str, args)),
                "false": False,
                "len": len,
                "probe": sheaf_probe,
                "second": lambda x: x[1],
                "sparse-cross-entropy": nn_ops.sparse_cross_entropy,
                "str": str,
                "true": True,
            }
        )
        self.trace = False
        self.registry = {}

    def compile(self, exp, local_vars):
        try:
            if local_vars is None:
                local_vars = {}

            if isinstance(exp, (int, float, bool)):
                return exp

            # --- Symbol and String Resolution ---
            if isinstance(exp, str):
                # String literal
                if exp.startswith('"') and exp.endswith('"'):
                    return exp.strip('"')

                # Local variable
                # We check local_vars explicitly
                if exp in local_vars:
                    return local_vars[exp]

                # Global environment (functions, constants, ops)
                if exp in self.env:
                    return self.env[exp]

                # Keywords
                if exp.startswith(":"):
                    return exp

                # If not found in Sheaf hierarchy, we block it to prevent
                # Python from leaking global modules into our math.

                if hasattr(builtins, exp) or exp in globals():
                    # This is likely where the 'module' leak happens.
                    # We should NOT allow Sheaf to pick up Python globals.
                    pass
                line_info = f" (line {exp.line})" if hasattr(exp, "line") else ""
                raise NameError(f"Symbol not found {line_info}: '{exp}'")

        except Exception as e:
            # Get the function name from context
            func_name = local_vars.get("__current_func__", "top-level")

            formatted_msg = format_error(e, exp, func_name)
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None

        op = exp[0]
        args = exp[1:]

        if isinstance(op, str) and op.startswith(":"):
            return [self.compile(x, local_vars) for x in exp]

        # --- Tensor Literal Syntax ---
        # Detect if it's a vector literal: starts with a number OR is a nested list of numbers
        is_tensor_literal = False
        if isinstance(op, (int, float)):
            is_tensor_literal = True
        elif isinstance(op, list) and len(op) > 0 and isinstance(op[0], (int, float)):
            is_tensor_literal = True

        if is_tensor_literal:
            import jax.numpy as jnp

            # Recursively ensure all elements are processed
            def finalize_literal(item):
                if isinstance(item, list):
                    return [finalize_literal(x) for x in item]
                return item

            return jnp.array(finalize_literal(exp))

        try:
            # --- Special ---

            if op == "->":
                # (-> x (layer1) (layer2))
                # x becomes the first arguments of the following function
                current_expr = args[0]
                for step in args[1:]:
                    if isinstance(step, list):
                        # (func arg1) -> becomes (func current_expr arg1)
                        # We insert current_expr just after the function name
                        func_name = step[0]
                        rest_args = step[1:]
                        current_expr = [func_name, current_expr] + rest_args
                    else:
                        # If 'func' alone (ex: relu) -> becomes (func current_expr)
                        current_expr = [step, current_expr]
                # Compile final expression
                return self.compile(current_expr, local_vars)

            if op == "as->":
                # (as-> initial_val name step1 step2 ...)
                val_exp = args[0]
                var_name = args[1]
                steps = args[2:]

                # Eval intial value
                current_val = self.compile(val_exp, local_vars)

                # Create a local context to avoid polluting
                # But permit to use 'name' in next steps
                context = dict(local_vars)

                for step in steps:
                    context[var_name] = current_val
                    current_val = self.compile(step, context)

                return current_val

            if op == "case":
                # (case target val1 result1 val2 result2 ... default)
                target_val = self.compile(args[0], local_vars)

                # Iterate through pairs
                for i in range(1, len(args) - 1, 2):
                    case_val = self.compile(args[i], local_vars)
                    if target_val == case_val:
                        return self.compile(args[i + 1], local_vars)

                # If odd number of arguments, the last one is the default
                if len(args) % 2 == 0:
                    return self.compile(args[-1], local_vars)
                return None

            if op == "defn":
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
                    original_trace_state = getattr(self, "trace", False)
                    if trace_call:
                        self.trace = True
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
                            res = self.compile(expression, context)

                        return tuple(res) if isinstance(res, list) else res

                    finally:
                        # 4. Clean up trace state
                        if trace_call:
                            shf_tracer.enabled = False
                            self.trace = original_trace_state

                # Apply JAX JIT if requested
                if is_jit:
                    import jax

                    static_argnums = []
                    if "config" in params:
                        static_argnums.append(params.index("config"))

                    # Create a wrapper to ensure dictionaries are hashable for JAX
                    base_func = generated_func

                    def jitted_wrapper(*args, **kwargs):
                        # Extract trace/log/scope kwargs BEFORE passing to jax.jit
                        # These cannot be passed through JIT as they cause TracerBoolConversionError
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

                    # Mark as JIT-compiled for tracing
                    generated_func._sheaf_is_jit = True

                # Register the function in both registry (for Python)
                # and env (for internal Sheaf calls)
                self.registry[name] = generated_func
                self.env[name] = generated_func
                # Crucial: we do NOT use setattr(self, name, generated_func)
                # to let __getattr__ handle the "magic" link.
                return generated_func

            if op == "get":
                # (get obj key1 key2 ...)
                # Compile the object and all keys
                obj = self.compile(args[0], local_vars)
                raw_keys = [self.compile(k, local_vars) for k in args[1:]]

                # Strip ':' prefix from keyword symbols
                clean_keys = [
                    k[1:] if isinstance(k, str) and k.startswith(":") else k
                    for k in raw_keys
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

            if op == "guard":
                # Format: (guard :type x) or (guard :type expected x)
                # Examples: (guard :no-nan x)
                #           (guard :shape [64 256] x)
                #           (guard :range [-1 1] x)
                shf_tracer.monitoring = True

                guard_type = args[0]

                if guard_type == ":no-nan":
                    # (guard :no-nan x)
                    val_expr = args[1]
                    val = self.compile(val_expr, local_vars)
                    return shf_tracer.trigger_guard(":no-nan", val)

                elif guard_type in (":shape", ":range"):
                    # (guard :shape expected x) or (guard :range expected x)
                    # IMPORTANT: Compile expected value
                    expected = self.compile(args[1], local_vars)
                    val_expr = args[2]
                    val = self.compile(val_expr, local_vars)
                    return shf_tracer.trigger_guard(guard_type, val, expected)

                raise SheafRuntimeError(f"Unknown guard type: {guard_type}", exp)

            if op == "if":
                cond = self.compile(args[0], local_vars)
                return (
                    self.compile(args[1], local_vars)
                    if cond
                    else self.compile(args[2], local_vars)
                )

            if op == "lambda":
                # Format: (lambda (params) body)
                l_params, *l_body = args

                # Capture the current local_vars at definition time
                # We use a default argument to 'freeze' the current context
                def anonymous_func(*l_args, closure_env=dict(local_vars)):
                    # Merge closure_env with current lambda arguments
                    l_context = dict(closure_env)
                    l_context.update(dict(zip(l_params, l_args)))

                    res = None
                    for expr in l_body:
                        res = self.compile(expr, l_context)
                    return res

                return anonymous_func

            if op == "last":
                return self.compile(args[0], local_vars)[-1]

            if op == "let":
                # args is [bindings_list, body_exp1, body_exp2, ...]
                bindings, *body = args

                # We copy the context to avoid polluting the parent scope
                current_context = dict(local_vars)

                # Process pairs: (var1 val1 var2 val2 ...)
                for i in range(0, len(bindings), 2):
                    target = bindings[i]
                    # We compile the value using the context updated by previous pairs
                    val = self.compile(bindings[i + 1], current_context)

                    if isinstance(target, list):  # Support for [a b] (split key)
                        for name, v in zip(target, val):
                            current_context[name] = v
                    else:
                        current_context[target] = val

                # Execute the body with the final context
                res = None
                for expression in body:
                    res = self.compile(expression, current_context)
                return res

            if op == "repeat":
                # Syntax: (repeat [i 6] [acc_name init_val] body)

                binding_iter = args[0]  # [i, 6]
                idx_name, count_expr = binding_iter[0], binding_iter[1]
                count = self.compile(count_expr, local_vars)

                binding_acc = args[1]  # [h, x]
                acc_name, init_expr = binding_acc[0], binding_acc[1]
                current_val = self.compile(init_expr, local_vars)

                body = args[2]

                for i in range(int(count)):
                    # Context for this iteration
                    loop_ctx = dict(local_vars)
                    loop_ctx[idx_name] = i
                    loop_ctx[acc_name] = current_val  # Inject previous value

                    # Evaluate body
                    current_val = self.compile(body, loop_ctx)

                return current_val

            if op == "static":
                # args[0] is the expression inside (static ...)
                val = self.compile(args[0], local_vars)
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
                    # We use args to avoid UnboundLocalError
                    raise SheafRuntimeError(msg, args) from None

            # In your compiler/evaluator class
            if op == "use":
                # Clean the input name
                raw_name = str(args[0]).strip('"')

                file_path = None
                extensions = ["", ".shf"]

                # If raw_name is already a path (contains / or \), we check it directly
                # Otherwise, we search in load_path
                search_roots = (
                    self.load_path
                    if not os.path.isabs(raw_name) and "/" not in raw_name
                    else [""]
                )

                for root in search_roots:
                    for ext in extensions:
                        potential_path = os.path.join(root, raw_name + ext)
                        if os.path.exists(potential_path) and os.path.isfile(
                            potential_path
                        ):
                            file_path = potential_path
                            break
                    if file_path:
                        break

                if file_path is None:
                    raise SheafRuntimeError(
                        f"Module '{raw_name}' not found. Searched in: {self.load_path}",
                        exp,
                    )

                try:
                    with open(file_path, "r") as f:
                        module_code = f.read()

                    expressions = parse_full(module_code)
                    for expr in expressions:
                        self.compile(expr, {})

                    return None
                except Exception as e:
                    raise SheafRuntimeError(
                        f"Error loading module {file_path}: {str(e)}", exp
                    )

            if op == "with-params":
                p_expr, *body = args
                p_val = self.compile(p_expr, local_vars)
                context = dict(local_vars)
                if isinstance(p_val, dict):
                    for k, v in p_val.items():
                        # Normalize the key in case it starts with ":"
                        clean_k = (
                            k[1:] if isinstance(k, str) and k.startswith(":") else k
                        )
                        context[clean_k] = v

                res = None
                for e in body:
                    res = self.compile(e, context)
                return res

        except Exception as e:
            if getattr(e, "is_sheaf_error", False):
                raise e

            formatted_msg = format_error(e, exp, "top-level")
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None

        # --- Standard Function Call ---

        try:
            func = self.compile(op, local_vars)

            import types

            if isinstance(func, types.ModuleType):
                raise TypeError(
                    f"Symbol '{op}' resolved to a module instead of a function."
                )

            if not callable(func):
                raise TypeError(f"Symbol '{op}' is not callable (Type: {type(func)}).")

            real_args, kwargs, i = [], {}, 0

            if isinstance(func, str):
                raise TypeError(f"Unknown function '{func}'.")

            # --- Trace Start: Log call BEFORE evaluating arguments ---
            is_jit_func = getattr(func, "_sheaf_is_jit", False)

            if getattr(self, "trace", False) or shf_tracer.monitoring:
                if is_jit_func:
                    # Special handling for JIT functions
                    shf_tracer.log_jit_call(op)
                else:
                    shf_tracer.log_call(op, [], {})

            real_args, kwargs, i = [], {}, 0
            is_dict_op = op == "dict"

            while i < len(args):
                if (
                    not is_dict_op
                    and isinstance(args[i], str)
                    and args[i].startswith(":")
                    and (i + 1) < len(args)
                ):
                    # Evaluate keyword argument
                    arg_expr = args[i + 1]
                    is_nested_call = isinstance(arg_expr, list) and len(arg_expr) > 0

                    val = self.compile(arg_expr, local_vars)
                    kwargs[args[i][1:]] = val

                    # Only log simple values, nested calls log themselves
                    # Skip logging for JIT functions
                    if (
                        (getattr(self, "trace", False) or shf_tracer.monitoring)
                        and not is_nested_call
                        and not is_jit_func
                    ):
                        shf_tracer.log_arg(val, name=args[i][1:])
                    i += 2
                else:
                    # Evaluate positional argument
                    arg_expr = args[i]
                    is_nested_call = isinstance(arg_expr, list) and len(arg_expr) > 0

                    val = self.compile(arg_expr, local_vars)
                    real_args.append(val)

                    # Only log simple values, nested calls log themselves
                    # Skip logging for JIT functions
                    if (
                        (getattr(self, "trace", False) or shf_tracer.monitoring)
                        and not is_nested_call
                        and not is_jit_func
                    ):
                        shf_tracer.log_arg(val)
                    i += 1

            res = func(*real_args, **kwargs)

            # --- Trace End ---
            # Skip log_return for JIT functions as they show cached message
            if (
                getattr(self, "trace", False) or shf_tracer.monitoring
            ) and not is_jit_func:
                shf_tracer.log_return(op, res)

            return res

        except Exception as e:
            # If it's already a SheafRuntimeError, let it propagate
            if isinstance(e, SheafRuntimeError):
                raise e

            # Otherwise, wrap it with current context
            func_name = local_vars.get("__current_func__", "top-level")

            formatted_msg = format_error(e, exp, func_name)

            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e  # Keep reference to original
            raise error from None

    def load(self, code, filename="<sheaf>"):
        # Store source code for better error messages
        set_source(code, filename)

        try:
            expressions = parse_full(code)  # Call global parser
            for ast in expressions:
                self.compile(ast, {})
        except SheafSyntaxError as e:
            # Create a fake expression with line info for the formatter
            class FakeSyntaxExp:
                def __init__(self, line):
                    self.line = line

                def __repr__(self):
                    return "<syntax error>"

            exp = FakeSyntaxExp(e.line_num) if e.line_num else None
            formatted_msg = format_error(e, exp, "parsing")
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None

        return self.registry
