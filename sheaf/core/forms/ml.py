# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Machine learning special forms: vmap, scan, with-params, with-dtype, static
"""

from ..parser import SheafRuntimeError, SheafVector
from .base import SpecialForm


class VmapForm(SpecialForm):
    """vmap vectorized mapping: (vmap f) or (vmap f in-axes)"""

    def __init__(self):
        super().__init__("vmap")

    def compile(self, compiler, args, local_vars):
        """
        Apply JAX vmap to a function.

        Syntax:
            (vmap f)              ; vmap all args over axis 0
            (vmap f 0)            ; vmap all args over specified axis
            (vmap f [0 nil])      ; vmap 1st arg on axis 0, don't vmap 2nd
            (vmap f [0 nil nil])  ; vmap 1st arg, keep 2nd and 3rd fixed

        Returns a new function that applies f independently across batch dimension.

        Examples:
            (vmap square)           ; Batch all arguments
            (vmap linear [0 nil nil])  ; Only batch first arg (X), keep W and b fixed
        """
        import jax

        if len(args) < 1 or len(args) > 2:
            raise ValueError(
                "vmap requires 1 or 2 arguments: (vmap f) or (vmap f in-axes)"
            )

        # Get the function
        func_expr = args[0]
        func = compiler.compile(func_expr, local_vars)

        # Get in_axes (default to 0 = vmap all args)
        in_axes = 0
        if len(args) == 2:
            axes_expr = args[1]

            # Handle list of axes: [0 nil] or [0 nil nil]
            # Don't compile the list - process it directly to preserve 'nil' symbols
            if isinstance(axes_expr, list):
                processed_axes = []
                for ax in axes_expr:
                    if ax == "nil" or ax is None:
                        processed_axes.append(None)
                    else:
                        # Compile numeric values
                        compiled_ax = compiler.compile(ax, local_vars)
                        if hasattr(compiled_ax, "item"):
                            processed_axes.append(int(compiled_ax.item()))
                        else:
                            processed_axes.append(int(compiled_ax))
                in_axes = tuple(processed_axes)
            else:
                # Single axis value
                axes = compiler.compile(axes_expr, local_vars)
                if hasattr(axes, "item"):
                    in_axes = int(axes.item())
                else:
                    in_axes = int(axes)

        # Apply vmap and return the vmapped function
        vmapped_func = jax.vmap(func, in_axes=in_axes)
        return vmapped_func


class ScanForm(SpecialForm):
    """scan looping primitive: (scan f init xs)"""

    def __init__(self):
        super().__init__("scan")

    def compile(self, compiler, args, local_vars):
        """
        Apply JAX scan for functional looping.

        Syntax:
            (scan f init xs)

        Where:
            f    : function (carry, x) -> (carry, y)
            init : initial carry value
            xs   : sequence tensor to iterate over

        Returns: (final-carry, ys)

        Example:
            (defn step [state x]
              (let (new-state (+ state x))
                [new-state new-state]))

            (scan step 0 [1 2 3 4])
            ; => [10, [1 3 6 10]]
        """
        import jax

        if len(args) != 3:
            raise ValueError("scan requires exactly 3 arguments: (scan f init xs)")

        # Get the function
        func_expr = args[0]
        func = compiler.compile(func_expr, local_vars)

        # Get initial carry
        init_expr = args[1]
        init = compiler.compile(init_expr, local_vars)

        # Get sequence to scan over
        xs_expr = args[2]
        xs = compiler.compile(xs_expr, local_vars)

        # Wrap the Sheaf function to ensure it returns a tuple
        def scan_func(carry, x):
            result = func(carry, x)
            # Ensure result is a tuple (carry, y)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                return result
            else:
                # If function returns single value, use it as both carry and output
                return (result, result)

        # Apply jax.lax.scan
        final_carry, ys = jax.lax.scan(scan_func, init, xs)

        # Return as a list [final_carry, ys] for Sheaf consumption
        return [final_carry, ys]


class WithParamsForm(SpecialForm):
    """with-params parameter destructuring: (with-params [p :key] body) or (with-params [expr] body)"""

    def __init__(self):
        super().__init__("with-params")

    def compile(self, compiler, args, local_vars):
        p_expr, *body = args

        # Syntax with brackets:
        # [p]        -> evaluate p (brackets are optional)
        # [p :key]   -> shorthand for [(get p :key)]
        # [(expr)]   -> evaluate expr
        if isinstance(p_expr, SheafVector):
            if (
                len(p_expr) == 2
                and isinstance(p_expr[1], str)
                and p_expr[1].startswith(":")
            ):
                # [dict :key] -> (get dict :key)
                dict_expr = p_expr[0]
                key = p_expr[1]
                dict_val = compiler.compile(dict_expr, local_vars)
                # Normalize key (remove leading colon)
                clean_key = key[1:] if key.startswith(":") else key
                p_val = (
                    dict_val.get(clean_key) if isinstance(dict_val, dict) else dict_val
                )
            elif len(p_expr) == 1:
                # [p] or [(expr)] -> evaluate the single element
                p_val = compiler.compile(p_expr[0], local_vars)
            else:
                raise ValueError(
                    "with-params expects [p], [p :key], or [(expr)], got vector with "
                    f"{len(p_expr)} elements"
                )
        else:
            # Legacy syntax: (with-params expr body) - still supported
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


class WithDtypeForm(SpecialForm):
    """with-dtype dtype body - temporarily change tensor dtype"""

    def __init__(self):
        super().__init__("with-dtype")

    def compile(self, compiler, args, local_vars):
        """
        Temporarily change dtype for all tensor creations in body.

        Syntax:
            (with-dtype :f32 body...)
            (with-dtype :bf16 body...)
            (with-dtype :f64 body...)
        """
        if len(args) < 2:
            raise ValueError("with-dtype requires dtype and body expressions")

        # Get dtype keyword
        dtype_keyword = args[0]
        if isinstance(dtype_keyword, str) and dtype_keyword.startswith(":"):
            dtype_map = {
                ":f32": "float32",
                ":bf16": "bfloat16",
                ":f64": "float64",
            }
            dtype = dtype_map.get(dtype_keyword)
            if not dtype:
                raise ValueError(
                    f"Unknown dtype: {dtype_keyword}. Use :f32, :bf16, or :f64"
                )
        else:
            raise ValueError(
                f"dtype must be a keyword (:f32, :bf16, :f64), got {dtype_keyword}"
            )

        # Save current dtype
        old_dtype = compiler.dtype
        compiler.dtype = dtype

        try:
            # Execute body with new dtype
            result = None
            for expr in args[1:]:
                result = compiler.compile(expr, local_vars)
            return result
        finally:
            # Restore old dtype
            compiler.dtype = old_dtype


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
