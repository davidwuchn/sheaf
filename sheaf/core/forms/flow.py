# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Threading and composition special forms: ->, as->
"""

from .base import SpecialForm


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
