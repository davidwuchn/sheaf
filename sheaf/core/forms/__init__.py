# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Special forms registry for Sheaf compiler.

Organized into modules:
- base: Base SpecialForm class and utilities
- control: if, case, guard, repeat
- binding: defn, lambda, let, defmacro
- flow: ->, as->
- ml: vmap, scan, with-params, with-dtype, static
- utils: get, dict, last, use, quote
"""

from .binding import DefmacroForm, DefnForm, LambdaForm, LetForm
from .control import CaseForm, GuardForm, IfForm, RepeatForm
from .flow import ThreadAsForm, ThreadFirstForm
from .ml import ScanForm, StaticForm, VmapForm, WithDtypeForm, WithParamsForm
from .utils import DictForm, GetForm, LastForm, QuoteForm, UseForm

# Registry of all special forms
special_forms = {
    "->": ThreadFirstForm(),
    "as->": ThreadAsForm(),
    "case": CaseForm(),
    "defmacro": DefmacroForm(),
    "defn": DefnForm(),
    "dict": DictForm(),
    "fn": LambdaForm(),
    "get": GetForm(),
    "guard": GuardForm(),
    "if": IfForm(),
    "lambda": LambdaForm(),
    "last": LastForm(),
    "let": LetForm(),
    "repeat": RepeatForm(),
    "scan": ScanForm(),
    "static": StaticForm(),
    "use": UseForm(),
    "vmap": VmapForm(),
    "with-params": WithParamsForm(),
    "with-dtype": WithDtypeForm(),
}

__all__ = [
    "special_forms",
    "ThreadFirstForm",
    "ThreadAsForm",
    "CaseForm",
    "DefmacroForm",
    "DefnForm",
    "DictForm",
    "LambdaForm",
    "GetForm",
    "GuardForm",
    "IfForm",
    "LastForm",
    "LetForm",
    "RepeatForm",
    "ScanForm",
    "StaticForm",
    "UseForm",
    "VmapForm",
    "WithDtypeForm",
    "WithParamsForm",
    "QuoteForm",
]
