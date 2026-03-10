// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Code generation - translate CompiledExpr to StableHLO MLIR

mod autodiff;
mod builtins;
mod control_flow;
mod helpers;
#[cfg(test)]
mod tests;

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
pub(crate) use helpers::{TupleLeaf, collect_tuple_leaves, expand_tuple_to_symbols};
use helpers::try_flatten_to_constant;
use std::collections::HashMap;

/// Code generator - converts CompiledExpr to StableHLO
pub struct CodeGenerator {
    emitter: StableHLOEmitter,
    /// Map from variable names to registers and their types
    bindings: HashMap<String, (Register, StableHLOType)>,
    /// Lambdas bound in let forms — stored for inlining, not emitted as SSA.
    lambda_bindings: HashMap<String, CompiledExpr>,
    /// Function registry for user-defined functions
    function_registry: HashMap<String, crate::core::compiler::FunctionDef>,
    /// Key-to-index layout for tuple-typed variables.
    /// Allows `(get sym "key")` to resolve when `sym` is bound to a tuple.
    /// Populated from param configs and propagated into reduce/scan lambdas.
    tuple_key_layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    /// Reverse map: (param_name, tuple_index) -> key_name.
    /// Used to resolve layouts for GetTupleElement collections in reduce/scan.
    idx_to_key: HashMap<(String, usize), String>,
    /// Map from SSA register to layout key name.
    /// Tracks which layout key a register corresponds to, enabling
    /// `(get (get x :k1) :k2)` where the outer get's operand is not a Symbol.
    layout_key_map: HashMap<Register, String>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            lambda_bindings: HashMap::new(),
            function_registry: HashMap::new(),
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
        }
    }

    pub fn with_registry(registry: HashMap<String, crate::core::compiler::FunctionDef>) -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            lambda_bindings: HashMap::new(),
            function_registry: registry,
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
        }
    }

    /// Create a CodeGenerator with function parameters bound to %arg0, %arg1, etc.
    pub fn with_function_params(
        registry: HashMap<String, crate::core::compiler::FunctionDef>,
        param_names: &[String],
        param_types: &[StableHLOType],
    ) -> Self {
        let mut bindings = HashMap::new();
        for (i, (name, ty)) in param_names.iter().zip(param_types.iter()).enumerate() {
            bindings.insert(name.clone(), (Register::arg(i), ty.clone()));
        }

        Self {
            emitter: StableHLOEmitter::new(),
            bindings,
            lambda_bindings: HashMap::new(),
            function_registry: registry,
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
        }
    }

    /// Set key-to-index layouts for tuple-typed symbols (e.g. `hidden -> {"W":0, "b":1}`).
    /// Called before codegen when dict param configs are available.
    pub fn set_tuple_key_layouts(
        &mut self,
        layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    ) {
        self.tuple_key_layouts = layouts;
    }

    pub fn set_idx_to_key(&mut self, map: HashMap<(String, usize), String>) {
        self.idx_to_key = map;
    }

    /// Bind a symbol name to an SSA register (for use by external codegen paths).
    pub fn bind_symbol(&mut self, name: &str, reg: Register, ty: StableHLOType) {
        self.bindings.insert(name.to_string(), (reg, ty));
    }

    /// Return a map of symbol name → shape for all current bindings.
    /// Used by reverse-mode AD to generate shape-aware gradient expressions.
    pub fn binding_shapes(&self) -> std::collections::HashMap<String, Vec<i64>> {
        self.bindings
            .iter()
            .map(|(name, (_, ty))| (name.clone(), ty.shape().to_vec()))
            .collect()
    }

    /// Generate and bind a single Let binding in the current scope.
    /// Handles Lambda storage, destructuring, and layout propagation —
    /// same logic as the Let codegen but without scope save/restore.
    pub fn generate_binding(&mut self, name: &str, value_expr: &CompiledExpr) -> SheafResult<()> {
        if matches!(value_expr, CompiledExpr::Lambda { .. }) {
            self.lambda_bindings.insert(name.to_string(), value_expr.clone());
        } else if name.starts_with('[') && name.ends_with(']') {
            let names: Vec<&str> = name[1..name.len() - 1].split_whitespace().collect();
            let (tuple_reg, tuple_ty) = self.generate(value_expr)?;
            let element_types = match &tuple_ty {
                StableHLOType::Tuple(tys) => tys.clone(),
                other => {
                    return Err(SheafError::Compile {
                        message: format!("Let destructuring requires a tuple, got: {}", other.to_mlir()),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            };
            for (i, n) in names.iter().enumerate() {
                let elem_reg = self.emitter.emit_get_tuple_element(
                    &tuple_reg, &tuple_ty, i, &element_types[i],
                );
                self.bindings.insert(n.to_string(), (elem_reg, element_types[i].clone()));
            }
        } else {
            let (reg, ty) = self.generate(value_expr)?;
            if matches!(&ty, StableHLOType::Tuple(_)) {
                if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args } = value_expr {
                    if fn_name == "get" && fn_args.len() >= 2 {
                        if let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last() {
                            if let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned() {
                                self.tuple_key_layouts.insert(name.to_string(), sub_layout);
                            }
                        }
                    }
                } else if let CompiledExpr::Symbol(src) = value_expr {
                    if let Some(layout) = self.tuple_key_layouts.get(src).cloned() {
                        self.tuple_key_layouts.insert(name.to_string(), layout);
                    }
                }
            }
            self.bindings.insert(name.to_string(), (reg, ty));
        }
        Ok(())
    }

    /// Emit a GetTupleElement operation (delegates to emitter).
    pub fn emit_get_tuple_element(
        &mut self,
        tuple_reg: &Register,
        tuple_ty: &StableHLOType,
        index: usize,
        elem_ty: &StableHLOType,
    ) -> Register {
        self.emitter
            .emit_get_tuple_element(tuple_reg, tuple_ty, index, elem_ty)
    }

    /// Generate StableHLO for a compiled expression
    pub fn generate(&mut self, expr: &CompiledExpr) -> SheafResult<(Register, StableHLOType)> {
        match expr {
            CompiledExpr::Integer(n) => {
                // Treat integers as floats for now (matches Python behavior)
                let reg = self.emitter.emit_constant_f32(*n as f64);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Float(x) => {
                let reg = self.emitter.emit_constant_f32(*x);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Vector(elements) => {
                match try_flatten_to_constant(elements) {
                    Some((data, shape)) => {
                        let (reg, ty) = self.emitter.emit_nd_tensor_constant(&data, &shape);
                        Ok((reg, ty))
                    }
                    None => {
                        // Non-constant vector: emit as tuple (e.g. [new-p new-m new-v new-t])
                        let mut regs = Vec::new();
                        let mut tys = Vec::new();
                        for elem in elements {
                            let (reg, ty) = self.generate(elem)?;
                            regs.push(reg);
                            tys.push(ty);
                        }
                        Ok(self.emitter.emit_tuple(&regs, &tys))
                    }
                }
            }

            CompiledExpr::Symbol(name) => {
                // Look up symbol in bindings
                if let Some(&(reg, ref ty)) = self.bindings.get(name) {
                    Ok((reg, ty.clone()))
                } else {
                    Err(SheafError::Compile {
                        message: format!("Undefined symbol in codegen: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }

            CompiledExpr::GetTupleElement { param, indices } => {
                // Resolve a field extracted by with-params via get_tuple_element
                // The param must be in bindings with a Tuple type
                let (param_reg, param_ty) =
                    self.bindings
                        .get(param)
                        .cloned()
                        .ok_or_else(|| SheafError::Compile {
                            message: format!(
                                "GetTupleElement: parameter '{}' not found in bindings",
                                param
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        })?;

                // Walk nested tuple type and emit get_tuple_element for each index
                let mut current_reg = param_reg;
                let mut current_ty = param_ty;

                for &idx in indices {
                    let element_ty = match &current_ty {
                        StableHLOType::Tuple(elems) => elems.get(idx).cloned().ok_or_else(|| {
                            SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: index {} out of range for tuple with {} elements",
                                    idx,
                                    elems.len()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            }
                        })?,
                        other => {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: expected tuple type, got {}",
                                    other.to_mlir()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                    };
                    let result_reg = self.emitter.emit_get_tuple_element(
                        &current_reg,
                        &current_ty,
                        idx,
                        &element_ty,
                    );
                    current_reg = result_reg;
                    current_ty = element_ty;
                }

                // Track layout key for the result register via idx_to_key chain
                {
                    let mut cur = param.clone();
                    for &idx in indices {
                        if let Some(key) = self.idx_to_key.get(&(cur.clone(), idx)) {
                            cur = key.clone();
                        } else {
                            cur = String::new();
                            break;
                        }
                    }
                    if !cur.is_empty() {
                        self.layout_key_map.insert(current_reg, cur);
                    }
                }

                Ok((current_reg, current_ty))
            }

            CompiledExpr::FunctionCall { name, args } => self.generate_function_call(name, args),

            CompiledExpr::Let { bindings, body } => {
                let saved_bindings = self.bindings.clone();
                let saved_lambda_bindings = self.lambda_bindings.clone();
                for (name, value_expr) in bindings {
                    if matches!(value_expr, CompiledExpr::Lambda { .. }) {
                        // Store lambda for inlining — no SSA emitted.
                        self.lambda_bindings
                            .insert(name.clone(), value_expr.clone());
                    } else if name.starts_with('[') && name.ends_with(']') {
                        // Destructuring bind: [a b c] = tuple -> get_tuple_element
                        let names: Vec<&str> =
                            name[1..name.len() - 1].split_whitespace().collect();
                        let (tuple_reg, tuple_ty) = self.generate(value_expr)?;
                        let element_types = match &tuple_ty {
                            StableHLOType::Tuple(tys) => tys.clone(),
                            other => {
                                return Err(SheafError::Compile {
                                    message: format!(
                                        "Let destructuring requires a tuple, got: {}",
                                        other.to_mlir()
                                    ),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            }
                        };
                        for (i, n) in names.iter().enumerate() {
                            let elem_reg = self.emitter.emit_get_tuple_element(
                                &tuple_reg,
                                &tuple_ty,
                                i,
                                &element_types[i],
                            );
                            self.bindings
                                .insert(n.to_string(), (elem_reg, element_types[i].clone()));
                        }
                    } else {
                        let (reg, ty) = self.generate(value_expr)?;
                        // Propagate sub-layout for Let-bound tuples
                        if matches!(&ty, StableHLOType::Tuple(_)) {
                            if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args } = value_expr {
                                if fn_name == "get" && fn_args.len() >= 2 {
                                    // Use the last keyword arg as the layout key
                                    if let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last() {
                                        if let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned() {
                                            self.tuple_key_layouts.insert(name.clone(), sub_layout);
                                        }
                                    }
                                }
                            }
                            // Symbol alias: (let [x y]) where y has a layout
                            else if let CompiledExpr::Symbol(src) = value_expr {
                                if let Some(layout) = self.tuple_key_layouts.get(src).cloned() {
                                    self.tuple_key_layouts.insert(name.clone(), layout);
                                }
                            }
                        }
                        self.bindings.insert(name.clone(), (reg, ty));
                    }
                }
                let result = self.generate(body)?;
                self.bindings = saved_bindings;
                self.lambda_bindings = saved_lambda_bindings;
                Ok(result)
            }

            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let (cond_reg, cond_ty) = self.generate(condition)?;
                let (then_reg, then_ty) = self.generate(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let (else_reg, else_ty) = self.generate(else_expr)?;
                    // Use stablehlo.select: result = select(cond, then, else)
                    let (result_reg, result_ty) = self.emitter.emit_select(
                        &cond_reg, &then_reg, &else_reg, &cond_ty, &then_ty, &else_ty,
                    );
                    Ok((result_reg, result_ty))
                } else {
                    // If without else: just return then_branch
                    // (assumes condition is always true for now)
                    Ok((then_reg, then_ty))
                }
            }

            CompiledExpr::Do(exprs) => {
                // Evaluate all expressions, return the last one
                let mut last_result = None;
                for expr in exprs {
                    last_result = Some(self.generate(expr)?);
                }
                last_result.ok_or_else(|| SheafError::Compile {
                    message: "do requires at least one expression".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }

            CompiledExpr::FunctionRef(name) => Err(SheafError::Compile {
                message: format!("Cannot generate code for bare function reference: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            // Lambda and LambdaCall: inline at call site.
            // A bare Lambda without a call is not directly emittable.
            CompiledExpr::Lambda { .. } => Err(SheafError::Compile {
                message: "Cannot emit a lambda without a call site".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::LambdaCall { callee, args } => {
                let callee = callee.clone();
                let args = args.clone();
                self.inline_lambda_call(&callee, &args)
            }

            CompiledExpr::ValueAndGrad { fn_name, .. } => Err(SheafError::Compile {
                message: format!(
                    "ValueAndGrad '{}' is a module-level form, not an inline expression",
                    fn_name
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::InlineValueAndGrad {
                lambda,
                args,
                wrt_indices,
            } => {
                let lambda = lambda.clone();
                let args = args.clone();
                let wrt_indices = wrt_indices.clone();
                self.generate_inline_value_and_grad(&lambda, &args, &wrt_indices)
            }

            CompiledExpr::Dict(pairs) => {
                // Emit dict as a tuple, keys sorted alphabetically for deterministic layout
                let mut sorted: Vec<_> = pairs.iter().collect();
                sorted.sort_by(|(k1, _), (k2, _)| {
                    let key1 = match k1 {
                        CompiledExpr::Keyword(k) => k.as_str(),
                        _ => "",
                    };
                    let key2 = match k2 {
                        CompiledExpr::Keyword(k) => k.as_str(),
                        _ => "",
                    };
                    key1.cmp(key2)
                });
                let mut regs = Vec::new();
                let mut tys = Vec::new();
                for (_, val) in &sorted {
                    let (r, t) = self.generate(val)?;
                    regs.push(r);
                    tys.push(t);
                }
                Ok(self.emitter.emit_tuple(&regs, &tys))
            }

            _ => Err(SheafError::Compile {
                message: format!("Code generation not yet implemented for: {:?}", expr),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    /// Extract a static shape Vec<i64> from a Vector of Integer literals.
    /// Returns a compile error (not panic) if any element is non-constant,
    /// so the function can be gracefully skipped rather than crashing.
    fn parse_shape_vec(elems: &[CompiledExpr]) -> SheafResult<Vec<i64>> {
        elems.iter().map(|e| match e {
            CompiledExpr::Integer(n) => Ok(*n),
            other => Err(SheafError::Compile {
                message: format!(
                    "shape element must be a constant integer, got: {:?} \
                     (use (static ...) or --trace-with to resolve)",
                    other
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }).collect()
    }

    /// Emit a complete function module
    pub fn emit_function(mut self, name: &str, expr: &CompiledExpr) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        Ok(self.emitter.emit_function_body(name, &result_ty))
    }

    /// Emit a function declaration from a compiled expression
    ///
    /// Generates the body instructions and wraps them in a func.func declaration
    /// with the given parameter types and return type
    /// Returns (mlir_declaration, actual_return_type).
    pub fn emit_func_declaration(
        mut self,
        name: &str,
        expr: &CompiledExpr,
        param_types: &[StableHLOType],
        _return_type: &StableHLOType,
    ) -> SheafResult<(String, StableHLOType)> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        // Use the actual generated type (not inferred) as the return type
        let body = self.emitter.body.clone();
        let decl = self
            .emitter
            .emit_func_declaration(name, param_types, &result_ty, &body);
        Ok((decl, result_ty))
    }

    /// Finalize a multi-output function declaration.
    ///
    /// Emits a `return %r0, %r1, ...` then wraps everything in a `func.func`
    /// with a multi-value return type `-> (t0, t1, ...)`.
    ///
    /// The caller has already called `generate()` for each output and collected
    /// the resulting (Register, StableHLOType) pairs.
    pub fn finish_multi(
        mut self,
        name: &str,
        param_types: &[StableHLOType],
        result_regs: &[crate::compiler::stablehlo::Register],
        result_types: &[StableHLOType],
    ) -> String {
        self.emitter.emit_return_multi(result_regs, result_types);
        let body = self.emitter.body.clone();
        self.emitter
            .emit_func_declaration_multi(name, param_types, result_types, &body)
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}
