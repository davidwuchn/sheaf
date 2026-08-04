// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! StableHLO code generation from `CompiledExpr`.

mod autodiff;
mod builtins;
mod collection_builtins;
mod control_flow;
mod helpers;
mod math_builtins;
mod reduction_builtins;
mod tensor_builtins;
#[cfg(test)]
mod tests;

use crate::lowering::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::{SheafError, SheafResult};
pub(crate) use helpers::{try_flatten_to_constant, TupleLeaf, collect_tuple_leaves, expand_tuple_to_symbols};
use std::collections::HashMap;

/// Recursively flatten tuple types into leaf tensor types.
pub fn flatten_param_types(param_types: &[StableHLOType]) -> Vec<StableHLOType> {
    let mut flat = Vec::new();
    for ty in param_types {
        flatten_type(ty, &mut flat);
    }
    flat
}

fn flatten_type(ty: &StableHLOType, out: &mut Vec<StableHLOType>) {
    match ty {
        StableHLOType::Tuple(elems, _) => {
            for elem in elems {
                flatten_type(elem, out);
            }
        }
        other => out.push(other.clone()),
    }
}

/// Converts CompiledExpr to StableHLO
pub struct CodeGenerator {
    emitter: StableHLOEmitter,
    bindings: HashMap<String, (Register, StableHLOType)>,
    /// Let-bound lambdas are inlined rather than emitted as SSA values.
    lambda_bindings: HashMap<String, CompiledExpr>,
    function_registry: HashMap<String, crate::core::expr::FunctionDef>,
    /// Dict key layouts for tuple-valued bindings.
    tuple_key_layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    /// Reverse tuple layout lookup used while unrolling collections.
    idx_to_key: HashMap<(String, usize), String>,
    /// Layout keys associated with registers selected from a tuple.
    layout_key_map: HashMap<Register, String>,
    /// Runtime scalar constants needed during code generation.
    scalar_constants: HashMap<(String, Vec<usize>), f64>,
    /// Dict-to-tuple indices used while lowering inlined calls.
    param_index_maps: Vec<(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)>,
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
            scalar_constants: HashMap::new(),
            param_index_maps: Vec::new(),
        }
    }

    pub fn with_registry(registry: HashMap<String, crate::core::expr::FunctionDef>) -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            lambda_bindings: HashMap::new(),
            function_registry: registry,
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
            scalar_constants: HashMap::new(),
            param_index_maps: Vec::new(),
        }
    }

    /// Binds function parameters to MLIR arguments and virtual tuples.
    pub fn with_function_params(
        registry: HashMap<String, crate::core::expr::FunctionDef>,
        param_names: &[String],
        param_types: &[StableHLOType],
    ) -> Self {
        let mut emitter = StableHLOEmitter::new();
        let mut bindings = HashMap::new();
        let mut flat_idx: usize = 0;

        for (name, ty) in param_names.iter().zip(param_types.iter()) {
            let reg = emitter.register_virtual_param(ty, &mut flat_idx);
            bindings.insert(name.clone(), (reg, ty.clone()));
        }

        Self {
            emitter,
            bindings,
            lambda_bindings: HashMap::new(),
            function_registry: registry,
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
            scalar_constants: HashMap::new(),
            param_index_maps: Vec::new(),
        }
    }

    /// Installs dict key layouts before generating a function.
    pub fn set_tuple_key_layouts(
        &mut self,
        layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    ) {
        self.tuple_key_layouts = layouts;
    }

    pub fn set_idx_to_key(&mut self, map: HashMap<(String, usize), String>) {
        self.idx_to_key = map;
    }

    pub fn set_scalar_constants(&mut self, constants: HashMap<(String, Vec<usize>), f64>) {
        self.scalar_constants = constants;
    }

    /// Records scalar parameters needed for shape-dependent operations.
    pub fn set_scalar_param_values(&mut self, values: &[(String, f64)]) {
        for (name, value) in values {
            if let Some(&(reg, _)) = self.bindings.get(name) {
                self.emitter.set_known_scalar(reg, *value);
            }
        }
    }

    pub fn set_param_index_maps(&mut self, maps: Vec<(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)>) {
        self.param_index_maps = maps;
    }

    /// Binds a symbol to an existing SSA register.
    pub fn bind_symbol(&mut self, name: &str, reg: Register, ty: StableHLOType) {
        self.bindings.insert(name.to_string(), (reg, ty));
    }

    /// Returns known binding shapes for reverse-mode autodiff.
    pub fn binding_shapes(&self) -> std::collections::HashMap<String, Vec<i64>> {
        self.bindings
            .iter()
            .map(|(name, (_, ty))| (name.clone(), ty.shape().to_vec()))
            .collect()
    }

    /// Generates one let binding without changing the enclosing scope.
    pub fn generate_binding(&mut self, name: &str, value_expr: &CompiledExpr) -> SheafResult<()> {
        if matches!(value_expr, CompiledExpr::Lambda { .. }) {
            self.lambda_bindings.insert(name.to_string(), value_expr.clone());
        } else if name.starts_with('[') && name.ends_with(']') {
            let names: Vec<&str> = name[1..name.len() - 1].split_whitespace().collect();
            let (tuple_reg, tuple_ty) = self.generate(value_expr)?;
            let element_types = match &tuple_ty {
                StableHLOType::Tuple(tys, _) => tys.clone(),
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
        if matches!(&ty, StableHLOType::Tuple(..)) {
            if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args, .. } = value_expr
                && fn_name == "get"
                && fn_args.len() >= 2
                && let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last()
                && let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned()
            {
                self.tuple_key_layouts.insert(name.to_string(), sub_layout);
            } else if let CompiledExpr::Symbol(src) = value_expr
                && let Some(layout) = self.tuple_key_layouts.get(src).cloned()
            {
                self.tuple_key_layouts.insert(name.to_string(), layout);
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

    /// Try to evaluate an expression to a compile-time constant.
    /// Used for constant-folding `if` conditions (e.g. `(== (ndim x) 3)`).
    fn try_const_eval(&self, expr: &CompiledExpr) -> Option<f64> {
        match expr {
            CompiledExpr::Integer(n) => Some(*n as f64),
            CompiledExpr::Float(f) => Some(*f),
            CompiledExpr::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            CompiledExpr::FunctionCall { name, args, .. } => {
                match name.as_str() {
                    "ndim" if args.len() == 1 => {
                        if let CompiledExpr::Symbol(sym) = &args[0] {
                            let (_, ty) = self.bindings.get(sym.as_str())?;
                            Some(ty.shape().len() as f64)
                        } else {
                            None
                        }
                    }
                    "len" | "length" if args.len() == 1 => {
                        if let CompiledExpr::FunctionCall { name: inner_name, args: inner_args, .. } = &args[0]
                            && inner_name == "shape"
                            && inner_args.len() == 1
                            && let CompiledExpr::Symbol(sym) = &inner_args[0]
                        {
                            let (_, ty) = self.bindings.get(sym.as_str())?;
                            return Some(ty.shape().len() as f64);
                        }
                        None
                    }
                    "==" | "!=" | "<" | ">" | "<=" | ">=" if args.len() == 2 => {
                        let a = self.try_const_eval(&args[0])?;
                        let b = self.try_const_eval(&args[1])?;
                        let result = match name.as_str() {
                            "==" => (a - b).abs() < 1e-10,
                            "!=" => (a - b).abs() >= 1e-10,
                            "<" => a < b,
                            ">" => a > b,
                            "<=" => a <= b,
                            ">=" => a >= b,
                            _ => return None,
                        };
                        Some(if result { 1.0 } else { 0.0 })
                    }
                    "-" if args.len() == 1 => {
                        let a = self.try_const_eval(&args[0])?;
                        Some(-a)
                    }
                    "+" | "-" | "*" | "/" if args.len() == 2 => {
                        let a = self.try_const_eval(&args[0])?;
                        let b = self.try_const_eval(&args[1])?;
                        Some(match name.as_str() {
                            "+" => a + b,
                            "-" => a - b,
                            "*" => a * b,
                            "/" => a / b,
                            _ => return None,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
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
                        // Note: classify_vectors should normally have converted
                        // these to a Tuple node; this is the fallback.
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

            // Tuple after classify_vectors: heterogeneous group of values
            // (e.g. [new-h new-c] from an LSTM cell).
            CompiledExpr::Tuple(elements) => {
                let mut regs = Vec::new();
                let mut tys = Vec::new();
                for elem in elements {
                    let (reg, ty) = self.generate(elem)?;
                    regs.push(reg);
                    tys.push(ty);
                }
                Ok(self.emitter.emit_tuple(&regs, &tys))
            }

            CompiledExpr::Symbol(name) => {
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

                let mut current_reg = param_reg;
                let mut current_ty = param_ty;

                for &idx in indices {
                    let element_ty = match &current_ty {
                        StableHLOType::Tuple(elems, _) => elems.get(idx).cloned().ok_or_else(|| {
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

                self.track_layout_key(param, indices, current_reg);

                Ok((current_reg, current_ty))
            }

            CompiledExpr::FunctionCall { name, args, .. } => self.generate_function_call(name, args),

    CompiledExpr::Let { bindings, body } => {
        let saved_bindings = self.bindings.clone();
        let saved_lambda_bindings = self.lambda_bindings.clone();
        for (pattern, value_expr) in bindings {
            if matches!(value_expr, CompiledExpr::Lambda { .. }) {
                if let BindingPattern::Simple(name) = pattern {
                    self.lambda_bindings.insert(name.clone(), value_expr.clone());
                }
            } else if let BindingPattern::Destructure(_) = pattern {
                return Err(SheafError::Compile {
                    message: "Destructuring patterns should have been desugared".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            } else if let BindingPattern::Simple(name) = pattern {
                let (reg, ty) = self.generate(value_expr)?;
                // Propagate sub-layout for Let-bound tuples
                if matches!(&ty, StableHLOType::Tuple(..)) {
                    if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args, .. } = value_expr {
                        if fn_name == "get"
                            && fn_args.len() >= 2
                            && let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last()
                            && let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned()
                        {
                            self.tuple_key_layouts.insert(name.clone(), sub_layout);
                        }
                    }
                    // GetTupleElement: walk idx_to_key chain to resolve layout
                    // (needed when inlined functions have lowered get calls)
                    else if let CompiledExpr::GetTupleElement { param, indices } = value_expr {
                        let mut cur = param.clone();
                        let mut resolved = true;
                        for &idx in indices {
                            if let Some(key) = self.idx_to_key.get(&(cur.clone(), idx)) {
                                cur = key.clone();
                            } else {
                                resolved = false;
                                break;
                            }
                        }
                        if resolved
                            && let Some(sub_layout) = self.tuple_key_layouts.get(&cur).cloned()
                        {
                            self.tuple_key_layouts.insert(name.clone(), sub_layout);
                        }
                    }
                    // Symbol alias: (let [x y]) where y has a layout
                    else if let CompiledExpr::Symbol(src) = value_expr
                        && let Some(layout) = self.tuple_key_layouts.get(src).cloned()
                    {
                        self.tuple_key_layouts.insert(name.clone(), layout);
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
                // Constant-fold: if condition is compile-time known, only emit the taken branch.
                // This avoids shape mismatches in select when branches have different types
                // (e.g. `(if (== (ndim x) 3) (sum x :axis -2) x)` where branches differ in rank).
                if let Some(const_val) = self.try_const_eval(condition) {
                    let is_true = const_val.abs() > 1e-10;
                    if is_true {
                        return self.generate(then_branch);
                    } else if let Some(else_expr) = else_branch {
                        return self.generate(else_expr);
                    } else {
                        // if-without-else, condition false: return scalar 0
                        let reg = self.emitter.emit_constant_f32(0.0);
                        return Ok((reg, StableHLOType::ScalarF32));
                    }
                }

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
                let mut keys = Vec::new();
                for (k, val) in &sorted {
                    let (r, t) = self.generate(val)?;
                    regs.push(r);
                    tys.push(t);
                    keys.push(match k {
                        CompiledExpr::Keyword(k) => k.clone(),
                        _ => String::new(),
                    });
                }
                let (reg, _) = self.emitter.emit_tuple(&regs, &tys);
                Ok((reg, StableHLOType::Tuple(tys, Some(keys))))
            }

            CompiledExpr::Quoted(val) => {
                // Quoted values: convert to vector of constants
                use crate::core::ast::SheafValue;
                match val.as_ref() {
                    SheafValue::Vector(elems, _) => {
                        let converted: Vec<CompiledExpr> = elems.iter().map(|e| match e {
                            SheafValue::Integer(n, _) => CompiledExpr::Integer(*n),
                            SheafValue::Float(f, _) => CompiledExpr::Float(*f),
                            _ => CompiledExpr::Integer(0),
                        }).collect();
                        self.generate(&CompiledExpr::Vector(converted))
                    }
                    _ => Err(SheafError::Compile {
                        message: format!("Code generation not yet implemented for: {:?}", expr),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
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
    /// Resolves Symbol references via bindings + known_scalars.
    fn parse_shape_vec(&self, elems: &[CompiledExpr]) -> SheafResult<Vec<i64>> {
        elems.iter().map(|e| match e {
            CompiledExpr::Integer(n) => Ok(*n),
            CompiledExpr::Float(f) if *f == f.floor() => Ok(*f as i64),
            CompiledExpr::Symbol(name) => {
                if let Some(&(reg, _)) = self.bindings.get(name.as_str()) {
                    if let Some(v) = self.emitter.known_scalar_value(&reg) {
                        Ok(v as i64)
                    } else {
                        Err(SheafError::Compile {
                            message: format!(
                                "shape element must be a constant integer, got: Symbol({:?}) (not a known constant)",
                                name
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                } else {
                    Err(SheafError::Compile {
                        message: format!(
                            "shape element must be a constant integer, got: Symbol({:?}) (not bound)",
                            name
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }
            other => Err(SheafError::Compile {
                message: format!(
                    "shape element must be a constant integer, got: {:?}",
                    other
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }).collect()
    }

    /// Track layout key for a register via the idx_to_key chain.
    fn track_layout_key(&mut self, param: &str, indices: &[usize], reg: Register) {
        let mut cur = param.to_string();
        for &idx in indices {
            if let Some(key) = self.idx_to_key.get(&(cur.clone(), idx)) {
                cur = key.clone();
            } else {
                cur = String::new();
                break;
            }
        }
        if !cur.is_empty() {
            self.layout_key_map.insert(reg, cur);
        }
    }

    /// Emit a complete function module
    pub fn emit_function(mut self, name: &str, expr: &CompiledExpr) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        Ok(self.emitter.emit_function_body(name, &result_ty))
    }

    /// Emit a function declaration from a compiled expression.
    ///
    /// Parameters are flattened (no tuple types at MLIR boundary).
    /// If the result is a tuple, it is decomposed via virtual leaves into a multi-value return.
    /// Returns (mlir_declaration, actual_return_type) where the return type
    /// preserves the original tuple structure for runtime reconstruction.
    pub fn emit_func_declaration(
        mut self,
        name: &str,
        expr: &CompiledExpr,
        param_types: &[StableHLOType],
        _return_type: &StableHLOType,
    ) -> SheafResult<(String, StableHLOType)> {
        let (result_reg, result_ty) = self.generate(expr)?;
        let flat_params = flatten_param_types(param_types);
        let leaves = self.emitter.collect_virtual_leaves(result_reg, &result_ty);
        let (leaf_regs, leaf_tys): (Vec<_>, Vec<_>) = leaves.into_iter().unzip();

    if leaf_regs.len() == 1 {
        self.emitter.emit_return(&leaf_regs[0], &leaf_tys[0]);
        let body = self.emitter.body.clone();
            let decl = self.emitter
                .emit_func_declaration(name, &flat_params, &leaf_tys[0], &body);
            Ok((decl, result_ty))
    } else {
        self.emitter.emit_return_multi(&leaf_regs, &leaf_tys);
        let body = self.emitter.body.clone();
            let decl = self.emitter
                .emit_func_declaration_multi(name, &flat_params, &leaf_tys, &body);
            Ok((decl, result_ty))
        }
    }

    /// Finalize a multi-output function declaration.
    ///
    /// Emits a `return %r0, %r1, ...` then wraps everything in a `func.func`
    /// with a multi-value return type `-> (t0, t1, ...)`.
    /// Parameters are flattened (no tuple types at MLIR boundary).
    /// Result registers are resolved through virtual tuples to collect all leaves.
    pub fn finish_multi(
        mut self,
        name: &str,
        param_types: &[StableHLOType],
        result_regs: &[crate::lowering::stablehlo::Register],
        result_types: &[StableHLOType],
    ) -> String {
        let flat_params = flatten_param_types(param_types);
        let mut all_regs = Vec::new();
        let mut all_tys = Vec::new();
        for (r, t) in result_regs.iter().zip(result_types.iter()) {
            for (leaf_r, leaf_t) in self.emitter.collect_virtual_leaves(*r, t) {
                all_regs.push(leaf_r);
                all_tys.push(leaf_t);
            }
        }
    self.emitter.emit_return_multi(&all_regs, &all_tys);
    let body = self.emitter.body.clone();
        self.emitter
            .emit_func_declaration_multi(name, &flat_params, &all_tys, &body)
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

