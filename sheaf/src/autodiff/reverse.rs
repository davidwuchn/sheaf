// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Reverse-mode autodiff on Administrative Normal Form (ANF).
//!
//! Two passes:
//! 1. `to_anf`: flatten a CompiledExpr tree into a flat Let chain where every
//!    sub-expression is named.  Symbols, literals and GetTupleElement are "trivial"
//!    and stay inline; everything else gets a `__anf_N` binding.
//!
//! 2. `reverse_grad`: walk the ANF bindings in reverse, emitting backward
//!    bindings that compute adjoint contributions.  Each adjoint is itself a
//!    named binding, so the backward expression is also flat.

use crate::autodiff::replace_symbol;
use crate::core::error::{SheafError, SheafResult};
use crate::core::expr::{BindingPattern, CompiledExpr};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static ANF_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fresh_anf_name() -> String {
    let n = ANF_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__anf_{}", n)
}

fn is_trivial(expr: &CompiledExpr) -> bool {
    matches!(
        expr,
        CompiledExpr::Symbol(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Integer(_)
            | CompiledExpr::Boolean(_)
            | CompiledExpr::Nil
            | CompiledExpr::String(_)
            | CompiledExpr::Keyword(_)
            | CompiledExpr::Quoted(_)
            | CompiledExpr::GetTupleElement { .. }
    )
}

/// Flatten a `CompiledExpr` tree into ANF: a single `Let` with a flat list of
/// bindings where every RHS contains only trivial sub-expressions (Symbols,
/// literals, GetTupleElement).
///
/// The result body is always a trivial expression (a Symbol referencing the
/// last binding, or the original expression if it was already trivial).
pub fn to_anf(expr: &CompiledExpr) -> CompiledExpr {
    let mut bindings = Vec::new();
    let body = anf_rec(expr, &mut bindings);
    if bindings.is_empty() {
        body
    } else {
        CompiledExpr::Let {
            bindings,
            body: Box::new(body),
        }
    }
}

/// Recursively convert `expr` to ANF, appending bindings to `out`.
/// Returns a trivial expression (Symbol or literal) that represents the value.
fn anf_rec(expr: &CompiledExpr, out: &mut Vec<(BindingPattern, CompiledExpr)>) -> CompiledExpr {
    match expr {
        // Trivial: return as-is
        _ if is_trivial(expr) => expr.clone(),

        // FunctionCall: ANF-ify all args, then bind the call
        CompiledExpr::FunctionCall { name, args, loc } => {
            let anf_args: Vec<CompiledExpr> = args.iter().map(|a| anf_rec(a, out)).collect();
            let result = CompiledExpr::FunctionCall {
                name: name.clone(),
                args: anf_args,
                loc: loc.clone(),
            };
            let sym = fresh_anf_name();
            out.push((BindingPattern::Simple(sym.clone()), result));
            CompiledExpr::Symbol(sym)
        }

        // Let: flatten bindings with alpha-renaming to avoid shadowing.
        // Each binding name gets a fresh __anf_N name. Using a HashMap
        // ensures that when a name is rebound (e.g. `as->` threading),
        // only the LATEST rename is applied to subsequent values and body.
        CompiledExpr::Let { bindings, body } => {
            let mut rename_map: HashMap<String, String> = HashMap::new();
            for (name, value) in bindings {
                // Apply accumulated renames to this binding's value
                let mut val = value.clone();
                for (old, new_name) in &rename_map {
                    val = replace_symbol(&val, old, &CompiledExpr::Symbol(new_name.clone()));
                }
                
                let anf_val = anf_rec(&val, out);
                let fresh = fresh_anf_name();
                out.push((BindingPattern::Simple(fresh.clone()), anf_val));
                // HashMap::insert overwrites previous entry for same name,
                // so body/subsequent values always see the LATEST binding.
                // Per the destructuring elim contract, the binding should be
                // desugared before reaching here, so a Destructure would be a
                // bug. debug_assert catches it loudly during development.
                debug_assert!(
                    matches!(name, BindingPattern::Simple(_)),
                    "ANF: expected Simple binding pattern (destructure should be desugared), got {:?}",
                    name
                );
                let simple_name = match name {
                    BindingPattern::Simple(s) => s.clone(),
                    BindingPattern::Destructure(_) => String::new(),
                };
                rename_map.insert(simple_name, fresh);
            }
            // Apply renames to body
            let mut renamed_body = body.as_ref().clone();
            for (old, new_name) in &rename_map {
                renamed_body = replace_symbol(&renamed_body, old, &CompiledExpr::Symbol(new_name.clone()));
            }
            anf_rec(&renamed_body, out)
        }

        // Do: process all expressions, return the last
        CompiledExpr::Do(exprs) => {
            let mut last = CompiledExpr::Nil;
            for e in exprs {
                last = anf_rec(e, out);
            }
            last
        }

        // Vector: ANF-ify elements but keep the Vector inline (not bound).
        // Vectors are structural (shape specs, etc.), not computational.
        // Binding them to symbols breaks pattern matching in codegen (e.g. reshape).
        CompiledExpr::Vector(elems) => {
            let anf_elems: Vec<CompiledExpr> = elems.iter().map(|e| anf_rec(e, out)).collect();
            CompiledExpr::Vector(anf_elems)
        }

        // Anything else: bind directly
        other => {
            let sym = fresh_anf_name();
            out.push((BindingPattern::Simple(sym.clone()), other.clone()));
            CompiledExpr::Symbol(sym)
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode AD on ANF bindings
// ---------------------------------------------------------------------------

static GRAD_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fresh_grad_name() -> String {
    let n = GRAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__grad_{}", n)
}

/// Compute reverse-mode gradients of an ANF expression w.r.t. a set of
/// parameter names.
///
/// Returns `(backward_bindings, grad_map)` where:
/// - `backward_bindings`: flat Let bindings computing all adjoint intermediates
/// - `grad_map`: maps each wrt name -> the Symbol name of its gradient
///
/// The backward bindings reference the forward ANF bindings by name.
/// The caller must emit both forward and backward bindings in a single scope.
pub fn reverse_grad(
    anf_bindings: &[(String, CompiledExpr)],
    anf_body: &CompiledExpr,
    wrt: &[String],
    shapes: &HashMap<String, Vec<i64>>,
) -> SheafResult<(Vec<(String, CompiledExpr)>, HashMap<String, String>)> {
    let dependencies = analyze_anf_dependencies(anf_bindings, wrt);
    let mut adj_names: HashMap<String, String> = HashMap::new();
    let mut backward_bindings: Vec<(String, CompiledExpr)> = Vec::new();

    let fwd_lookup: HashMap<String, String> = anf_bindings.iter()
        .filter_map(|(name, expr)| {
            match expr {
                CompiledExpr::FunctionCall { name: fn_name, .. }
                    if matches!(fn_name.as_str(), "mean" | "softmax" | "tanh" | "sigmoid" | "exp" | "log" | "sqrt" | "sin" | "cos" | "tan") =>
                {
                    Some((format!("{:?}", expr), name.clone()))
                }
                _ => None,
            }
        })
        .collect();

    if let CompiledExpr::Symbol(s) = anf_body {
        let seed_name = fresh_grad_name();
        backward_bindings.push((seed_name.clone(), CompiledExpr::Float(1.0)));
        adj_names.insert(s.clone(), seed_name);
    }

    // Walk bindings in reverse
    for (name, value) in anf_bindings.iter().rev() {
        let adj_sym = match adj_names.get(name) {
            Some(s) => CompiledExpr::Symbol(s.clone()),
            None => continue,
        };

        distribute_adjoint_named(
            value,
            &adj_sym,
            &mut adj_names,
            &mut backward_bindings,
            shapes,
            name,
            &fwd_lookup,
            &dependencies,
        )?;
    }

    // Build grad_map: wrt name -> adjoint symbol name
    let grad_map: HashMap<String, String> = wrt
        .iter()
        .filter_map(|param| {
            adj_names.get(param).map(|adj_name| (param.clone(), adj_name.clone()))
        })
        .collect();

    Ok((backward_bindings, grad_map))
}

/// Determine which ANF bindings transitively depend on differentiated inputs.
///
/// Unknown ANF forms are conservatively dependent, so they cannot turn a
/// missing reverse rule into a plausible zero gradient.
fn analyze_anf_dependencies(
    anf_bindings: &[(String, CompiledExpr)],
    wrt: &[String],
) -> HashMap<String, bool> {
    let mut dependencies: HashMap<String, bool> = wrt
        .iter()
        .map(|name| (name.clone(), true))
        .collect();

    for (name, value) in anf_bindings {
        let depends = anf_expr_depends_on_wrt(value, &dependencies);
        dependencies.insert(name.clone(), depends);
    }

    dependencies
}

fn anf_expr_depends_on_wrt(expr: &CompiledExpr, dependencies: &HashMap<String, bool>) -> bool {
    match expr {
        CompiledExpr::Integer(_)
        | CompiledExpr::Float(_)
        | CompiledExpr::Boolean(_)
        | CompiledExpr::Nil
        | CompiledExpr::String(_)
        | CompiledExpr::Keyword(_)
        | CompiledExpr::Quoted(_)
        | CompiledExpr::FunctionRef(_)
        | CompiledExpr::ValueAndGrad { .. } => false,
        CompiledExpr::Symbol(name) => dependencies.get(name).copied().unwrap_or(false),
        CompiledExpr::GetTupleElement { param, .. } => {
            dependencies.get(param).copied().unwrap_or(false)
        }
        CompiledExpr::Vector(elems) | CompiledExpr::Tuple(elems) => elems
            .iter()
            .any(|elem| anf_expr_depends_on_wrt(elem, dependencies)),
        CompiledExpr::Dict(pairs) => pairs.iter().any(|(key, value)| {
            anf_expr_depends_on_wrt(key, dependencies)
                || anf_expr_depends_on_wrt(value, dependencies)
        }),
        CompiledExpr::FunctionCall { name, .. } if name == "shape" => false,
        CompiledExpr::FunctionCall { args, .. } => args
            .iter()
            .any(|arg| anf_expr_depends_on_wrt(arg, dependencies)),
        CompiledExpr::Def { value, .. } => anf_expr_depends_on_wrt(value, dependencies),
        CompiledExpr::Let { .. }
        | CompiledExpr::If { .. }
        | CompiledExpr::Do(_)
        | CompiledExpr::Lambda { .. }
        | CompiledExpr::LambdaCall { .. }
        | CompiledExpr::Repeat { .. }
        | CompiledExpr::While { .. }
        | CompiledExpr::Guard { .. } => true,
    }
}

/// Add a contribution to the adjoint of `var_name`, emitting a new binding.
///
/// If `var_name` already has an adjoint, emit `new_adj = old_adj + contribution`.
/// Otherwise, the contribution IS the adjoint.
fn accumulate_named(
    var_name: &str,
    contribution: CompiledExpr,
    adj_names: &mut HashMap<String, String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
) {
    if is_zero(&contribution) {
        return;
    }

    match adj_names.get(var_name) {
        None => {
            // First contribution: just name it
            let grad_name = fresh_grad_name();
            bindings.push((grad_name.clone(), contribution));
            adj_names.insert(var_name.to_string(), grad_name);
        }
        Some(existing) => {
            // Add to existing: emit new_name = existing + contribution
            let contrib_name = fresh_grad_name();
            bindings.push((contrib_name.clone(), contribution));
            let sum_name = fresh_grad_name();
            bindings.push((
                sum_name.clone(),
                CompiledExpr::FunctionCall {
                    name: "+".to_string(),
                    args: vec![
                        CompiledExpr::Symbol(existing.clone()),
                        CompiledExpr::Symbol(contrib_name),
                    ],
                    loc: None,
                },
            ));
            adj_names.insert(var_name.to_string(), sum_name);
        }
    }
}

fn is_zero(expr: &CompiledExpr) -> bool {
    matches!(expr, CompiledExpr::Float(v) if *v == 0.0)
        || matches!(expr, CompiledExpr::Integer(0))
}

fn sym(name: &str) -> CompiledExpr {
    CompiledExpr::Symbol(name.to_string())
}

fn float(v: f64) -> CompiledExpr {
    CompiledExpr::Float(v)
}

fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
    CompiledExpr::FunctionCall {
        name: name.to_string(),
        args,
        loc: None,
    }
}

fn shape_vec(shape: &[i64]) -> CompiledExpr {
    CompiledExpr::Vector(
        shape
            .iter()
            .map(|&d| CompiledExpr::Integer(d))
            .collect(),
    )
}

/// Wrap `adj` with a `sum_to_shape` call if the adjoint shape differs from
/// the target operand shape (i.e. when the forward op involved broadcasting).
fn maybe_unbroadcast(
    adj: CompiledExpr,
    target_arg: &CompiledExpr,
    shapes: &HashMap<String, Vec<i64>>,
    bindings: &mut Vec<(String, CompiledExpr)>,
) -> CompiledExpr {
    if let Some(target_shape) = arg_shape(target_arg, shapes) {
        // We don't know the adj shape directly, but if we can look up the
        // result shape (shape of the binding that produced the forward
        // value), we can compare.  For now, always emit sum_to_shape and
        // let the codegen handle the identity case (same shape -> no-op).
        let reduced = emit_binding(
            bindings,
            call("sum_to_shape", vec![adj, shape_vec(&target_shape)]),
        );
        sym(&reduced)
    } else {
        adj
    }
}

fn reject_missing_rule_if_dependent(
    operation: &str,
    fwd_name: &str,
    dependencies: &HashMap<String, bool>,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<()> {
    if dependencies.get(fwd_name).copied().unwrap_or(true) {
        Err(SheafError::AutodiffMissingRule {
            operation: operation.to_string(),
            location,
        })
    } else {
        Ok(())
    }
}

fn anf_form_name(expr: &CompiledExpr) -> &'static str {
    match expr {
        CompiledExpr::Integer(_) => "integer",
        CompiledExpr::Float(_) => "float",
        CompiledExpr::Boolean(_) => "boolean",
        CompiledExpr::Nil => "nil",
        CompiledExpr::String(_) => "string",
        CompiledExpr::Keyword(_) => "keyword",
        CompiledExpr::Vector(_) => "vector",
        CompiledExpr::Tuple(_) => "tuple",
        CompiledExpr::Dict(_) => "dict",
        CompiledExpr::Quoted(_) => "quoted",
        CompiledExpr::FunctionRef(_) => "function reference",
        CompiledExpr::Let { .. } => "let",
        CompiledExpr::If { .. } => "if",
        CompiledExpr::Do(_) => "do",
        CompiledExpr::Lambda { .. } => "lambda",
        CompiledExpr::LambdaCall { .. } => "lambda call",
        CompiledExpr::ValueAndGrad { .. } => "value-and-grad",
        CompiledExpr::Repeat { .. } => "repeat",
        CompiledExpr::While { .. } => "while",
        CompiledExpr::Guard { .. } => "guard",
        CompiledExpr::Def { .. } => "def",
        CompiledExpr::Symbol(_) => "symbol",
        CompiledExpr::GetTupleElement { .. } => "tuple element",
        CompiledExpr::FunctionCall { .. } => "function call",
    }
}

/// Distribute the adjoint `adj_sym` (a Symbol) to the operands of `expr`.
///
/// All emitted expressions reference `adj_sym` by Symbol, never clone the
/// underlying expression: this guarantees O(1) per distribution step.
fn distribute_adjoint_named(
    expr: &CompiledExpr,
    adj_sym: &CompiledExpr,
    adj_names: &mut HashMap<String, String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
    shapes: &HashMap<String, Vec<i64>>,
    fwd_name: &str,
    fwd_lookup: &HashMap<String, String>,
    dependencies: &HashMap<String, bool>,
) -> SheafResult<()> {
    match expr {
        CompiledExpr::Symbol(s) => {
            accumulate_named(s, adj_sym.clone(), adj_names, bindings);
            Ok(())
        }
        CompiledExpr::GetTupleElement { param, .. } => {
            accumulate_named(param, adj_sym.clone(), adj_names, bindings);
            Ok(())
        }
        CompiledExpr::FunctionCall { name, args, loc } => distribute_fn_adjoint_named(
            name,
            args,
            adj_sym,
            adj_names,
            bindings,
            shapes,
            fwd_name,
            fwd_lookup,
            dependencies,
            loc.clone(),
        ),
        _ => reject_missing_rule_if_dependent(
            anf_form_name(expr),
            fwd_name,
            dependencies,
            None,
        ),
    }
}

fn distribute_fn_adjoint_named(
    name: &str,
    args: &[CompiledExpr],
    adj: &CompiledExpr,
    adj_names: &mut HashMap<String, String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
    shapes: &HashMap<String, Vec<i64>>,
    fwd_name: &str,
    fwd_lookup: &HashMap<String, String>,
    dependencies: &HashMap<String, bool>,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<()> {
    match name {
        // stop-gradient: forward is identity, but no gradient flows backward.
        "stop-gradient" => {}

        "+" => {
            // da += unbroadcast(adj, shape_a), db += unbroadcast(adj, shape_b)
            let adj_a = maybe_unbroadcast(adj.clone(), &args[0], shapes, bindings);
            acc_arg(&args[0], adj_a, adj_names, bindings);
            let adj_b = maybe_unbroadcast(adj.clone(), &args[1], shapes, bindings);
            acc_arg(&args[1], adj_b, adj_names, bindings);
        }

        "-" if args.len() == 2 => {
            let adj_a = maybe_unbroadcast(adj.clone(), &args[0], shapes, bindings);
            acc_arg(&args[0], adj_a, adj_names, bindings);
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            let adj_b = maybe_unbroadcast(sym(&neg), &args[1], shapes, bindings);
            acc_arg(&args[1], adj_b, adj_names, bindings);
        }

        "-" if args.len() == 1 => {
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            acc_arg(&args[0], sym(&neg), adj_names, bindings);
        }

        "*" => {
            let da = emit_binding(bindings, call("*", vec![adj.clone(), args[1].clone()]));
            let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
            acc_arg(&args[0], da_ub, adj_names, bindings);
            let db = emit_binding(bindings, call("*", vec![adj.clone(), args[0].clone()]));
            let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
            acc_arg(&args[1], db_ub, adj_names, bindings);
        }

        "/" if args.len() == 2 => {
            let da = emit_binding(bindings, call("/", vec![adj.clone(), args[1].clone()]));
            let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
            acc_arg(&args[0], da_ub, adj_names, bindings);
            let b_sq = emit_binding(bindings, call("*", vec![args[1].clone(), args[1].clone()]));
            let a_over_b2 = emit_binding(bindings, call("/", vec![args[0].clone(), sym(&b_sq)]));
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), sym(&a_over_b2)]));
            let db = emit_binding(bindings, call("*", vec![adj.clone(), sym(&neg)]));
            let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
            acc_arg(&args[1], db_ub, adj_names, bindings);
        }

        "@" => {
            let a_ndim = arg_shape(&args[0], shapes).map(|s| s.len()).unwrap_or(2);
            let b_ndim = arg_shape(&args[1], shapes).map(|s| s.len()).unwrap_or(2);

            // dL/dA = G @ B^T
            if b_ndim == 1 {
                // B is 1D [n], adj is scalar or matching: dL/dA = adj * B (broadcast)
                let da = emit_binding(bindings, call("*", vec![adj.clone(), args[1].clone()]));
                let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                acc_arg(&args[0], da_ub, adj_names, bindings);
            } else if a_ndim == 1 {
                // A is 1D [m], B is 2D [m, n], result is 1D [n], adj is 1D [n]
                // dL/dA = adj @ B^T but adj is 1D: reshape to [1, n], matmul, reshape back to [m]
                if let Some(b_shape) = arg_shape(&args[1], shapes) {
                    let n = b_shape[b_shape.len() - 1];
                    let adj_row = emit_binding(bindings, call("reshape", vec![
                        adj.clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(n)]),
                    ]));
                    let bt = emit_binding(bindings, call("transpose", vec![args[1].clone()]));
                    let da_2d = emit_binding(bindings, call("@", vec![sym(&adj_row), sym(&bt)]));
                    let m = b_shape[0];
                    let da = emit_binding(bindings, call("reshape", vec![
                        sym(&da_2d),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(m)]),
                    ]));
                    acc_arg(&args[0], sym(&da), adj_names, bindings);
                } else {
                    let bt = emit_binding(bindings, call("transpose", vec![args[1].clone()]));
                    let da = emit_binding(bindings, call("@", vec![adj.clone(), sym(&bt)]));
                    let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                    acc_arg(&args[0], da_ub, adj_names, bindings);
                }
            } else {
                let bt = emit_binding(bindings, call("transpose", vec![args[1].clone()]));
                let da = emit_binding(bindings, call("@", vec![adj.clone(), sym(&bt)]));
                let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                acc_arg(&args[0], da_ub, adj_names, bindings);
            }

            // dL/dB = A^T @ G
            if a_ndim == 1 {
                // A is 1D [m], G may be 1D [n] -> outer product -> [m, n]
                // reshape(A,[m,1]) @ reshape(G,[1,n])
                if let (Some(a_shape), Some(b_shape)) = (arg_shape(&args[0], shapes), arg_shape(&args[1], shapes)) {
                    let m = a_shape[0];
                    let n = b_shape[b_shape.len() - 1];
                    let a_col = emit_binding(bindings, call("reshape", vec![
                        args[0].clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(m), CompiledExpr::Integer(1)]),
                    ]));
                    let g_row = emit_binding(bindings, call("reshape", vec![
                        adj.clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(n)]),
                    ]));
                    let db = emit_binding(bindings, call("@", vec![sym(&a_col), sym(&g_row)]));
                    let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                    acc_arg(&args[1], db_ub, adj_names, bindings);
                } else {
                    // Fallback: emit A^T @ G (may fail for 1D, but no shape info)
                    let at = emit_binding(bindings, call("transpose", vec![args[0].clone()]));
                    let db = emit_binding(bindings, call("@", vec![sym(&at), adj.clone()]));
                    let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                    acc_arg(&args[1], db_ub, adj_names, bindings);
                }
            } else {
                let at = emit_binding(bindings, call("transpose", vec![args[0].clone()]));
                let db = emit_binding(bindings, call("@", vec![sym(&at), adj.clone()]));
                let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                acc_arg(&args[1], db_ub, adj_names, bindings);
            }
        }

        "einsum" => {
            // einsum("lhs_sub,rhs_sub->out_sub", A, B)
            // dL/dA = einsum("out_sub,rhs_sub->lhs_sub", adj, B)
            // dL/dB = einsum("lhs_sub,out_sub->rhs_sub", A, adj)
            let Some(CompiledExpr::String(sub)) = args.first() else {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            };
            if args.len() != 3 {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            }
            let Some((inputs, output)) = sub.split_once("->") else {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            };
            let parts: Vec<&str> = inputs.split(',').collect();
            if parts.len() != 2 {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            }
            let lhs_sub = parts[0].trim();
            let rhs_sub = parts[1].trim();
            let out_sub = output.trim();

            let grad_lhs_sub = format!("{},{}->{}", out_sub, rhs_sub, lhs_sub);
            let da = emit_binding(bindings, call("einsum", vec![
                CompiledExpr::String(grad_lhs_sub),
                adj.clone(),
                args[2].clone(),
            ]));
            acc_arg(&args[1], sym(&da), adj_names, bindings);

            let grad_rhs_sub = format!("{},{}->{}", lhs_sub, out_sub, rhs_sub);
            let db = emit_binding(bindings, call("einsum", vec![
                CompiledExpr::String(grad_rhs_sub),
                args[1].clone(),
                adj.clone(),
            ]));
            acc_arg(&args[2], sym(&db), adj_names, bindings);
        }

        "transpose" | "tr" => {
            let dt = emit_binding(bindings, call("transpose", vec![adj.clone()]));
            acc_arg(&args[0], sym(&dt), adj_names, bindings);
        }

        "reshape" => {
            // d_input = reshape(adj, input_shape)
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let shape_vec = CompiledExpr::Vector(
                    input_shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                );
                let dr = emit_binding(bindings, call("reshape", vec![adj.clone(), shape_vec]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                // Fallback: pass through (may produce shape errors)
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "swapaxes" if args.len() == 3 => {
            // swapaxes is self-inverse: d_input = swapaxes(adj, axis1, axis2)
            let dr = emit_binding(bindings, call("swapaxes", vec![adj.clone(), args[1].clone(), args[2].clone()]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "relu" => {
            // d_input = adj * (x > 0)
            let cond = emit_binding(bindings, call(">", vec![args[0].clone(), float(0.0)]));
            let dr = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "gelu" => {
            // tanh-approximation GELU: gelu(x) = 0.5 * x * (1 + tanh(k))
            // where k = sqrt(2/pi) * (x + 0.044715 * x^3)
            // gelu'(x) = 0.5 * (1 + tanh(k)) + 0.5 * x * sech²(k) * k'
            // k' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
            // sech²(k) = 1 - tanh²(k)
            let x = &args[0];
            let x2 = emit_binding(bindings, call("*", vec![x.clone(), x.clone()]));
            let x3 = emit_binding(bindings, call("*", vec![sym(&x2), x.clone()]));
            // k = 0.7978846 * (x + 0.044715 * x^3)
            let coeff_x3 = emit_binding(bindings, call("*", vec![float(0.044715), sym(&x3)]));
            let inner = emit_binding(bindings, call("+", vec![x.clone(), sym(&coeff_x3)]));
            let k = emit_binding(bindings, call("*", vec![float(0.7978845608), sym(&inner)]));
            let tanh_k = emit_binding(bindings, call("tanh", vec![sym(&k)]));
            // 0.5 * (1 + tanh(k))
            let one_plus_tanh = emit_binding(bindings, call("+", vec![float(1.0), sym(&tanh_k)]));
            let half_term1 = emit_binding(bindings, call("*", vec![float(0.5), sym(&one_plus_tanh)]));
            // sech^2(k) = 1 - tanh^2(k)
            let tanh_sq = emit_binding(bindings, call("*", vec![sym(&tanh_k), sym(&tanh_k)]));
            let sech2 = emit_binding(bindings, call("-", vec![float(1.0), sym(&tanh_sq)]));
            // k' = 0.7978846 * (1 + 3 * 0.044715 * x^2) = 0.7978846 * (1 + 0.134145 * x^2)
            let coeff_x2 = emit_binding(bindings, call("*", vec![float(0.134145), sym(&x2)]));
            let kp_inner = emit_binding(bindings, call("+", vec![float(1.0), sym(&coeff_x2)]));
            let kp = emit_binding(bindings, call("*", vec![float(0.7978845608), sym(&kp_inner)]));
            // term2 = 0.5 * x * sech^2(k) * k'
            let x_sech2 = emit_binding(bindings, call("*", vec![x.clone(), sym(&sech2)]));
            let x_sech2_kp = emit_binding(bindings, call("*", vec![sym(&x_sech2), sym(&kp)]));
            let half_term2 = emit_binding(bindings, call("*", vec![float(0.5), sym(&x_sech2_kp)]));
            // gelu'(x) = term1 + term2
            let gelu_grad = emit_binding(bindings, call("+", vec![sym(&half_term1), sym(&half_term2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&gelu_grad)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "softmax" => {
            let axis = parse_keyword_int(args, "axis").unwrap_or(-1);
            let sm = fwd_name.to_string();
            let prod = emit_binding(bindings, call("*", vec![adj.clone(), sym(&sm)]));
            let s = emit_binding(bindings, call("sum", vec![
                sym(&prod),
                CompiledExpr::Keyword("axis".to_string()),
                CompiledExpr::Integer(axis),
                CompiledExpr::Keyword("keepdims".to_string()),
                CompiledExpr::Boolean(true),
            ]));
            let diff = emit_binding(bindings, call("-", vec![adj.clone(), sym(&s)]));
            let dr = emit_binding(bindings, call("*", vec![sym(&sm), sym(&diff)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "log-softmax" => {
            // d_input = adj - exp(log_softmax(x)) * sum(adj, axis=-1, keepdims=true)
            let axis = parse_keyword_int(args, "axis").unwrap_or(-1);
            let fwd_args: Vec<CompiledExpr> = args.to_vec();
            let lsm = emit_binding(bindings, call("log-softmax", fwd_args));
            let sm = emit_binding(bindings, call("exp", vec![sym(&lsm)]));
            let s = emit_binding(bindings, call("sum", vec![
                adj.clone(),
                CompiledExpr::Keyword("axis".to_string()),
                CompiledExpr::Integer(axis),
                CompiledExpr::Keyword("keepdims".to_string()),
                CompiledExpr::Boolean(true),
            ]));
            let prod = emit_binding(bindings, call("*", vec![sym(&sm), sym(&s)]));
            let dr = emit_binding(bindings, call("-", vec![adj.clone(), sym(&prod)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "sum" => {
            // d_input = broadcast adj back to input shape
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let sv = shape_vec(&input_shape);
                let dr = emit_binding(bindings, call("broadcast", vec![adj.clone(), sv]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "mean" => {
            // d_input = broadcast(adj / n, input_shape)
            // n = product of reduced axes (or all if no axis specified)
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let axis = parse_keyword_int(args, "axis");
                let n: i64 = if let Some(ax) = axis {
                    let ndim = input_shape.len();
                    let ax_usize = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };
                    input_shape[ax_usize]
                } else {
                    input_shape.iter().product()
                };
                let scale = emit_binding(bindings, call("/", vec![adj.clone(), float(n as f64)]));
                let sv = shape_vec(&input_shape);
                let dr = emit_binding(bindings, call("broadcast", vec![sym(&scale), sv]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "var" => {
            // d_input = adj * 2 * (x - mean(x, axis)) / n
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let axis = parse_keyword_int(args, "axis");
                let mut mean_args = vec![args[0].clone()];
                if let Some(ax) = axis {
                    mean_args.push(CompiledExpr::Keyword("axis".to_string()));
                    mean_args.push(CompiledExpr::Integer(ax));
                }
                mean_args.push(CompiledExpr::Keyword("keepdims".to_string()));
        let lookup_key = format!("{:?}", call("mean", mean_args.clone()));
        let m = match fwd_lookup.get(&lookup_key) {
                    Some(sym_name) => sym_name.clone(),
                    None => {
                        mean_args.push(CompiledExpr::Boolean(true));
                        emit_binding(bindings, call("mean", mean_args))
                    }
                };
                let diff = emit_binding(bindings, call("-", vec![args[0].clone(), sym(&m)]));
                let ndim = input_shape.len();
                let n: i64 = if let Some(ax) = axis {
                    let ax_usize = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };
                    input_shape[ax_usize]
                } else {
                    input_shape.iter().product()
                };
                let scaled = emit_binding(bindings, call("*", vec![float(2.0 / n as f64), sym(&diff)]));
                let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&scaled)]));
                let ub = maybe_unbroadcast(sym(&contrib), &args[0], shapes, bindings);
                acc_arg(&args[0], ub, adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "max" | "min" => {
            // d_input = adj * (x == max/min(x, axis, keepdims=true))
            // Gradient is 1 where x equals the max/min, 0 elsewhere.
            let axis = parse_keyword_int(args, "axis");
            let mut fwd_args = vec![args[0].clone()];
            if let Some(ax) = axis {
                fwd_args.push(CompiledExpr::Keyword("axis".to_string()));
                fwd_args.push(CompiledExpr::Integer(ax));
                fwd_args.push(CompiledExpr::Keyword("keepdims".to_string()));
                fwd_args.push(CompiledExpr::Boolean(true));
            }
            let fwd_val = emit_binding(bindings, call(name, fwd_args));
            let mask = emit_binding(bindings, call("==", vec![args[0].clone(), sym(&fwd_val)]));
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let sv = shape_vec(&input_shape);
                let adj_bc = emit_binding(bindings, call("broadcast", vec![adj.clone(), sv]));
                let dr = emit_binding(bindings, call("*", vec![sym(&adj_bc), sym(&mask)]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                let dr = emit_binding(bindings, call("*", vec![adj.clone(), sym(&mask)]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            }
        }

        "sigmoid" => {
            let fwd_key = format!("{:?}", call("sigmoid", vec![args[0].clone()]));
            let sig = match fwd_lookup.get(&fwd_key) {
                Some(fwd_sym) => fwd_sym.clone(),
                None => emit_binding(bindings, call("sigmoid", vec![args[0].clone()])),
            };
            let one_minus_sig = emit_binding(bindings, call("-", vec![float(1.0), sym(&sig)]));
            let local = emit_binding(bindings, call("*", vec![sym(&sig), sym(&one_minus_sig)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "exp" => {
            let fwd_key = format!("{:?}", call("exp", vec![args[0].clone()]));
            let ex = match fwd_lookup.get(&fwd_key) {
                Some(fwd_sym) => fwd_sym.clone(),
                None => emit_binding(bindings, call("exp", vec![args[0].clone()])),
            };
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&ex)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "log" => {
            let inv = emit_binding(bindings, call("/", vec![float(1.0), args[0].clone()]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&inv)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "sqrt" => {
            let sq = emit_binding(bindings, call("sqrt", vec![args[0].clone()]));
            let two_sq = emit_binding(bindings, call("*", vec![float(2.0), sym(&sq)]));
            let inv = emit_binding(bindings, call("/", vec![float(1.0), sym(&two_sq)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&inv)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "tanh" => {
            let fwd_key = format!("{:?}", call("tanh", vec![args[0].clone()]));
            let t = match fwd_lookup.get(&fwd_key) {
                Some(fwd_sym) => fwd_sym.clone(),
                None => emit_binding(bindings, call("tanh", vec![args[0].clone()])),
            };
            let t2 = emit_binding(bindings, call("*", vec![sym(&t), sym(&t)]));
            let local = emit_binding(bindings, call("-", vec![float(1.0), sym(&t2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "sin" => {
            // d/dx sin(x) = cos(x)
            let c = emit_binding(bindings, call("cos", vec![args[0].clone()]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&c)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "cos" => {
            // d/dx cos(x) = -sin(x)
            let s = emit_binding(bindings, call("sin", vec![args[0].clone()]));
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), sym(&s)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&neg)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "tan" => {
            // d/dx tan(x) = 1 + tan(x)^2
            let fwd_key = format!("{:?}", call("tan", vec![args[0].clone()]));
            let t = match fwd_lookup.get(&fwd_key) {
                Some(fwd_sym) => fwd_sym.clone(),
                None => emit_binding(bindings, call("tan", vec![args[0].clone()])),
            };
            let t2 = emit_binding(bindings, call("*", vec![sym(&t), sym(&t)]));
            let local = emit_binding(bindings, call("+", vec![float(1.0), sym(&t2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "maximum" if args.len() == 2 => {
            // d/da maximum(a, b) = adj * (a >= b)
            // d/db maximum(a, b) = adj * (a < b)
            let cond = emit_binding(bindings, call(">=", vec![args[0].clone(), args[1].clone()]));
            let da = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![sym(&cond), float(0.0), adj.clone()]));
            acc_arg(&args[0], sym(&da), adj_names, bindings);
            acc_arg(&args[1], sym(&db), adj_names, bindings);
        }

        "minimum" if args.len() == 2 => {
            // d/da minimum(a, b) = adj * (a <= b)
            // d/db minimum(a, b) = adj * (a > b)
            let cond = emit_binding(bindings, call("<=", vec![args[0].clone(), args[1].clone()]));
            let da = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![sym(&cond), float(0.0), adj.clone()]));
            acc_arg(&args[0], sym(&da), adj_names, bindings);
            acc_arg(&args[1], sym(&db), adj_names, bindings);
        }

        "where" if args.len() == 3 => {
            let da = emit_binding(bindings, call("where", vec![args[0].clone(), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![args[0].clone(), float(0.0), adj.clone()]));
            acc_arg(&args[1], sym(&da), adj_names, bindings);
            acc_arg(&args[2], sym(&db), adj_names, bindings);
        }

        "slice" if args.len() >= 3 => {
            // d_input = pad(adj, input_shape, start_offset)
            // The slice extracts [start..end] along the last axis.
            // Gradient pads adj with zeros to restore the original shape.
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let target_vec = shape_vec(&input_shape);
                // args[1] is the start offset (Integer)
                let dr = emit_binding(
                    bindings,
                    call("slice_grad", vec![adj.clone(), target_vec, args[1].clone()]),
                );
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "neg" => {
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            acc_arg(&args[0], sym(&neg), adj_names, bindings);
        }

        "abs" => {
            let cond = emit_binding(bindings, call(">=", vec![args[0].clone(), float(0.0)]));
            let sign = emit_binding(bindings, call("where", vec![sym(&cond), float(1.0), float(-1.0)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&sign)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "**" if args.len() == 2 => {
            let n = match &args[1] {
                CompiledExpr::Float(n) => *n,
                CompiledExpr::Integer(n) => *n as f64,
                _ => return reject_missing_rule_if_dependent("**", fwd_name, dependencies, location),
            };
            let pow = emit_binding(bindings, call("**", vec![args[0].clone(), float(n - 1.0)]));
            let local = emit_binding(bindings, call("*", vec![float(n), sym(&pow)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        // first(x) extracts element 0.  Adjoint passes through to arg.
        "first" if args.len() == 1 => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }

        "second" if args.len() == 1 => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }

        // scan(lambda, init, coll): emit a __scan_vjp__ call that the codegen compiles.
        // The adjoint flows to init and coll via the backward scan.
        // get(table, indices): embedding lookup / gather on axis 0.
        // Backward: scatter-add the adjoint into a zeros table.
        // For tensor indices: transpose(one-hot(indices, V)) @ adj
        // For scalar index: reshape(one-hot(idx, V), [V,1]) @ reshape(adj, [1,D])
        //
        // Dict-access get (e.g. get(dict, :key)) is not tensor indexing:
        // just pass the adjoint through to the dict argument.
        "get" if args.len() == 2 && matches!(&args[1], CompiledExpr::Keyword(_)) => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }
        "get" if args.len() == 2 => {
            let is_scalar_index = matches!(&args[1],
                CompiledExpr::Integer(_) | CompiledExpr::Float(_)
            ) || matches!(&args[1], CompiledExpr::Symbol(s) if {
                // Check if the symbol's shape in the shape map is scalar
                shapes.get(s.as_str()).map_or(false, |sh| sh.is_empty() || sh == &[1])
            });

            // V = shape(table)[0]
            let v = emit_binding(bindings, call("shape", vec![args[0].clone(), CompiledExpr::Integer(0)]));
            // oh = one-hot(idx, V)  -> [V] for scalar, [N, V] for tensor
            let oh = emit_binding(bindings, call("one-hot", vec![args[1].clone(), sym(&v)]));

            if is_scalar_index {
                // Scalar index: oh is [V], adj is [D]
                // grad = reshape(oh, [V, 1]) @ reshape(adj, [1, D])
                let oh_col = emit_binding(bindings, call("reshape", vec![
                    sym(&oh),
                    CompiledExpr::Vector(vec![sym(&v), CompiledExpr::Integer(1)]),
                ]));
                let adj_row = emit_binding(bindings, call("reshape", vec![
                    adj.clone(),
                    CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(-1)]),
                ]));
                let grad = emit_binding(bindings, call("@", vec![sym(&oh_col), sym(&adj_row)]));
                acc_arg(&args[0], sym(&grad), adj_names, bindings);
            } else {
                // Tensor indices: oh is [N, V], adj is [N, D]
                // grad = transpose(oh) @ adj  -> [V, D]
                let oh_t = emit_binding(bindings, call("tr", vec![sym(&oh)]));
                let grad = emit_binding(bindings, call("@", vec![sym(&oh_t), adj.clone()]));
                acc_arg(&args[0], sym(&grad), adj_names, bindings);
            }
        }

        "scan" if args.len() == 3 => {
            let vjp_result = emit_binding(bindings, call("__scan_vjp__", vec![
                args[0].clone(),  // lambda
                args[1].clone(),  // init
                args[2].clone(),  // coll
                adj.clone(),      // adj of scan result (carry adjoint)
            ]));
            // __scan_vjp__ returns [adj_init, adj_coll]: extract with first/second
            let adj_init = emit_binding(bindings, call("first", vec![sym(&vjp_result)]));
            let adj_coll = emit_binding(bindings, call("second", vec![sym(&vjp_result)]));
            acc_arg(&args[1], sym(&adj_init), adj_names, bindings);
            acc_arg(&args[2], sym(&adj_coll), adj_names, bindings);
        }

        "reduce" if args.len() == 3 => {
            let vjp_result = emit_binding(bindings, call("__scan_vjp__", vec![
                args[0].clone(),  // lambda
                args[1].clone(),  // init
                args[2].clone(),  // coll
                adj.clone(),      // adj of reduce result (final carry adjoint)
            ]));
            // __scan_vjp__ returns [adj_init, adj_coll]: extract with first/second
            let adj_init = emit_binding(bindings, call("first", vec![sym(&vjp_result)]));
            let adj_coll = emit_binding(bindings, call("second", vec![sym(&vjp_result)]));
            acc_arg(&args[1], sym(&adj_init), adj_names, bindings);
            acc_arg(&args[2], sym(&adj_coll), adj_names, bindings);
        }

        _ => return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location),
    }
    Ok(())
}

/// Look up the shape of an argument (must be a Symbol to resolve).
/// Parse a keyword argument from a function call's args list.
/// E.g. for args [x, Keyword("axis"), Integer(-1)], returns Some(-1).
fn parse_keyword_int(args: &[CompiledExpr], key: &str) -> Option<i64> {
    for (i, arg) in args.iter().enumerate() {
        if let CompiledExpr::Keyword(k) = arg
            && k == key
            && let Some(CompiledExpr::Integer(n)) = args.get(i + 1) {
                    return Some(*n);
                }
    }
    None
}

fn arg_shape(arg: &CompiledExpr, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    match arg {
        CompiledExpr::Symbol(s) => shapes.get(s).cloned(),
        _ => None,
    }
}

/// Emit a new named binding and return its name.
fn emit_binding(
    bindings: &mut Vec<(String, CompiledExpr)>,
    value: CompiledExpr,
) -> String {
    let name = fresh_grad_name();
    bindings.push((name.clone(), value));
    name
}

/// Accumulate adjoint for the Symbol inside `arg`.
fn acc_arg(
    arg: &CompiledExpr,
    contribution: CompiledExpr,
    adj_names: &mut HashMap<String, String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
) {
    match arg {
        CompiledExpr::Symbol(s) => {
            accumulate_named(s, contribution, adj_names, bindings);
        }
        CompiledExpr::GetTupleElement { param, .. } => {
            accumulate_named(param, contribution, adj_names, bindings);
        }
        _ => {} // Literal: no gradient
    }
}

#[cfg(test)]
mod tests {
    use super::{reverse_grad, to_anf};
    use crate::core::error::{SheafError, SourceLocation};
    use crate::core::expr::{BindingPattern, CompiledExpr};
    use std::collections::HashMap;
    use std::rc::Rc;

    fn symbol(name: &str) -> CompiledExpr {
        CompiledExpr::Symbol(name.to_string())
    }

    fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.to_string(),
            args,
            loc: None,
        }
    }

    fn missing_rule(error: SheafError, operation: &str) {
        assert!(matches!(
            error,
            SheafError::AutodiffMissingRule {
                operation: actual, ..
            } if actual == operation
        ));
    }

    #[test]
    fn rejects_unknown_operation_depending_on_parameter() {
        let bindings = vec![("result".to_string(), call("unknown", vec![symbol("x")]))];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "unknown");
    }

    #[test]
    fn tracks_dependency_through_multiple_anf_bindings() {
        let bindings = vec![
            ("a".to_string(), call("unknown", vec![symbol("x")])),
            ("result".to_string(), call("+", vec![symbol("a"), CompiledExpr::Float(1.0)])),
        ];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "unknown");
    }

    #[test]
    fn rejects_nonliteral_power_exponent() {
        let bindings = vec![(
            "result".to_string(),
            call("**", vec![symbol("x"), symbol("exponent")]),
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "**");
    }

    #[test]
    fn rejects_unsupported_einsum_form() {
        let bindings = vec![(
            "result".to_string(),
            call("einsum", vec![CompiledExpr::String("ij".to_string()), symbol("x"), symbol("y")]),
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "einsum");
    }

    #[test]
    fn rejects_active_non_call_anf_form() {
        let bindings = vec![(
            "result".to_string(),
            CompiledExpr::If {
                condition: Box::new(CompiledExpr::Boolean(true)),
                then_branch: Box::new(symbol("x")),
                else_branch: Some(Box::new(CompiledExpr::Float(0.0))),
            },
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "if");
    }

    #[test]
    fn permits_unknown_operation_independent_of_parameters() {
        let bindings = vec![("result".to_string(), call("unknown", vec![CompiledExpr::Float(1.0)]))];
        let (backward, gradients) = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert_eq!(backward.len(), 1);
        assert!(gradients.is_empty());
    }

    #[test]
    fn permits_unknown_operation_with_a_static_shape_argument() {
        let bindings = vec![
            ("shape".to_string(), call("shape", vec![symbol("x")])),
            ("classes".to_string(), call("last", vec![symbol("shape")])),
            (
                "result".to_string(),
                call("one-hot", vec![CompiledExpr::Integer(0), symbol("classes")]),
            ),
        ];
        let (_, gradients) = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(gradients.is_empty());
    }

    #[test]
    fn ignores_unknown_operation_in_dead_binding() {
        let bindings = vec![
            ("dead".to_string(), call("unknown", vec![symbol("x")])),
            ("result".to_string(), call("+", vec![symbol("x"), CompiledExpr::Float(1.0)])),
        ];
        let (_, gradients) = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(gradients.contains_key("x"));
    }

    #[test]
    fn stop_gradient_is_an_explicit_zero_gradient() {
        let bindings = vec![("result".to_string(), call("stop-gradient", vec![symbol("x")]))];
        let (_, gradients) = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(gradients.is_empty());
    }

    #[test]
    fn preserves_unknown_operation_name_and_location() {
        let location = SourceLocation::new(7, 3, Rc::from("loss.shf"));
        let expr = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Simple("input".to_string()), symbol("x"))],
            body: Box::new(CompiledExpr::FunctionCall {
                name: "unknown".to_string(),
                args: vec![symbol("input")],
                loc: Some(location.clone()),
            }),
        };
        let anf = to_anf(&expr);
        let CompiledExpr::Let { bindings, body } = anf else {
            panic!("expected ANF binding");
        };
        let bindings: Vec<(String, CompiledExpr)> = bindings.into_iter().map(|(name, value)| {
            (name.as_simple().unwrap().to_string(), value)
        }).collect();
        let error = reverse_grad(
            &bindings,
            &body,
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        assert!(matches!(
            error,
            SheafError::AutodiffMissingRule {
                operation,
                location: Some(actual),
            } if operation == "unknown" && actual == location
        ));
    }
}
