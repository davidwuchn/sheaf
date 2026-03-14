// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Helper types and free functions for codegen.

use crate::core::compiler::CompiledExpr;
use std::collections::HashSet;

/// A leaf of a tuple parameter: maps indices to a synthetic symbol name.
#[derive(Debug, Clone)]
pub(crate) struct TupleLeaf {
    /// Tuple access indices (e.g. [0, 1] for second field of first sub-tuple)
    pub(crate) indices: Vec<usize>,
    /// Synthetic symbol name used in the expanded body (e.g. "p__0_1")
    pub(crate) symbol: String,
}

/// Collect all unique `GetTupleElement` leaves referencing `param_name` in an expression.
pub(crate) fn collect_tuple_leaves(expr: &CompiledExpr, param_name: &str) -> Vec<TupleLeaf> {
    let mut leaves = Vec::new();
    let mut seen = HashSet::new();
    collect_tuple_leaves_rec(expr, param_name, &mut leaves, &mut seen);
    leaves
}

fn collect_tuple_leaves_rec(
    expr: &CompiledExpr,
    param_name: &str,
    out: &mut Vec<TupleLeaf>,
    seen: &mut HashSet<Vec<usize>>,
) {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } if param == param_name => {
            if seen.insert(indices.clone()) {
                let symbol = format!("{}_{}", param_name, indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_"));
                out.push(TupleLeaf {
                    indices: indices.clone(),
                    symbol,
                });
            }
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                collect_tuple_leaves_rec(a, param_name, out, seen);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                collect_tuple_leaves_rec(v, param_name, out, seen);
            }
            collect_tuple_leaves_rec(body, param_name, out, seen);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_tuple_leaves_rec(e, param_name, out, seen);
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            collect_tuple_leaves_rec(condition, param_name, out, seen);
            collect_tuple_leaves_rec(then_branch, param_name, out, seen);
            if let Some(e) = else_branch {
                collect_tuple_leaves_rec(e, param_name, out, seen);
            }
        }
        CompiledExpr::Lambda { body, .. } => {
            collect_tuple_leaves_rec(body, param_name, out, seen);
        }
        CompiledExpr::LambdaCall { callee, args } => {
            collect_tuple_leaves_rec(callee, param_name, out, seen);
            for a in args {
                collect_tuple_leaves_rec(a, param_name, out, seen);
            }
        }
        _ => {}
    }
}

/// Replace all `GetTupleElement { param, indices }` referencing `param_name`
/// with `Symbol(synthetic_name)` for autodiff.
pub(crate) fn expand_tuple_to_symbols(expr: &CompiledExpr, param_name: &str) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } if param == param_name => {
            let symbol = format!("{}_{}", param_name, indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_"));
            CompiledExpr::Symbol(symbol)
        }
        CompiledExpr::FunctionCall { name, args, .. } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| expand_tuple_to_symbols(a, param_name)).collect(),
            loc: None,
        },
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings: bindings.iter().map(|(k, v)| (k.clone(), expand_tuple_to_symbols(v, param_name))).collect(),
            body: Box::new(expand_tuple_to_symbols(body, param_name)),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| expand_tuple_to_symbols(e, param_name)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(expand_tuple_to_symbols(condition, param_name)),
            then_branch: Box::new(expand_tuple_to_symbols(then_branch, param_name)),
            else_branch: else_branch.as_ref().map(|e| Box::new(expand_tuple_to_symbols(e, param_name))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(expand_tuple_to_symbols(body, param_name)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(expand_tuple_to_symbols(callee, param_name)),
            args: args.iter().map(|a| expand_tuple_to_symbols(a, param_name)).collect(),
        },
        other => other.clone(),
    }
}

/// Try to flatten a `Vector` of `CompiledExpr` into a constant tensor.
///
/// Returns `Some((flat_data, shape))` if every leaf is a numeric literal
/// (Float or Integer) and all sub-vectors have consistent dimensions.
/// Returns `None` if any element is a non-literal expression (Symbol,
/// FunctionCall, etc.) -- the caller should then emit a tuple or report
/// an error.
///
/// Works recursively for arbitrary nesting depth:
///   `[1.0 2.0]`                  -> `([1.0, 2.0], [2])`
///   `[[1.0 2.0] [3.0 4.0]]`     -> `([1.0, 2.0, 3.0, 4.0], [2, 2])`
///   `[[[1] [2]] [[3] [4]]]`     -> `([1.0, 2.0, 3.0, 4.0], [2, 2, 1])`
pub(super) fn try_flatten_to_constant(elements: &[CompiledExpr]) -> Option<(Vec<f64>, Vec<i64>)> {
    if elements.is_empty() {
        return Some((vec![], vec![0]));
    }

    match &elements[0] {
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) => {
            // Leaf level: all elements must be numeric
            let mut data = Vec::with_capacity(elements.len());
            for e in elements {
                match e {
                    CompiledExpr::Float(x) => data.push(*x),
                    CompiledExpr::Integer(n) => data.push(*n as f64),
                    _ => return None,
                }
            }
            Some((data, vec![elements.len() as i64]))
        }
        CompiledExpr::Vector(_) => {
            // Nested: recurse into each sub-vector, check shapes are uniform
            let mut all_data = Vec::new();
            let mut inner_shape: Option<Vec<i64>> = None;

            for e in elements {
                let sub = match e {
                    CompiledExpr::Vector(sub_elems) => try_flatten_to_constant(sub_elems)?,
                    _ => return None, // mixed Vector / non-Vector
                };
                let (sub_data, sub_shape) = sub;
                match &inner_shape {
                    None => inner_shape = Some(sub_shape),
                    Some(expected) if *expected != sub_shape => return None, // ragged
                    _ => {}
                }
                all_data.extend(sub_data);
            }

            let mut shape = vec![elements.len() as i64];
            shape.extend(inner_shape.unwrap_or_default());
            Some((all_data, shape))
        }
        _ => None,
    }
}
