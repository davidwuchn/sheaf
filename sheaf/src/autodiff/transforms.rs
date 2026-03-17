// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IR transforms: function inlining and common subexpression elimination.

use crate::core::expr::{CompiledExpr, FunctionDef};
use std::collections::HashMap;

/// Inline user-defined function calls so that autodiff can see through them.
///
/// Replaces `FunctionCall("f", [a, b])` where `f` is in `registry` with:
///   `Let { bindings: [(p1, a), (p2, b)], body: f.body_compiled }`
///
/// Recurses into the result (the inlined body may itself contain calls).
/// `depth` guards against infinite recursion (mutual/self-recursive functions).
pub fn inline_function_calls(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
) -> CompiledExpr {
    inline_calls_rec(expr, registry, 0)
}

const MAX_INLINE_DEPTH: usize = 16;

fn inline_calls_rec(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    depth: usize,
) -> CompiledExpr {
    if depth > MAX_INLINE_DEPTH {
        return expr.clone();
    }

    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            // First, inline in arguments
            let inlined_args: Vec<CompiledExpr> = args
                .iter()
                .map(|a| inline_calls_rec(a, registry, depth))
                .collect();

            // Try to inline this call if it's a user-defined function
            if let Some(func_def) = registry.get(name.as_str()) {
                if let Some(body) = &func_def.body_compiled {
                    let bindings: Vec<(String, CompiledExpr)> = func_def
                        .params
                        .iter()
                        .zip(inlined_args.iter())
                        .map(|(p, a)| (p.clone(), a.clone()))
                        .collect();
                    let inlined = CompiledExpr::Let {
                        bindings,
                        body: Box::new(body.clone()),
                    };
                    // Recurse into the inlined body
                    return inline_calls_rec(&inlined, registry, depth + 1);
                }
            }

            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: inlined_args,
                loc: None,
            }
        }

        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<(String, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| (k.clone(), inline_calls_rec(v, registry, depth)))
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(inline_calls_rec(body, registry, depth)),
            }
        }

        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| inline_calls_rec(e, registry, depth))
                .collect(),
        ),

        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(inline_calls_rec(condition, registry, depth)),
            then_branch: Box::new(inline_calls_rec(then_branch, registry, depth)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(inline_calls_rec(e, registry, depth))),
        },

        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(inline_calls_rec(body, registry, depth)),
        },

        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(inline_calls_rec(callee, registry, depth)),
            args: args
                .iter()
                .map(|a| inline_calls_rec(a, registry, depth))
                .collect(),
        },

        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| inline_calls_rec(e, registry, depth))
                .collect(),
        ),

        // Leaves: no recursion needed
        _ => expr.clone(),
    }
}

/// Common Subexpression Elimination.
///
/// Traverses the expression tree, finds structurally identical non-trivial
/// sub-expressions that appear more than once, and hoists them into `Let`
/// bindings so they are computed only once.
///
/// A sub-expression is "non-trivial" if it is a `FunctionCall` (not a leaf).
pub fn cse(expr: CompiledExpr) -> CompiledExpr {
    let mut counts: HashMap<String, usize> = HashMap::new();
    count_exprs(&expr, &mut counts);

    let mut seen_keys: Vec<String> = Vec::new();
    let mut bindings: Vec<(String, CompiledExpr)> = Vec::new();
    let mut subst: HashMap<String, String> = HashMap::new();

    collect_cse_candidates(&expr, &counts, &mut seen_keys, &mut bindings, &mut subst);

    if bindings.is_empty() {
        return expr;
    }

    let body = substitute(expr, &subst);

    CompiledExpr::Let {
        bindings,
        body: Box::new(body),
    }
}

fn expr_key(expr: &CompiledExpr) -> String {
    format!("{:?}", expr)
}

fn is_trivial(expr: &CompiledExpr) -> bool {
    matches!(
        expr,
        CompiledExpr::Symbol(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Integer(_)
            | CompiledExpr::GetTupleElement { .. }
    )
}

fn count_exprs(expr: &CompiledExpr, counts: &mut HashMap<String, usize>) {
    if is_trivial(expr) {
        return;
    }
    let key = expr_key(expr);
    *counts.entry(key).or_insert(0) += 1;

    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                count_exprs(a, counts);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                count_exprs(v, counts);
            }
            count_exprs(body, counts);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                count_exprs(e, counts);
            }
        }
        _ => {}
    }
}

fn collect_cse_candidates(
    expr: &CompiledExpr,
    counts: &HashMap<String, usize>,
    seen_keys: &mut Vec<String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
    subst: &mut HashMap<String, String>,
) {
    if is_trivial(expr) {
        return;
    }
    let key = expr_key(expr);
    if counts.get(&key).copied().unwrap_or(0) > 1 {
        if !seen_keys.contains(&key) {
            seen_keys.push(key.clone());
            let name = format!("__cse{}", bindings.len());
            subst.insert(key, name.clone());
            bindings.push((name, expr.clone()));
        }
        return;
    }

    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                collect_cse_candidates(a, counts, seen_keys, bindings, subst);
            }
        }
        CompiledExpr::Let { bindings: b, body } => {
            for (_, v) in b {
                collect_cse_candidates(v, counts, seen_keys, bindings, subst);
            }
            collect_cse_candidates(body, counts, seen_keys, bindings, subst);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_cse_candidates(e, counts, seen_keys, bindings, subst);
            }
        }
        _ => {}
    }
}

fn substitute(
    expr: CompiledExpr,
    subst: &HashMap<String, String>,
) -> CompiledExpr {
    if is_trivial(&expr) {
        return expr;
    }
    let key = expr_key(&expr);
    if let Some(name) = subst.get(&key) {
        return CompiledExpr::Symbol(name.clone());
    }
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            let args = args.into_iter().map(|a| substitute(a, subst)).collect();
            CompiledExpr::FunctionCall { name, args, loc: None }
        }
        CompiledExpr::Let { bindings, body } => {
            let bindings = bindings
                .into_iter()
                .map(|(k, v)| (k, substitute(v, subst)))
                .collect();
            CompiledExpr::Let {
                bindings,
                body: Box::new(substitute(*body, subst)),
            }
        }
        CompiledExpr::Do(exprs) => {
            CompiledExpr::Do(exprs.into_iter().map(|e| substitute(e, subst)).collect())
        }
        other => other,
    }
}
