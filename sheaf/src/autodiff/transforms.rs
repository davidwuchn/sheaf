// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IR transforms: function inlining and common subexpression elimination.

use crate::core::expr::{BindingPattern, CompiledExpr, FunctionDef};
use std::collections::HashMap;

/// Inline user-defined function calls so that autodiff can see through them.
///
/// Replaces `FunctionCall("f", [a, b])` where `f` is in `registry` with:
//   `Let { bindings: [(p1, a), (p2, b)], body: f.body_compiled }`
///
/// Recurses into the result (the inlined body may itself contain calls).
/// `depth` guards against infinite recursion (mutual/self-recursive functions).
pub fn inline_function_calls(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
) -> CompiledExpr {
    inline_calls_rec(expr, registry, &HashMap::new(), 0)
}

const MAX_INLINE_DEPTH: usize = 16;

fn inline_calls_rec(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    local_lambdas: &HashMap<String, CompiledExpr>,
    depth: usize,
) -> CompiledExpr {
    if depth > MAX_INLINE_DEPTH {
        return expr.clone();
    }

    match expr {
        CompiledExpr::FunctionCall { name, args, loc } => {
            // First, inline in arguments
            let inlined_args: Vec<CompiledExpr> = args
                .iter()
                .map(|a| inline_calls_rec(a, registry, local_lambdas, depth))
                .collect();

            // Try to inline: check local lambda bindings first, then registry
            if let Some(CompiledExpr::Lambda { params, body }) = local_lambdas.get(name.as_str()) {
                let bindings: Vec<(BindingPattern, CompiledExpr)> = params
                    .iter()
                    .zip(inlined_args.iter())
                    .map(|(p, a)| (BindingPattern::Simple(p.clone()), a.clone()))
                    .collect();
                let inlined = CompiledExpr::Let {
                    bindings,
                    body: body.clone(),
                };
                return inline_calls_rec(&inlined, registry, local_lambdas, depth + 1);
            }

            if let Some(func_def) = registry.get(name.as_str())
                && let Some(body) = &func_def.body_compiled
            {
                let bindings: Vec<(BindingPattern, CompiledExpr)> = func_def
                    .params
                    .iter()
                    .zip(inlined_args.iter())
                    .map(|(p, a)| (BindingPattern::Simple(p.clone()), a.clone()))
                    .collect();
                let inlined = CompiledExpr::Let {
                    bindings,
                    body: Box::new(body.clone()),
                };
                return inline_calls_rec(&inlined, registry, local_lambdas, depth + 1);
            }

            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: inlined_args,
                loc: loc.clone(),
            }
        }

        CompiledExpr::Let { bindings, body } => {
            let mut new_lambdas = local_lambdas.clone();
            let new_bindings: Vec<(BindingPattern, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| {
                    let inlined_v = inline_calls_rec(v, registry, &new_lambdas, depth);
                    if let BindingPattern::Simple(name) = k
                        && matches!(&inlined_v, CompiledExpr::Lambda { .. })
                    {
                        new_lambdas.insert(name.clone(), inlined_v.clone());
                    }
                    (k.clone(), inlined_v)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(inline_calls_rec(body, registry, &new_lambdas, depth)),
            }
        }

        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| inline_calls_rec(e, registry, local_lambdas, depth))
                .collect(),
        ),

        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(inline_calls_rec(condition, registry, local_lambdas, depth)),
            then_branch: Box::new(inline_calls_rec(then_branch, registry, local_lambdas, depth)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(inline_calls_rec(e, registry, local_lambdas, depth))),
        },

        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(inline_calls_rec(body, registry, local_lambdas, depth)),
        },

        CompiledExpr::LambdaCall { callee, args } => {
            let inlined_args: Vec<CompiledExpr> = args
                .iter()
                .map(|a| inline_calls_rec(a, registry, local_lambdas, depth))
                .collect();
            let inlined_callee = inline_calls_rec(callee, registry, local_lambdas, depth);

            // If callee is a direct lambda, inline it as a Let binding
            if let CompiledExpr::Lambda { params, body } = &inlined_callee {
                let bindings: Vec<(BindingPattern, CompiledExpr)> = params
                    .iter()
                    .zip(inlined_args.iter())
                    .map(|(p, a)| (BindingPattern::Simple(p.clone()), a.clone()))
                    .collect();
                let inlined = CompiledExpr::Let {
                    bindings,
                    body: body.clone(),
                };
                return inline_calls_rec(&inlined, registry, local_lambdas, depth + 1);
            }

            CompiledExpr::LambdaCall {
                callee: Box::new(inlined_callee),
                args: inlined_args,
            }
        }

        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| inline_calls_rec(e, registry, local_lambdas, depth))
                .collect(),
        ),

        // Leaves: no recursion needed
        _ => expr.clone(),
    }
}

/// Fold `(get <dict_literal> :key)` into the dict value directly.
pub fn fold_dict_gets(expr: &CompiledExpr) -> CompiledExpr {
    fold_dict_gets_rec(expr, &HashMap::new())
}

fn fold_dict_gets_rec(
    expr: &CompiledExpr,
    dict_bindings: &HashMap<String, Vec<(CompiledExpr, CompiledExpr)>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_dicts = dict_bindings.clone();
            let new_bindings: Vec<(BindingPattern, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| {
                    let folded = fold_dict_gets_rec(v, &new_dicts);
                    if let (BindingPattern::Simple(name), CompiledExpr::Dict(pairs)) = (k, &folded) {
                        new_dicts.insert(name.clone(), pairs.clone());
                    }
                    (k.clone(), folded)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(fold_dict_gets_rec(body, &new_dicts)),
            }
        }

        CompiledExpr::FunctionCall { name, args, loc } if name == "get" && args.len() == 2 => {
            // (get sym :key) where sym is bound to a dict literal
            if let (CompiledExpr::Symbol(dict_name), CompiledExpr::Keyword(key)) =
                (&args[0], &args[1])
                && let Some(pairs) = dict_bindings.get(dict_name.as_str())
            {
                for (k, v) in pairs
                {
                    if let CompiledExpr::Keyword(kw) = k
                        && kw == key
                    {
                        return fold_dict_gets_rec(v, dict_bindings);
                    }
                }
            }
            // Not foldable, recurse into args
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| fold_dict_gets_rec(a, dict_bindings)).collect(),
                loc: loc.clone(),
            }
        }

        CompiledExpr::FunctionCall { name, args, loc } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| fold_dict_gets_rec(a, dict_bindings)).collect(),
            loc: loc.clone(),
        },

        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| fold_dict_gets_rec(e, dict_bindings)).collect(),
        ),

        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(fold_dict_gets_rec(condition, dict_bindings)),
            then_branch: Box::new(fold_dict_gets_rec(then_branch, dict_bindings)),
            else_branch: else_branch.as_ref().map(|e| Box::new(fold_dict_gets_rec(e, dict_bindings))),
        },

        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(fold_dict_gets_rec(body, dict_bindings)),
        },

        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| fold_dict_gets_rec(e, dict_bindings)).collect(),
        ),

        _ => expr.clone(),
    }
}
