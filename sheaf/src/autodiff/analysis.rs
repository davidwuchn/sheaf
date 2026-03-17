// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Static analysis of expressions for autodiff compatibility.

use crate::core::expr::CompiledExpr;

/// Check whether an expression contains ops that the symbolic AD cannot differentiate.
///
/// Returns the list of undifferentiable op names found.
pub fn find_undiffable_ops(expr: &CompiledExpr) -> Vec<String> {
    let mut ops = Vec::new();
    find_undiffable_rec(expr, &mut ops);
    ops.sort();
    ops.dedup();
    ops
}

fn find_undiffable_rec(expr: &CompiledExpr, ops: &mut Vec<String>) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            match name.as_str() {
                "get" | "reduce" | "map" | "filter" | "find"
                | "range" | "len" | "first" | "last" | "rest"
                | "cons" | "append" | "concat" | "slice"
                | "shape" | "print" | "io" => ops.push(name.clone()),
                _ => {}
            }
            for a in args { find_undiffable_rec(a, ops); }
        }
        CompiledExpr::LambdaCall { callee, args } => {
            ops.push("LambdaCall".to_string());
            find_undiffable_rec(callee, ops);
            for a in args { find_undiffable_rec(a, ops); }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            ops.push("If".to_string());
            find_undiffable_rec(condition, ops);
            find_undiffable_rec(then_branch, ops);
            if let Some(e) = else_branch { find_undiffable_rec(e, ops); }
        }
        CompiledExpr::While { .. } => ops.push("While".to_string()),
        CompiledExpr::Repeat { .. } => ops.push("Repeat".to_string()),
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings { find_undiffable_rec(v, ops); }
            find_undiffable_rec(body, ops);
        }
        CompiledExpr::Do(exprs) => { for e in exprs { find_undiffable_rec(e, ops); } }
        CompiledExpr::Lambda { body, .. } => find_undiffable_rec(body, ops),
        CompiledExpr::Vector(elems) => { for e in elems { find_undiffable_rec(e, ops); } }
        CompiledExpr::Dict(pairs) => { for (k, v) in pairs { find_undiffable_rec(k, ops); find_undiffable_rec(v, ops); } }
        _ => {}
    }
}

pub fn contains_undiffable_ops(expr: &CompiledExpr) -> bool {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            match name.as_str() {
                "get" | "reduce" | "map" | "filter" | "find"
                | "range" | "len" | "first" | "last" | "rest"
                | "cons" | "append" | "concat" | "slice"
                | "shape" | "print" | "io" => true,
                _ => args.iter().any(contains_undiffable_ops),
            }
        }
        CompiledExpr::LambdaCall { .. } => true,
        CompiledExpr::If { .. } | CompiledExpr::While { .. } | CompiledExpr::Repeat { .. } => true,
        CompiledExpr::Let { bindings, body } => {
            bindings.iter().any(|(_, v)| contains_undiffable_ops(v))
                || contains_undiffable_ops(body)
        }
        CompiledExpr::Do(exprs) => exprs.iter().any(contains_undiffable_ops),
        CompiledExpr::Lambda { body, .. } => contains_undiffable_ops(body),
        CompiledExpr::Vector(elems) => elems.iter().any(contains_undiffable_ops),
        CompiledExpr::Dict(pairs) => {
            pairs.iter().any(|(k, v)| contains_undiffable_ops(k) || contains_undiffable_ops(v))
        }
        _ => false,
    }
}
