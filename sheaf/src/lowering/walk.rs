// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use crate::core::expr::CompiledExpr;

pub(crate) fn walk_expr(expr: &CompiledExpr, visit: &mut impl FnMut(&CompiledExpr)) {
    visit(expr);

    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, value) in bindings {
                walk_expr(value, visit);
            }
            walk_expr(body, visit);
        }
        CompiledExpr::Do(exprs)
        | CompiledExpr::Vector(exprs)
        | CompiledExpr::Tuple(exprs) => {
            for expr in exprs {
                walk_expr(expr, visit);
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            walk_expr(condition, visit);
            walk_expr(then_branch, visit);
            if let Some(else_branch) = else_branch {
                walk_expr(else_branch, visit);
            }
        }
        CompiledExpr::Lambda { body, .. } => walk_expr(body, visit),
        CompiledExpr::LambdaCall { callee, args } => {
            walk_expr(callee, visit);
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        CompiledExpr::Dict(pairs) => {
            for (key, value) in pairs {
                walk_expr(key, visit);
                walk_expr(value, visit);
            }
        }
        CompiledExpr::Repeat { count, acc_init, body, .. } => {
            walk_expr(count, visit);
            walk_expr(acc_init, visit);
            walk_expr(body, visit);
        }
        CompiledExpr::While { condition, acc_init, body, .. } => {
            walk_expr(condition, visit);
            walk_expr(acc_init, visit);
            walk_expr(body, visit);
        }
        CompiledExpr::Guard { expr, .. } => walk_expr(expr, visit),
        CompiledExpr::Def { value, .. } => walk_expr(value, visit),
        CompiledExpr::Integer(_)
        | CompiledExpr::Float(_)
        | CompiledExpr::Boolean(_)
        | CompiledExpr::Nil
        | CompiledExpr::String(_)
        | CompiledExpr::Keyword(_)
        | CompiledExpr::Symbol(_)
        | CompiledExpr::FunctionRef(_)
        | CompiledExpr::Quoted(_)
        | CompiledExpr::GetTupleElement { .. }
        | CompiledExpr::ValueAndGrad { .. } => {}
    }
}
