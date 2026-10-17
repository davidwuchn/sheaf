// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Lexical binding analysis.

use crate::core::expr::{BindingPattern, CompiledExpr};
use std::collections::HashSet;

pub fn pattern_names(pattern: &BindingPattern, names: &mut HashSet<String>) {
    match pattern {
        BindingPattern::Simple(name) => { names.insert(name.clone()); }
        BindingPattern::Destructure(patterns) => {
            for pattern in patterns { pattern_names(pattern, names); }
        }
    }
}

/// Collect local references that are not bound within the expression.
pub fn free_locals(expr: &CompiledExpr) -> Vec<String> {
    let mut names = HashSet::new();
    collect_locals(expr, &HashSet::new(), &mut names);
    let mut names: Vec<_> = names.into_iter().collect();
    names.sort();
    names
}

fn collect_locals(expr: &CompiledExpr, bound: &HashSet<String>, names: &mut HashSet<String>) {
    match expr {
        CompiledExpr::Symbol(name) | CompiledExpr::GetTupleElement { param: name, .. } => {
            if !bound.contains(name) { names.insert(name.clone()); }
        }
        CompiledExpr::Let { bindings, body } => {
            let mut scope = bound.clone();
            for (pattern, value) in bindings {
                collect_locals(value, &scope, names);
                pattern_names(pattern, &mut scope);
            }
            collect_locals(body, &scope, names);
        }
        CompiledExpr::Lambda { params, body, .. } => {
            let mut scope = bound.clone();
            scope.extend(params.iter().cloned());
            collect_locals(body, &scope, names);
        }
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
            collect_locals(count, bound, names);
            collect_locals(acc_init, bound, names);
            let mut scope = bound.clone();
            scope.insert(index_var.clone());
            scope.insert(acc_var.clone());
            collect_locals(body, &scope, names);
        }
        CompiledExpr::While { condition, acc_var, acc_init, body } => {
            collect_locals(acc_init, bound, names);
            let mut scope = bound.clone();
            scope.insert(acc_var.clone());
            collect_locals(condition, &scope, names);
            collect_locals(body, &scope, names);
        }
        _ => expr.for_each_child(|child| collect_locals(child, bound, names)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(name: &str) -> CompiledExpr { CompiledExpr::Symbol(name.to_string()) }

    #[test]
    fn sequential_bindings_and_nested_patterns_hide_only_bound_references() {
        let body = CompiledExpr::Let {
            bindings: vec![
                (BindingPattern::Simple("x".into()), symbol("x")),
                (BindingPattern::Destructure(vec![BindingPattern::Simple("y".into()),
                    BindingPattern::Destructure(vec![BindingPattern::Simple("z".into())])]),
                 CompiledExpr::Tuple(vec![symbol("x"), symbol("outer")])),
            ],
            body: Box::new(CompiledExpr::Tuple(vec![symbol("x"), symbol("y"),
                CompiledExpr::GetTupleElement { param: "z".into(), indices: vec![0] }, symbol("external")])),
        };
        assert_eq!(free_locals(&body), ["external", "outer", "x"]);
    }

    #[test]
    fn lambda_parameters_do_not_hide_outer_container_inputs() {
        let body = CompiledExpr::Lambda { params: vec!["x".into()], body: Box::new(CompiledExpr::Tuple(vec![
            symbol("x"), CompiledExpr::Dict(vec![(symbol("key"), symbol("outer"))]),
            CompiledExpr::GetTupleElement { param: "params".into(), indices: vec![0] },
            CompiledExpr::FunctionRef("module-function".into()),
        ])) };
        assert_eq!(free_locals(&body), ["key", "outer", "params"]);
    }

    #[test]
    fn loop_initializers_and_bound_bodies_have_separate_scopes() {
        let repeat = CompiledExpr::Repeat {
            index_var: "i".into(), count: Box::new(symbol("i")), acc_var: "a".into(),
            acc_init: Box::new(symbol("a")),
            body: Box::new(CompiledExpr::Tuple(vec![symbol("i"), symbol("a"), symbol("outside")])),
        };
        assert_eq!(free_locals(&repeat), ["a", "i", "outside"]);
        let while_loop = CompiledExpr::While {
            acc_var: "a".into(), acc_init: Box::new(symbol("initial")),
            condition: Box::new(CompiledExpr::Tuple(vec![symbol("a"), symbol("condition")])),
            body: Box::new(CompiledExpr::Tuple(vec![symbol("a"), symbol("body")])),
        };
        assert_eq!(free_locals(&while_loop), ["body", "condition", "initial"]);
    }
}
