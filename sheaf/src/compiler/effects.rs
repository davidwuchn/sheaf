// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Static side-effect analysis for `CompiledExpr`.
//!
//! A function has side effects if its body calls any builtin that performs I/O,
//! random number generation, or any other operation that is not a pure
//! mathematical transformation.
//!
//! `sheaf build` refuses to compile functions that have side effects; the
//! interpreter can use this analysis to suggest compilation for pure files.

use crate::core::compiler::CompiledExpr;

/// Names of builtins that have side effects.
///
/// These are calls that cannot be emitted as StableHLO:
/// - I/O: `print`, `io`
/// - Randomness: all `random-*` variants
const EFFECTFUL_BUILTINS: &[&str] = &[
    "print",
    "io",
    "random-key",
    "random-split",
    "random-normal",
    "random-uniform",
    "random-randint",
    "random-bernoulli",
    "random-choice",
];

/// Names of higher-order functions that cannot be compiled to StableHLO.
/// Detected before codegen so the build output shows a clear reason.
pub const HOF_BUILTINS: &[&str] = &[
    "map", "filter", "sort",
    "grad", "jit",
];

/// A single side-effect site found in a function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSite {
    /// The name of the effectful builtin called.
    pub name: String,
}

impl EffectSite {
    fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }
}

/// Check whether a `CompiledExpr` has side effects.
///
/// Returns `true` if any effectful builtin is called anywhere in the expression,
/// including nested lambdas and sub-expressions.
pub fn has_side_effects(expr: &CompiledExpr) -> bool {
    !collect_effects(expr).is_empty()
}

/// Collect the names of any higher-order functions used in the expression.
/// Returns a deduplicated list (e.g. `["map", "reduce"]`).
pub fn collect_hof_calls(expr: &CompiledExpr) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    collect_hof_rec(expr, &mut found);
    found.sort();
    found.dedup();
    found
}

fn collect_hof_rec(expr: &CompiledExpr, out: &mut Vec<String>) {
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            if HOF_BUILTINS.contains(&name.as_str()) {
                out.push(name.clone());
            }
            for arg in args {
                collect_hof_rec(arg, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings { collect_hof_rec(v, out); }
            collect_hof_rec(body, out);
        }
        CompiledExpr::Do(exprs) => { for e in exprs { collect_hof_rec(e, out); } }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            collect_hof_rec(condition, out);
            collect_hof_rec(then_branch, out);
            if let Some(e) = else_branch { collect_hof_rec(e, out); }
        }
        CompiledExpr::Lambda { body, .. } => { collect_hof_rec(body, out); }
        CompiledExpr::LambdaCall { callee, args } => {
            collect_hof_rec(callee, out);
            for arg in args { collect_hof_rec(arg, out); }
        }
        CompiledExpr::Vector(exprs) => { for e in exprs { collect_hof_rec(e, out); } }
        CompiledExpr::Repeat { count, acc_init, body, .. } => {
            collect_hof_rec(count, out);
            collect_hof_rec(acc_init, out);
            collect_hof_rec(body, out);
        }
        // ValueAndGrad (deferred form) is higher-order; InlineValueAndGrad is compilable
        CompiledExpr::ValueAndGrad { .. } => {
            out.push("value-and-grad".to_string());
        }
        _ => {}
    }
}

/// Collect all side-effect sites in a `CompiledExpr`.
///
/// Traverses the entire expression tree and returns every call to an
/// effectful builtin, in encounter order.
pub fn collect_effects(expr: &CompiledExpr) -> Vec<EffectSite> {
    let mut sites = Vec::new();
    collect_effects_rec(expr, &mut sites);
    sites
}

fn collect_effects_rec(expr: &CompiledExpr, out: &mut Vec<EffectSite>) {
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            if EFFECTFUL_BUILTINS.contains(&name.as_str()) {
                out.push(EffectSite::new(name));
            }
            for arg in args {
                collect_effects_rec(arg, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, val) in bindings {
                collect_effects_rec(val, out);
            }
            collect_effects_rec(body, out);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_effects_rec(e, out);
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            collect_effects_rec(condition, out);
            collect_effects_rec(then_branch, out);
            if let Some(e) = else_branch {
                collect_effects_rec(e, out);
            }
        }
        CompiledExpr::Lambda { body, .. } => {
            collect_effects_rec(body, out);
        }
        CompiledExpr::LambdaCall { callee, args } => {
            collect_effects_rec(callee, out);
            for arg in args {
                collect_effects_rec(arg, out);
            }
        }
        CompiledExpr::Vector(exprs) => {
            for e in exprs {
                collect_effects_rec(e, out);
            }
        }
        CompiledExpr::Dict(pairs) => {
            for (k, v) in pairs {
                collect_effects_rec(k, out);
                collect_effects_rec(v, out);
            }
        }
        CompiledExpr::Repeat { count, acc_init, body, .. } => {
            collect_effects_rec(count, out);
            collect_effects_rec(acc_init, out);
            collect_effects_rec(body, out);
        }
        CompiledExpr::While { condition, acc_init, body, .. } => {
            collect_effects_rec(condition, out);
            collect_effects_rec(acc_init, out);
            collect_effects_rec(body, out);
        }
        CompiledExpr::Guard { expr, .. } => {
            out.push(EffectSite::new("guard"));
            collect_effects_rec(expr, out);
        }
        // Leaf nodes and nodes with no sub-expressions
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
        | CompiledExpr::ValueAndGrad { .. }
        | CompiledExpr::InlineValueAndGrad { .. } => {}
    }
}

/// Format a list of effect sites into a human-readable message.
///
/// Example: `"print, io"`
pub fn format_effects(sites: &[EffectSite]) -> String {
    let mut seen = std::collections::BTreeSet::new();
    for s in sites {
        seen.insert(s.name.clone());
    }
    seen.into_iter().collect::<Vec<_>>().join(", ")
}
