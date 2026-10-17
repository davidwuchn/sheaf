// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use crate::core::expr::CompiledExpr;

pub(crate) fn walk_expr(expr: &CompiledExpr, visit: &mut impl FnMut(&CompiledExpr)) {
    visit(expr);
    expr.for_each_child(|child| walk_expr(child, visit));
}
