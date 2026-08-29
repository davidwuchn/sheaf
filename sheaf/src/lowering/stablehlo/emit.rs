// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Function and module emission: declarations, calls, returns, tuples.

use super::{Register, StableHLOEmitter, StableHLOType};
use std::fmt::Write;

impl StableHLOEmitter {
    pub fn emit_get_tuple_element(
        &mut self,
        tuple_reg: &Register,
        tuple_ty: &StableHLOType,
        index: usize,
        result_ty: &StableHLOType,
    ) -> Register {
        if let Some(constituents) = self.virtual_tuples.get(tuple_reg) {
            return constituents[index].0;
        }
        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.get_tuple_element {} [{index}] : ({}) -> {}",
            reg.to_mlir(),
            tuple_reg.to_mlir(),
            tuple_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        reg
    }

    pub fn emit_tuple(
        &mut self,
        regs: &[Register],
        types: &[StableHLOType],
    ) -> (Register, StableHLOType) {
        let vreg = self.fresh_register();
        let constituents: Vec<_> = regs.iter().zip(types.iter())
            .map(|(r, t)| (*r, t.clone()))
            .collect();
        self.virtual_tuples.insert(vreg, constituents);
        (vreg, StableHLOType::Tuple(types.to_vec(), None))
    }

    pub fn emit_tuple_with_keys(
        &mut self,
        regs: &[Register],
        types: &[StableHLOType],
        keys: &Option<Vec<String>>,
    ) -> (Register, StableHLOType) {
        let (reg, _) = self.emit_tuple(regs, types);
        (reg, StableHLOType::Tuple(types.to_vec(), keys.clone()))
    }

    pub fn emit_return(&mut self, reg: &Register, ty: &StableHLOType) {
        self.body
            .push(format!("    return {} : {}", reg.to_mlir(), ty.to_mlir()));
    }

    pub fn emit_return_multi(&mut self, regs: &[Register], types: &[StableHLOType]) {
        let vals = regs
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        let tys = types
            .iter()
            .map(|t| t.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        self.body.push(format!("    return {} : {}", vals, tys));
    }

    pub fn emit_func_declaration_multi(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_types: &[StableHLOType],
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        let ret_str = if return_types.len() == 1 {
            return_types[0].to_mlir()
        } else {
            format!(
                "({})",
                return_types
                    .iter()
                    .map(|t| t.to_mlir())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let _ = writeln!(output, " func.func @{}({}) -> {} {{", sanitized_name, params_str, ret_str);

        for line in body_instructions {
            let _ = writeln!(output, "{}", line);
        }

        let _ = writeln!(output, " }}");
        output
    }

    pub fn emit_func_declaration(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_type: &StableHLOType,
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        let _ = writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name,
            params_str,
            return_type.to_mlir()
        );

        for line in body_instructions {
            let _ = writeln!(output, "{}", line);
        }

        let _ = writeln!(output, " }}");
        output
    }

    pub fn emit_func_call(
        &mut self,
        name: &str,
        arg_registers: &[Register],
        arg_types: &[StableHLOType],
        return_type: &StableHLOType,
    ) -> Register {
        let sanitized_name = Self::sanitize_func_name(name);
        let reg = self.fresh_register();

        let args_str = arg_registers
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        let arg_types_str = arg_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = func.call @{}({}) : ({}) -> {}",
            reg.to_mlir(),
            sanitized_name,
            args_str,
            arg_types_str,
            return_type.to_mlir()
        ));

        reg
    }

    pub(crate) fn sanitize_func_name(name: &str) -> String {
        name.replace('-', "_").replace('?', "_q").replace('!', "_b")
    }

    pub fn emit_function_body(&self, name: &str, result_ty: &StableHLOType) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        let _ = writeln!(output, "// Generated by Sheaf");
        let _ = writeln!(output, "//");
        let _ = writeln!(output);
        let _ = writeln!(output, "module {{");
        let _ = writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        );

        for line in &self.body {
            let _ = writeln!(output, "{}", line);
        }

        let _ = writeln!(output, " }}");
        let _ = writeln!(output, "}}");

        output
    }

    pub fn emit_function(&mut self, name: &str, expr: &crate::core::ast::SheafValue) -> String {
        let (result_reg, result_ty) = self.compile_expr(expr);
        self.emit_return(&result_reg, &result_ty);

        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        let _ = writeln!(output, "// Generated by Sheaf");
        let _ = writeln!(output, "//");
        let _ = writeln!(output, "// Source: {}", expr);
        let _ = writeln!(output);
        let _ = writeln!(output, "module {{");
        let _ = writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        );

        for line in &self.body {
            let _ = writeln!(output, "{}", line);
        }

        let _ = writeln!(output, " }}");
        let _ = writeln!(output, "}}");

        output
    }

    pub fn emit_module(func_declarations: &[String]) -> String {
        Self::emit_module_named(None, func_declarations)
    }

    pub fn emit_module_named(name: Option<&str>, func_declarations: &[String]) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "// Generated by Sheaf");
        let _ = writeln!(output, "//");
        let _ = writeln!(output);
        match name {
            Some(name) => {
                let _ = writeln!(output, "module @{} {{", name);
            }
            None => {
                let _ = writeln!(output, "module {{");
            }
        }

        for func_decl in func_declarations {
            let _ = write!(output, "{}", func_decl);
        }

        let _ = writeln!(output, "}}");
        output
    }
}

#[cfg(test)]
mod module_name_tests {
    use super::StableHLOEmitter;

    #[test]
    fn named_module_scopes_exported_functions() {
        let mlir = StableHLOEmitter::emit_module_named(
            Some("first"),
            &["  func.func @run() { return }\n".to_string()],
        );
        assert!(mlir.contains("module @first {"));
        assert!(mlir.contains("func.func @run"));
    }
}
