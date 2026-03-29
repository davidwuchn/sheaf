// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Function and module emission: declarations, calls, returns, tuples.

use super::{Register, StableHLOEmitter, StableHLOType};
use std::fmt::Write;

impl StableHLOEmitter {
    /// Extract field from a tuple at a given index.
    /// Virtual tuples resolve without emitting MLIR.
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
        // Fallback: real tuple (shouldn't happen with virtual-only approach)
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

    /// Pack registers into a virtual tuple (no MLIR emitted).
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

    /// Emit a return statement
    pub fn emit_return(&mut self, reg: &Register, ty: &StableHLOType) {
        self.body
            .push(format!("    return {} : {}", reg.to_mlir(), ty.to_mlir()));
    }

    /// Emit a multi-value return: `return %0, %1 : type0, type1`
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

    /// Emit a function with multiple return values.
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

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name, params_str, ret_str
        )
        .unwrap();

        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function declaration (func.func)
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

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name,
            params_str,
            return_type.to_mlir()
        )
        .unwrap();

        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function call (func.call)
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

    /// Sanitize function name for MLIR (replace dashes with underscores)
    pub(crate) fn sanitize_func_name(name: &str) -> String {
        name.replace('-', "_").replace('?', "_q").replace('!', "_b")
    }

    /// Generate a complete MLIR module with a function body already emitted
    pub fn emit_function_body(&self, name: &str, result_ty: &StableHLOType) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with a function
    pub fn emit_function(&mut self, name: &str, expr: &crate::core::ast::SheafValue) -> String {
        let (result_reg, result_ty) = self.compile_expr(expr);
        self.emit_return(&result_reg, &result_ty);

        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output, "// Source: {}", expr).unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with multiple function declarations
    pub fn emit_module(func_declarations: &[String]) -> String {
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();

        for func_decl in func_declarations {
            write!(output, "{}", func_decl).unwrap();
        }

        writeln!(output, "}}").unwrap();
        output
    }
}
