// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Constant emission: scalar and tensor constants for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_constant_f32(&mut self, value: f64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::scalar_f32();
        let value_str = if value.fract() == 0.0 && value.is_finite() {
            format!("{:.1}", value)
        } else {
            format!("{}", value)
        };
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir()
        ));
        self.known_scalars.insert(reg, value);
        reg
    }

    pub fn emit_constant_i32(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::typed_tensor(vec![], "i32");
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        self.known_scalars.insert(reg, value as f64);
        reg
    }

    pub fn emit_constant_i64(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::ScalarI64;
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        reg
    }

    pub fn emit_tensor_constant(&mut self, values: &[Vec<f64>]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let shape = vec![rows as i64, cols as i64];
        let ty = StableHLOType::f32_tensor(shape);

        let rows_str: Vec<String> = values
            .iter()
            .map(|row| {
                let row_values: Vec<String> = row
                    .iter()
                    .map(|&v| {
                        if v.fract() == 0.0 && v.is_finite() {
                            format!("{:.1}", v)
                        } else {
                            format!("{}", v)
                        }
                    })
                    .collect();
                format!("[{}]", row_values.join(", "))
            })
            .collect();

        let values_str = rows_str.join(", ");

        self.body.push(format!(
            "    {} = stablehlo.constant dense<[{}]> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    pub fn emit_nd_tensor_constant(
        &mut self,
        data: &[f64],
        shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        let values_str = if shape.is_empty() {
            format_f64(data[0])
        } else {
            format_dense_attr(data, shape, 0)
        };

        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }
}

pub(super) fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

pub(super) fn format_dense_attr(data: &[f64], shape: &[i64], dim: usize) -> String {
    if dim == shape.len() - 1 {
        let n = shape[dim] as usize;
        let vals: Vec<String> = data[..n].iter().map(|&v| format_f64(v)).collect();
        format!("[{}]", vals.join(", "))
    } else {
        let stride: usize = shape[dim + 1..].iter().map(|&d| d as usize).product();
        let n = shape[dim] as usize;
        let subs: Vec<String> = (0..n)
            .map(|i| format_dense_attr(&data[i * stride..], shape, dim + 1))
            .collect();
        format!("[{}]", subs.join(", "))
    }
}
