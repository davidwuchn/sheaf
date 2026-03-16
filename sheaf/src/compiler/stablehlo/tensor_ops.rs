// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor creation and manipulation operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

/// Compute the broadcast result shape from up to 3 input shapes.
/// Follows numpy broadcasting rules: align from the right, each dim is max.
fn broadcast_result_shape(a: &[i64], b: &[i64], c: &[i64]) -> Vec<i64> {
    let ab = broadcast_two(a, b);
    broadcast_two(&ab, c)
}

fn broadcast_two(a: &[i64], b: &[i64]) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = vec![1i64; len];
    for i in 0..len {
        let da = if i < len - a.len() { 1 } else { a[i - (len - a.len())] };
        let db = if i < len - b.len() { 1 } else { b[i - (len - b.len())] };
        result[i] = da.max(db);
    }
    result
}

impl StableHLOEmitter {
    /// Emit zeros tensor: (zeros [M N]) -> tensor<MxNxf32>
    pub fn emit_zeros(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit random-key: integer seed -> tensor<2xf32> (opaque key).
    /// Stores lo/hi i32 halves as f32 via bitcast, consistent with emit_random_split.
    pub fn emit_random_key(&mut self, seed: i64) -> (Register, StableHLOType) {
        let lo_val = (seed as u64) & 0xFFFFFFFF;
        let hi_val = ((seed as u64) >> 32) & 0xFFFFFFFF;

        let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
        let one_i32 = StableHLOType::i32_tensor(vec![1]);
        let two_i32 = StableHLOType::i32_tensor(vec![2]);

        // Create i32 constants for lo and hi halves
        let lo = self.emit_constant_i32(lo_val as i64);
        let hi = self.emit_constant_i32(hi_val as i64);

        // Reshape scalars to tensor<1xi32>
        let lo_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            lo_1d.to_mlir(), lo.to_mlir(), scalar_i32.to_mlir(), one_i32.to_mlir(),
        ));
        let hi_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            hi_1d.to_mlir(), hi.to_mlir(), scalar_i32.to_mlir(), one_i32.to_mlir(),
        ));

        // Concatenate to tensor<2xi32>
        let key_i32 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.concatenate {}, {}, dim = 0 : ({}, {}) -> {}",
            key_i32.to_mlir(), lo_1d.to_mlir(), hi_1d.to_mlir(),
            one_i32.to_mlir(), one_i32.to_mlir(), two_i32.to_mlir(),
        ));

        // Bitcast i32 -> f32 (opaque key representation)
        self.emit_bitcast_convert(&key_i32, &two_i32, "f32")
    }

    /// Emit random-normal tensor: (random-normal key [M N])
    /// Uses i32 hash + Box-Muller transform to generate normal distribution.
    /// Key is tensor<2xf32> (opaque, stores i32 via bitcast).
    pub fn emit_random_normal(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        shape: &[i64],
    ) -> (Register, StableHLOType) {
        let total: i64 = shape.iter().product();
        // Box-Muller needs pairs: generate 2*total uniform samples
        let n2 = total * 2;
        let n2_i32_ty = StableHLOType::i32_tensor(vec![n2]);
        let n2_f32_ty = StableHLOType::f32_tensor(vec![n2]);
        let n_f32_ty = StableHLOType::f32_tensor(vec![total]);
        let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };

        // Bitcast key to i32, extract lo and hi
        let i32_2_ty = StableHLOType::i32_tensor(vec![2]);
        let (key_i32, _) = self.emit_bitcast_convert(key, key_ty, "i32");

        let lo_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [0:1:1] : ({}) -> tensor<1xi32>",
            lo_1d.to_mlir(), key_i32.to_mlir(), i32_2_ty.to_mlir(),
        ));
        let lo = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            lo.to_mlir(), lo_1d.to_mlir(), scalar_i32.to_mlir(),
        ));
        let hi_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [1:2:1] : ({}) -> tensor<1xi32>",
            hi_1d.to_mlir(), key_i32.to_mlir(), i32_2_ty.to_mlir(),
        ));
        let hi = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            hi.to_mlir(), hi_1d.to_mlir(), scalar_i32.to_mlir(),
        ));

        // Create i32 iota [0, 1, 2, ..., 2N-1]
        let iota = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            iota.to_mlir(), n2_i32_ty.to_mlir(),
        ));

        // Broadcast lo and hi to tensor<2N x i32>
        let lo_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            lo_bc.to_mlir(), lo.to_mlir(), scalar_i32.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let hi_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            hi_bc.to_mlir(), hi.to_mlir(), scalar_i32.to_mlir(), n2_i32_ty.to_mlir(),
        ));

        // Hash round 1: x = (iota + lo) * c1 + hi
        let c1 = self.emit_constant_i32(1540483477); // murmurhash constant
        let c1_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c1_bc.to_mlir(), c1.to_mlir(), scalar_i32.to_mlir(), n2_i32_ty.to_mlir(),
        ));

        let x0 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x0.to_mlir(), iota.to_mlir(), lo_bc.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let x1 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            x1.to_mlir(), x0.to_mlir(), c1_bc.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let x2 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x2.to_mlir(), x1.to_mlir(), hi_bc.to_mlir(), n2_i32_ty.to_mlir(),
        ));

        // Hash round 2: x = x * c2 + c3
        let c2 = self.emit_constant_i32(668265263); // golden ratio * 2^30
        let c2_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c2_bc.to_mlir(), c2.to_mlir(), scalar_i32.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let c3 = self.emit_constant_i32(1013904223);
        let c3_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c3_bc.to_mlir(), c3.to_mlir(), scalar_i32.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let x3 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            x3.to_mlir(), x2.to_mlir(), c2_bc.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let x4 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x4.to_mlir(), x3.to_mlir(), c3_bc.to_mlir(), n2_i32_ty.to_mlir(),
        ));

        // Convert to uniform [0,1): abs(x) as f32 / 2^31
        let abs_x = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.abs {} : {}",
            abs_x.to_mlir(), x4.to_mlir(), n2_i32_ty.to_mlir(),
        ));
        let float_x = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.convert {} : ({}) -> {}",
            float_x.to_mlir(), abs_x.to_mlir(), n2_i32_ty.to_mlir(), n2_f32_ty.to_mlir(),
        ));
        let max_int = self.emit_constant_f32(2147483648.0);
        let max_int_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            max_int_bc.to_mlir(), max_int.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n2_f32_ty.to_mlir(),
        ));
        let uniform = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.divide {}, {} : {}",
            uniform.to_mlir(), float_x.to_mlir(), max_int_bc.to_mlir(), n2_f32_ty.to_mlir(),
        ));

        // Split into u1 [0:N] and u2 [N:2N]
        let u1 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [0:{}:1] : ({}) -> {}",
            u1.to_mlir(), uniform.to_mlir(), total,
            n2_f32_ty.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let u2 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}:{}:1] : ({}) -> {}",
            u2.to_mlir(), uniform.to_mlir(), total, n2,
            n2_f32_ty.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // Box-Muller: result = sqrt(-2 * log(u1 + eps)) * cos(2*pi*u2)
        let eps = self.emit_constant_f32(1e-7);
        let eps_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            eps_bc.to_mlir(), eps.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let u1_safe = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            u1_safe.to_mlir(), u1.to_mlir(), eps_bc.to_mlir(), n_f32_ty.to_mlir(),
        ));

        let log_u1 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.log {} : {}",
            log_u1.to_mlir(), u1_safe.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let neg2 = self.emit_constant_f32(-2.0);
        let neg2_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            neg2_bc.to_mlir(), neg2.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let neg2log = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            neg2log.to_mlir(), neg2_bc.to_mlir(), log_u1.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let radius = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.sqrt {} : {}",
            radius.to_mlir(), neg2log.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // theta = 2*pi*u2
        let two_pi = self.emit_constant_f32(std::f64::consts::TAU);
        let two_pi_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            two_pi_bc.to_mlir(), two_pi.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let theta = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            theta.to_mlir(), two_pi_bc.to_mlir(), u2.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let cos_theta = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.cosine {} : {}",
            cos_theta.to_mlir(), theta.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // result = radius * cos(theta)
        let flat_result = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            flat_result.to_mlir(), radius.to_mlir(), cos_theta.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // Reshape to target shape
        let result_ty = StableHLOType::f32_tensor(shape.to_vec());
        if shape.len() == 1 && shape[0] == total {
            (flat_result, result_ty)
        } else {
            let result = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result.to_mlir(), flat_result.to_mlir(),
                n_f32_ty.to_mlir(), result_ty.to_mlir(),
            ));
            (result, result_ty)
        }
    }

    /// Emit random-randint: (random-randint key [M N] low high)
    /// Generates integer values in [low, high) stored as f32.
    /// Uses i32 hash to produce uniform values, then scales to range.
    pub fn emit_random_randint(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        shape: &[i64],
        low: i64,
        high: i64,
    ) -> (Register, StableHLOType) {
        let total: i64 = shape.iter().product();
        let n_i32_ty = StableHLOType::i32_tensor(vec![total]);
        let n_f32_ty = StableHLOType::f32_tensor(vec![total]);
        let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };

        // Bitcast key to i32, extract lo and hi
        let i32_2_ty = StableHLOType::i32_tensor(vec![2]);
        let (key_i32, _) = self.emit_bitcast_convert(key, key_ty, "i32");

        let lo_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [0:1:1] : ({}) -> tensor<1xi32>",
            lo_1d.to_mlir(), key_i32.to_mlir(), i32_2_ty.to_mlir(),
        ));
        let lo = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            lo.to_mlir(), lo_1d.to_mlir(), scalar_i32.to_mlir(),
        ));
        let hi_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [1:2:1] : ({}) -> tensor<1xi32>",
            hi_1d.to_mlir(), key_i32.to_mlir(), i32_2_ty.to_mlir(),
        ));
        let hi = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            hi.to_mlir(), hi_1d.to_mlir(), scalar_i32.to_mlir(),
        ));

        // Create i32 iota [0, 1, ..., N-1]
        let iota = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            iota.to_mlir(), n_i32_ty.to_mlir(),
        ));

        // Broadcast lo and hi
        let lo_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            lo_bc.to_mlir(), lo.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let hi_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            hi_bc.to_mlir(), hi.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
        ));

        // Hash: x = ((iota + lo) * c1 + hi) * c2 + c3
        let c1 = self.emit_constant_i32(1540483477);
        let c1_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c1_bc.to_mlir(), c1.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let c2 = self.emit_constant_i32(668265263);
        let c2_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c2_bc.to_mlir(), c2.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let c3 = self.emit_constant_i32(1013904223);
        let c3_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c3_bc.to_mlir(), c3.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
        ));

        let x0 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x0.to_mlir(), iota.to_mlir(), lo_bc.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let x1 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            x1.to_mlir(), x0.to_mlir(), c1_bc.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let x2 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x2.to_mlir(), x1.to_mlir(), hi_bc.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let x3 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            x3.to_mlir(), x2.to_mlir(), c2_bc.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let x4 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            x4.to_mlir(), x3.to_mlir(), c3_bc.to_mlir(), n_i32_ty.to_mlir(),
        ));

        // Convert to uniform [0,1): abs(x) as f32 / 2^31
        let abs_x = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.abs {} : {}",
            abs_x.to_mlir(), x4.to_mlir(), n_i32_ty.to_mlir(),
        ));
        let float_x = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.convert {} : ({}) -> {}",
            float_x.to_mlir(), abs_x.to_mlir(), n_i32_ty.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let max_int = self.emit_constant_f32(2147483648.0);
        let max_int_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            max_int_bc.to_mlir(), max_int.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let uniform = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.divide {}, {} : {}",
            uniform.to_mlir(), float_x.to_mlir(), max_int_bc.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // Scale to [low, high): floor(uniform * range) + low
        let range_val = (high - low) as f64;
        let range_c = self.emit_constant_f32(range_val);
        let range_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            range_bc.to_mlir(), range_c.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let scaled = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            scaled.to_mlir(), uniform.to_mlir(), range_bc.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let floored = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.floor {} : {}",
            floored.to_mlir(), scaled.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let low_c = self.emit_constant_f32(low as f64);
        let low_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            low_bc.to_mlir(), low_c.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let flat_result = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.add {}, {} : {}",
            flat_result.to_mlir(), floored.to_mlir(), low_bc.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // Clamp to [low, high-1]
        let high_m1_c = self.emit_constant_f32((high - 1) as f64);
        let high_m1_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            high_m1_bc.to_mlir(), high_m1_c.to_mlir(),
            StableHLOType::ScalarF32.to_mlir(), n_f32_ty.to_mlir(),
        ));
        let clamped = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.clamp {}, {}, {} : {}",
            clamped.to_mlir(), low_bc.to_mlir(), flat_result.to_mlir(),
            high_m1_bc.to_mlir(), n_f32_ty.to_mlir(),
        ));

        // Reshape to target shape
        let result_ty = StableHLOType::f32_tensor(shape.to_vec());
        if shape.len() == 1 && shape[0] == total {
            (clamped, result_ty)
        } else {
            let result = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result.to_mlir(), clamped.to_mlir(),
                n_f32_ty.to_mlir(), result_ty.to_mlir(),
            ));
            (result, result_ty)
        }
    }

    /// Emit ones tensor: (ones [M N]) -> tensor<MxNxf32>
    pub fn emit_ones(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<1.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
    pub fn emit_reshape(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        new_shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        // Preserve dtype from operand
        let dtype = match operand_ty {
            StableHLOType::Tensor { dtype, .. } => dtype.clone(),
            _ => "f32".to_string(),
        };
        let result_ty = StableHLOType::Tensor { shape: new_shape.to_vec(), dtype };

        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit transpose: (transpose tensor [1 0]) -> permutes dimensions
    pub fn emit_transpose(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        permutation: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape by applying permutation
        let operand_shape = operand_ty.shape();

        // Transpose of a scalar or 1D is identity
        if operand_shape.len() <= 1 {
            return (operand.clone(), operand_ty.clone());
        }

        let result_shape: Vec<i64> = permutation
            .iter()
            .map(|&i| operand_shape[i as usize])
            .collect();

        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format permutation as [0, 1, 2]
        let perm_str = permutation
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.transpose {}, dims = [{}] : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            perm_str,
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit iota (arange): (arange N) -> tensor<Nxf32> with values [0, 1, 2, ..., N-1]
    pub fn emit_iota(&mut self, shape: &[i64], dimension: i64) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            reg.to_mlir(),
            dimension,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit concatenate: (concat [tensor1 tensor2 ...] axis)
    pub fn emit_concatenate(
        &mut self,
        operands: &[Register],
        operand_types: &[StableHLOType],
        dimension: i64,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape: same as first operand except for concat dimension
        let first_shape = operand_types[0].shape();
        let mut result_shape = first_shape.to_vec();

        // Sum the sizes along the concatenation dimension
        let concat_dim_size: i64 = operand_types
            .iter()
            .map(|ty| ty.shape()[dimension as usize])
            .sum();

        result_shape[dimension as usize] = concat_dim_size;
        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format operands as %0, %1, %2
        let operands_str = operands
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        // Format types as (tensor<2x3xf32>, tensor<2x3xf32>)
        let types_str = operand_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.concatenate {}, dim = {} : ({}) -> {}",
            reg.to_mlir(),
            operands_str,
            dimension,
            types_str,
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit where (conditional selection): (where condition x y)
    /// Selects elements from x when condition is true, from y when false.
    /// Broadcasts condition and y to match x's shape (NumPy semantics).
    pub fn emit_where(
        &mut self,
        condition: &Register,
        x: &Register,
        y: &Register,
        condition_ty: &StableHLOType,
        x_ty: &StableHLOType,
        y_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Result shape = broadcast of x, y, and condition shapes
        let x_shape = x_ty.shape();
        let y_shape = y_ty.shape();
        let cond_shape = condition_ty.shape();
        let result_shape = broadcast_result_shape(x_shape, y_shape, cond_shape);

        // Broadcast condition in f32 (comparisons return f32 0.0/1.0)
        let (actual_cond, actual_cond_ty) = if cond_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape.clone());
            let r = self.emit_broadcast(condition, condition_ty, &target);
            (r, target)
        } else {
            (condition.clone(), condition_ty.clone())
        };

        // Broadcast x (on_true) to result shape if needed
        let (actual_x, actual_x_ty) = if x_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape.clone());
            let r = self.emit_broadcast(x, x_ty, &target);
            (r, target)
        } else {
            (x.clone(), x_ty.clone())
        };

        // Broadcast y (on_false) to result shape if needed
        let (actual_y, actual_y_ty) = if y_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape);
            let r = self.emit_broadcast(y, y_ty, &target);
            (r, target)
        } else {
            (y.clone(), y_ty.clone())
        };

        self.emit_select(&actual_cond, &actual_x, &actual_y, &actual_cond_ty, &actual_x_ty, &actual_y_ty)
    }

    /// Emit swapaxes: (swapaxes x axis1 axis2)
    /// Interchanges two axes using transpose
    pub fn emit_swapaxes(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        axis1: i64,
        axis2: i64,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let rank = operand_shape.len() as i64;

        // Normalize negative indices
        let a1 = if axis1 < 0 { (rank + axis1) as usize } else { axis1 as usize };
        let a2 = if axis2 < 0 { (rank + axis2) as usize } else { axis2 as usize };

        // Build permutation that swaps axis1 and axis2
        let mut permutation: Vec<i64> = (0..rank).collect();
        permutation[a1] = a2 as i64;
        permutation[a2] = a1 as i64;

        // Use transpose with the permutation
        self.emit_transpose(operand, operand_ty, &permutation)
    }

    /// Emit tril (lower triangular): (tril x)
    /// Returns the lower triangular part of a matrix, zeros above diagonal
    pub fn emit_tril(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = operand_ty.clone();

        let shape = operand_ty.shape();
        if shape.len() != 2 {
            panic!("tril requires a 2D tensor");
        }

        let m = shape[0];
        let n = shape[1];

        // Create row indices: iota dim=0
        let row_iota = self.fresh_register();
        let row_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            row_iota.to_mlir(),
            row_iota_ty.to_mlir()
        ));

        // Create col indices: iota dim=1
        let col_iota = self.fresh_register();
        let col_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 1 : {}",
            col_iota.to_mlir(),
            col_iota_ty.to_mlir()
        ));

        // Compare: row_idx >= col_idx (i >= j for lower triangle)
        // emit_compare returns f32 (0.0/1.0)
        let (mask, _mask_ty) = self.emit_compare(">=", &row_iota, &col_iota, &row_iota_ty, &col_iota_ty);

        // Multiply: operand * mask_f32 (zeros out upper triangle)
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            reg.to_mlir(),
            operand.to_mlir(),
            mask.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit slice along axis 0 at a given index, then reshape to remove that axis.
    /// E.g. tensor<5x9xf32> at index 0 -> tensor<9xf32>
    pub fn emit_index_axis0(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "Cannot index a scalar");

        let ndim = shape.len();

        // Build start_indices and limit_indices
        let mut start = vec![0i64; ndim];
        let mut limit = shape.to_vec();
        let strides = vec![1i64; ndim];
        start[0] = index;
        limit[0] = index + 1;

        let dims_str = format_slice_dims(&start, &limit, &strides);

        // Slice to [1, d1, d2, ...]
        let slice_reg = self.fresh_register();
        let mut slice_shape = shape.to_vec();
        slice_shape[0] = 1;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        // Reshape to remove axis 0: [1, d1, d2, ...] -> [d1, d2, ...]
        let result_shape: Vec<i64> = shape[1..].to_vec();
        if result_shape.is_empty() {
            let result_ty = StableHLOType::scalar_f32();
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        } else {
            let result_ty = StableHLOType::f32_tensor(result_shape);
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        }
    }

    /// Emit slice on the last axis, then squeeze that dimension if size 1.
    /// E.g. tensor<5x9xf32> slice [3:4] on last axis -> tensor<5xf32>
    /// E.g. tensor<5x9xf32> slice [2:5] on last axis -> tensor<5x3xf32>
    pub fn emit_slice_last_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[ndim - 1] = start;
        limit_indices[ndim - 1] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut slice_shape = shape.to_vec();
        slice_shape[ndim - 1] = end - start;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());

        let slice_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        // If slice size is 1 on last axis, squeeze it
        if end - start == 1 {
            let result_shape: Vec<i64> = shape[..ndim - 1].to_vec();
            if result_shape.is_empty() {
                let result_ty = StableHLOType::scalar_f32();
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            } else {
                let result_ty = StableHLOType::f32_tensor(result_shape);
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            }
        } else {
            (slice_reg, slice_ty)
        }
    }

    /// Emit identity matrix: (eye N) or (eye N M)
    /// Strategy: iota(dim=0) == iota(dim=1) -> select(mask, 1.0, 0.0)
    pub fn emit_eye(&mut self, n: i64, m: i64) -> (Register, StableHLOType) {
        let shape = vec![n, m];
        let result_ty = StableHLOType::f32_tensor(shape.clone());

        // Row indices: [0,0,...; 1,1,...; 2,2,...] shape [N,M]
        let (row_iota, _) = self.emit_iota(&shape, 0);
        // Col indices: [0,1,2...; 0,1,2...; ...] shape [N,M]
        let (col_iota, iota_ty) = self.emit_iota(&shape, 1);

        // Compare row == col -> bool mask [N, M]
        let (mask_reg, mask_ty) = self.emit_compare("==", &row_iota, &col_iota, &iota_ty, &iota_ty);

        // ones and zeros tensors
        let one_scalar = self.emit_constant_f32(1.0);
        let zero_scalar = self.emit_constant_f32(0.0);
        let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &result_ty);
        let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &result_ty);

        // select(mask, 1.0, 0.0)
        self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &result_ty, &result_ty)
    }

    /// Emit one-hot encoding: (one-hot indices num_classes)
    /// indices: tensor<Nxf32> (integer values as f32), num_classes: static int
    /// Returns tensor<NxCxf32>
    pub fn emit_one_hot(
        &mut self,
        indices: &Register,
        indices_ty: &StableHLOType,
        num_classes: i64,
    ) -> (Register, StableHLOType) {
        let indices_shape = indices_ty.shape();

        if indices_shape.is_empty() {
            // Scalar index -> output [C]
            let out_shape = vec![num_classes];
            let out_ty = StableHLOType::f32_tensor(out_shape.clone());

            // iota [C] along dim 0
            let (class_iota, iota_ty) = self.emit_iota(&out_shape, 0);

            // Broadcast scalar index to [C]
            let idx_broadcast = self.emit_broadcast(indices, indices_ty, &iota_ty);

            // Compare indices == iota
            let (mask_reg, mask_ty) = self.emit_compare("==", &idx_broadcast, &class_iota, &iota_ty, &iota_ty);

            let one_scalar = self.emit_constant_f32(1.0);
            let zero_scalar = self.emit_constant_f32(0.0);
            let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &out_ty);
            let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &out_ty);

            self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &out_ty, &out_ty)
        } else {
            // Tensor indices [N] -> output [N, C]
            let n = indices_shape[0];
            let out_shape = vec![n, num_classes];
            let out_ty = StableHLOType::f32_tensor(out_shape.clone());

            // iota [N, C] along dim 1 -> class indices
            let (class_iota, iota_ty) = self.emit_iota(&out_shape, 1);

            // Reshape indices [N] -> [N, 1] then broadcast to [N, C]
            let idx_2d_shape = vec![n, 1];
            let idx_2d_ty = StableHLOType::f32_tensor(idx_2d_shape);
            let idx_2d = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                idx_2d.to_mlir(),
                indices.to_mlir(),
                indices_ty.to_mlir(),
                idx_2d_ty.to_mlir(),
            ));
            let idx_broadcast = self.emit_broadcast(&idx_2d, &idx_2d_ty, &iota_ty);

            // Compare indices == iota
            let (mask_reg, mask_ty) = self.emit_compare("==", &idx_broadcast, &class_iota, &iota_ty, &iota_ty);

            let one_scalar = self.emit_constant_f32(1.0);
            let zero_scalar = self.emit_constant_f32(0.0);
            let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &out_ty);
            let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &out_ty);

            self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &out_ty, &out_ty)
        }
    }

    /// Emit slice along axis 0: (dynamic-slice tensor start end)
    /// start is inclusive, end is inclusive (matches interpreter semantics)
    pub fn emit_slice_range(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[0] = start;
        limit_indices[0] = end + 1; // inclusive end

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[0] = end + 1 - start;
        let result_ty = StableHLOType::f32_tensor(result_shape);

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    /// Emit roll (circular shift): (roll tensor shift)
    /// Positive shift moves elements forward (right), wrapping around.
    /// Implemented as: concat(slice[n-shift:], slice[:n-shift])
    pub fn emit_roll(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        shift: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let n = shape[0];
        let shift = ((shift % n) + n) % n; // normalize to [0, n)

        if shift == 0 {
            return (input.clone(), input_ty.clone());
        }

        // Split point: elements from [n-shift..n-1] go first, then [0..n-shift-1]
        let split = n - shift;

        // slice [split:n-1] (inclusive end)
        let (tail, tail_ty) = self.emit_slice_range(input, input_ty, split, n - 1);
        // slice [0:split-1] (inclusive end)
        let (head, _head_ty) = self.emit_slice_range(input, input_ty, 0, split - 1);

        // concat tail + head along axis 0
        self.emit_concatenate(&[tail, head], &[tail_ty.clone(), input_ty.clone()], 0)
    }

    /// Emit index-update: (index-update tensor idx new-value)
    /// Returns a new tensor with tensor[idx] replaced by new-value.
    /// Uses stablehlo.dynamic_update_slice.
    pub fn emit_index_update(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
        value: &Register,
        value_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        // Reshape value to have leading dim of 1 for the update slice
        let mut update_shape = vec![1i64];
        update_shape.extend_from_slice(&value_ty.shape());
        // If value is scalar and tensor is 1D, update_shape = [1]
        if value_ty.shape().is_empty() && ndim == 1 {
            // scalar update into 1D tensor
            let update_ty = StableHLOType::f32_tensor(vec![1]);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            let idx_reg = self.emit_constant_i32(index);
            let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                idx_reg.to_mlir(),
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                scalar_i32.to_mlir(),
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        } else {
            // N-D: value has shape [D1, D2, ...], update slice has shape [1, D1, D2, ...]
            let update_ty = StableHLOType::f32_tensor(update_shape);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            // Start indices: [index, 0, 0, ...]
            let idx_reg = self.emit_constant_i32(index);
            let zero_idx = self.emit_constant_i32(0);
            let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
            let mut start_regs = vec![idx_reg.to_mlir()];
            for _ in 1..ndim {
                start_regs.push(zero_idx.to_mlir());
            }
            let start_str = start_regs.join(", ");

            let mut start_types = vec![scalar_i32.to_mlir()];
            for _ in 1..ndim {
                start_types.push(scalar_i32.to_mlir());
            }
            let start_types_str = start_types.join(", ");

            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                start_str,
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                start_types_str,
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        }
    }

    /// Emit slice along axis 0 with exclusive end: (slice tensor start end)
    /// start inclusive, end exclusive: matches standard Python/NumPy semantics
    pub fn emit_slice_exclusive(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        self.emit_slice_axis(input, input_ty, start, end, 0)
    }

    /// Emit slice along a given axis: (slice tensor start end :axis N)
    pub fn emit_slice_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
        axis: usize,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[axis] = start;
        limit_indices[axis] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[axis] = end - start;
        let dtype = match input_ty {
            StableHLOType::Tensor { dtype, .. } => dtype.clone(),
            _ => "f32".to_string(),
        };
        let result_ty = StableHLOType::Tensor { shape: result_shape, dtype };

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    /// Emit tensor-split: split tensor into N equal sections along axis 0
    /// Returns a tuple of N tensors
    pub fn emit_tensor_split(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        num_sections: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let total = shape[0];
        let section_size = total / num_sections;

        let mut section_regs = Vec::new();
        let mut section_types = Vec::new();

        for i in 0..num_sections {
            let start = i * section_size;
            let end = start + section_size;
            let (reg, ty) = self.emit_slice_exclusive(input, input_ty, start, end);
            section_regs.push(reg);
            section_types.push(ty);
        }

        // Pack into a tuple
        self.emit_tuple(&section_regs, &section_types)
    }

    /// Emit gather along axis 0: (get operand indices) where indices is a tensor.
    /// operand shape [N, D1, D2, ...], indices shape [I1, I2, ...]
    /// result shape [I1, I2, ..., D1, D2, ...]
    pub fn emit_gather_axis0(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        indices: &Register,
        indices_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let indices_shape = indices_ty.shape();

        // Convert indices to i32 (Sheaf tensors are f32; i32 avoids SPIRV crash)
        let indices_int_ty = StableHLOType::i32_tensor(indices_shape.to_vec());
        let indices_int_reg = self.emit_convert(indices, indices_ty, &indices_int_ty);

        // Reshape indices to add trailing index_vector_dim: [I1, I2, ...] -> [I1, I2, ..., 1]
        let mut reshaped_shape: Vec<i64> = indices_shape.to_vec();
        reshaped_shape.push(1);
        let indices_3d_reg = self.fresh_register();
        let indices_3d_ty = StableHLOType::i32_tensor(reshaped_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            indices_3d_reg.to_mlir(),
            indices_int_reg.to_mlir(),
            indices_int_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
        ));

        // Compute result shape: indices_shape + operand_shape[1:]
        let row_shape = &operand_shape[1..];
        let mut result_shape: Vec<i64> = indices_shape.to_vec();
        result_shape.extend_from_slice(row_shape);
        let result_ty = StableHLOType::f32_tensor(result_shape);

        // offset_dims = [rank(indices), rank(indices)+1, ..., rank(result)-1]
        let idx_rank = indices_shape.len();
        let offset_dims: Vec<i64> = (idx_rank..idx_rank + row_shape.len())
            .map(|d| d as i64)
            .collect();

        // slice_sizes = [1, D1, D2, ...]
        let mut slice_sizes: Vec<i64> = vec![1];
        slice_sizes.extend_from_slice(row_shape);

        let index_vector_dim = reshaped_shape.len() - 1;

        let result_reg = self.fresh_register();
        let offset_dims_str = offset_dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        let slice_sizes_str = slice_sizes.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");

        self.body.push(format!(
            "    {} = \"stablehlo.gather\"({}, {}) {{\n\
             \x20     dimension_numbers = #stablehlo.gather<\n\
             \x20       offset_dims = [{}],\n\
             \x20       collapsed_slice_dims = [0],\n\
             \x20       operand_batching_dims = [],\n\
             \x20       start_indices_batching_dims = [],\n\
             \x20       start_index_map = [0],\n\
             \x20       index_vector_dim = {}>,\n\
             \x20     slice_sizes = array<i64: {}>,\n\
             \x20     indices_are_sorted = false\n\
             \x20   }} : ({}, {}) -> {}",
            result_reg.to_mlir(),
            operand.to_mlir(),
            indices_3d_reg.to_mlir(),
            offset_dims_str,
            index_vector_dim,
            slice_sizes_str,
            operand_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        (result_reg, result_ty)
    }

    /// Emit top_k: sort descending + slice first K elements.
    /// Returns (values: tensor<Kxf32>, indices: tensor<Kxf32>).
    pub fn emit_top_k(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        k: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "top_k requires at least 1D input");
        let last_axis = shape.len() - 1;

        // Create iota indices [0, 1, ..., N-1] as i32 (required by chlo.top_k pattern)
        let iota_reg = self.fresh_register();
        let iota_ty = StableHLOType::i32_tensor(shape.to_vec());
        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            iota_reg.to_mlir(),
            last_axis,
            iota_ty.to_mlir()
        ));

        // Sort descending along last axis: returns (sorted_values, sorted_indices)
        let sorted_vals = self.fresh_register();
        let sorted_idxs = self.fresh_register();

        // Use fresh registers for comparator block args to avoid name collisions
        let cmp_lhs = self.fresh_register();
        let cmp_rhs = self.fresh_register();
        let cmp_lhs_idx = self.fresh_register();
        let cmp_rhs_idx = self.fresh_register();
        let cmp_pred = self.fresh_register();

        self.body.push(format!(
            "    {}, {} = \"stablehlo.sort\"({}, {}) ({{\n\
             \x20   ^bb0({}: tensor<f32>, {}: tensor<f32>, {}: tensor<i32>, {}: tensor<i32>):\n\
             \x20     {} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction GT>}} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n\
             \x20     \"stablehlo.return\"({}) : (tensor<i1>) -> ()\n\
             \x20   }}) {{dimension = {} : i64, is_stable = true}} : ({}, {}) -> ({}, {})",
            sorted_vals.to_mlir(),
            sorted_idxs.to_mlir(),
            input.to_mlir(),
            iota_reg.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_lhs_idx.to_mlir(),
            cmp_rhs_idx.to_mlir(),
            cmp_pred.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_pred.to_mlir(),
            last_axis,
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
        ));

        // Slice first K elements along last axis
        let (top_vals, top_vals_ty) = self.emit_slice_axis(
            &sorted_vals, input_ty, 0, k, last_axis,
        );
        let (top_idxs_i32, top_idxs_i32_ty) = self.emit_slice_axis(
            &sorted_idxs, &iota_ty, 0, k, last_axis,
        );

        // Convert indices i32 -> f32 for f32-only codegen consistency
        let top_idxs_f32_ty = StableHLOType::f32_tensor(top_idxs_i32_ty.shape().to_vec());
        let top_idxs = self.emit_convert(&top_idxs_i32, &top_idxs_i32_ty, &top_idxs_f32_ty);

        // Pack into virtual tuple
        self.emit_tuple(
            &[top_vals, top_idxs],
            &[top_vals_ty, top_idxs_f32_ty],
        )
    }

    /// Emit bitcast_convert: reinterpret bits between f32 and i32.
    pub fn emit_bitcast_convert(
        &mut self,
        reg: &Register,
        from_ty: &StableHLOType,
        to_dtype: &str,
    ) -> (Register, StableHLOType) {
        let result_reg = self.fresh_register();
        let result_ty = match from_ty {
            StableHLOType::ScalarF32 => StableHLOType::Tensor { shape: vec![], dtype: to_dtype.to_string() },
            StableHLOType::Tensor { shape, .. } => StableHLOType::Tensor { shape: shape.clone(), dtype: to_dtype.to_string() },
            _ => panic!("bitcast_convert: unsupported type {:?}", from_ty),
        };
        self.body.push(format!(
            "    {} = stablehlo.bitcast_convert {} : ({}) -> {}",
            result_reg.to_mlir(),
            reg.to_mlir(),
            from_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    /// Emit random-split: deterministic key splitting using i32 hash.
    /// Key is tensor<2xf32> (stores 2 i32 values via bitcast).
    /// Returns tuple of N sub-keys (default N=2).
    pub fn emit_random_split(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        self.emit_random_split_n(key, key_ty, 2)
    }

    /// Emit random-split with N outputs: deterministic key splitting.
    /// For each i in 0..N, derives a child key via i32 hash arithmetic.
    /// Returns tuple<tensor<2xf32>, ...> (N sub-keys).
    pub fn emit_random_split_n(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        n: usize,
    ) -> (Register, StableHLOType) {
        let i32_ty = StableHLOType::i32_tensor(vec![2]);
        let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
        let one_i32 = StableHLOType::i32_tensor(vec![1]);

        // Bitcast f32 key to i32 for integer arithmetic
        let (key_i32, _) = self.emit_bitcast_convert(key, key_ty, "i32");

        // Extract lo and hi as scalar i32
        let lo_slice = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [0:1:1] : ({}) -> tensor<1xi32>",
            lo_slice.to_mlir(), key_i32.to_mlir(), i32_ty.to_mlir(),
        ));
        let lo_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            lo_scalar.to_mlir(), lo_slice.to_mlir(), scalar_i32.to_mlir(),
        ));
        let hi_slice = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [1:2:1] : ({}) -> tensor<1xi32>",
            hi_slice.to_mlir(), key_i32.to_mlir(), i32_ty.to_mlir(),
        ));
        let hi_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            hi_scalar.to_mlir(), hi_slice.to_mlir(), scalar_i32.to_mlir(),
        ));

        // Splitmix-like constants (mod 2^32)
        // 6364136223846793005 mod 2^32 = 2069105475
        // 1442695040888963407 mod 2^32 = 1013904223
        let c_mult = self.emit_constant_i32(2069105475);
        let c_add = self.emit_constant_i32(1013904223);

        // Generate N sub-keys by unrolling the hash loop
        let mut key_regs = Vec::with_capacity(n);
        let mut key_types = Vec::with_capacity(n);

        for i in 0..n {
            // offset_lo = lo + i
            let offset_lo = if i == 0 {
                lo_scalar
            } else {
                let c_i = self.emit_constant_i32(i as i64);
                let r = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.add {}, {} : {}",
                    r.to_mlir(), lo_scalar.to_mlir(), c_i.to_mlir(), scalar_i32.to_mlir(),
                ));
                r
            };

            // new_lo = offset_lo * c_mult + c_add
            let mul_r = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.multiply {}, {} : {}",
                mul_r.to_mlir(), offset_lo.to_mlir(), c_mult.to_mlir(), scalar_i32.to_mlir(),
            ));
            let new_lo = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.add {}, {} : {}",
                new_lo.to_mlir(), mul_r.to_mlir(), c_add.to_mlir(), scalar_i32.to_mlir(),
            ));

            // new_hi = hi + offset_lo
            let new_hi = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.add {}, {} : {}",
                new_hi.to_mlir(), hi_scalar.to_mlir(), offset_lo.to_mlir(), scalar_i32.to_mlir(),
            ));

            // Reshape scalars to tensor<1xi32> then concatenate to tensor<2xi32>
            let lo_1d = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                lo_1d.to_mlir(), new_lo.to_mlir(), scalar_i32.to_mlir(), one_i32.to_mlir(),
            ));
            let hi_1d = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                hi_1d.to_mlir(), new_hi.to_mlir(), scalar_i32.to_mlir(), one_i32.to_mlir(),
            ));
            let key_i32_r = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.concatenate {}, {}, dim = 0 : ({}, {}) -> {}",
                key_i32_r.to_mlir(), lo_1d.to_mlir(), hi_1d.to_mlir(),
                one_i32.to_mlir(), one_i32.to_mlir(), i32_ty.to_mlir(),
            ));

            // Bitcast back to f32
            let (key_f32, key_f32_ty) = self.emit_bitcast_convert(&key_i32_r, &i32_ty, "f32");
            key_regs.push(key_f32);
            key_types.push(key_f32_ty);
        }

        self.emit_tuple(&key_regs, &key_types)
    }

    /// Emit choice: categorical sampling from probabilities.
    /// key: tensor<2xf32>, probs: tensor<Kxf32>
    /// Returns scalar f32 (selected index).
    ///
    /// Algorithm:
    /// 1. Hash key to uniform float u in [0, 1)
    /// 2. Compute cumsum via tril(ones(K,K)) @ probs
    /// 3. mask = (cumsum >= u), count = sum(mask), index = K - count
    pub fn emit_choice(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        probs: &Register,
        probs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let k = probs_ty.shape()[0];

        // Step 1: Hash key to uniform float
        // Bitcast to i32, take lo, abs, convert to f32, divide by 2^31
        let i32_2_ty = StableHLOType::i32_tensor(vec![2]);
        let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
        let (key_i32, _) = self.emit_bitcast_convert(key, key_ty, "i32");

        let lo_1d = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [0:1:1] : ({}) -> tensor<1xi32>",
            lo_1d.to_mlir(), key_i32.to_mlir(), i32_2_ty.to_mlir(),
        ));
        let lo = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : (tensor<1xi32>) -> {}",
            lo.to_mlir(), lo_1d.to_mlir(), scalar_i32.to_mlir(),
        ));
        let abs_lo = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.abs {} : {}",
            abs_lo.to_mlir(), lo.to_mlir(), scalar_i32.to_mlir(),
        ));
        // Convert i32 to f32
        let u_raw = self.fresh_register();
        let scalar_f32_ty = StableHLOType::ScalarF32;
        self.body.push(format!(
            "    {} = stablehlo.convert {} : ({}) -> {}",
            u_raw.to_mlir(), abs_lo.to_mlir(), scalar_i32.to_mlir(), scalar_f32_ty.to_mlir(),
        ));
        // Divide by 2^31 to get [0, 1)
        let max_i31 = self.emit_constant_f32(2147483648.0);
        let u = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.divide {}, {} : {}",
            u.to_mlir(), u_raw.to_mlir(), max_i31.to_mlir(), scalar_f32_ty.to_mlir(),
        ));

        // Step 2: Compute cumulative sum via tril(ones(K,K)) @ probs
        let (ones_kk, ones_kk_ty) = self.emit_ones(&[k, k]);
        let (tril_kk, tril_kk_ty) = self.emit_tril(&ones_kk, &ones_kk_ty);
        let (cumsum, cumsum_ty) = self.emit_matmul(&tril_kk, probs, &tril_kk_ty, probs_ty);

        // Step 3: mask = (cumsum >= u), broadcast u to match cumsum shape
        let u_bc = self.emit_broadcast(&u, &scalar_f32_ty, &cumsum_ty);
        let (mask, _mask_ty) = self.emit_compare(">=", &cumsum, &u_bc, &cumsum_ty, &cumsum_ty);

        // count = sum(mask), index = K - count
        let (count, _) = self.emit_reduce_sum(&mask, &cumsum_ty, 0, false);
        let k_const = self.emit_constant_f32(k as f64);
        let index = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.subtract {}, {} : {}",
            index.to_mlir(), k_const.to_mlir(), count.to_mlir(), scalar_f32_ty.to_mlir(),
        ));

        // Clamp to [0, K-1]
        let zero = self.emit_constant_f32(0.0);
        let k_minus_1 = self.emit_constant_f32((k - 1) as f64);
        let clamped = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.clamp {}, {}, {} : {}",
            clamped.to_mlir(), zero.to_mlir(), index.to_mlir(), k_minus_1.to_mlir(),
            scalar_f32_ty.to_mlir(),
        ));

        (clamped, scalar_f32_ty)
    }
}

/// Format slice dimensions as `start:limit:stride, ...` for StableHLO assembly.
fn format_slice_dims(starts: &[i64], limits: &[i64], strides: &[i64]) -> String {
    starts
        .iter()
        .zip(limits.iter())
        .zip(strides.iter())
        .map(|((s, l), st)| format!("{}:{}:{}", s, l, st))
        .collect::<Vec<_>>()
        .join(", ")
}
