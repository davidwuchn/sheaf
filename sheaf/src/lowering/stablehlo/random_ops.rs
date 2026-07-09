// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Random number generation operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit random-key: integer seed -> tensor<2xf32>.
    /// Stores lo/hi i32 halves as f32 via bitcast.
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

    /// Emit random-uniform tensor: (random-uniform key [M N])
    /// Generates values in [0, 1) using the same i32 hash as random-normal.
    /// Key is tensor<2xf32> (opaque, stores i32 via bitcast).
    pub fn emit_random_uniform(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        shape: &[i64],
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

        // Create i32 iota [0, 1, 2, ..., N-1]
        let iota = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            iota.to_mlir(), n_i32_ty.to_mlir(),
        ));

        // Broadcast lo and hi to tensor<N x i32>
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

        // Hash round 1: x = (iota + lo) * c1 + hi
        let c1 = self.emit_constant_i32(1540483477); // murmurhash constant
        let c1_bc = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
            c1_bc.to_mlir(), c1.to_mlir(), scalar_i32.to_mlir(), n_i32_ty.to_mlir(),
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

        // Hash round 2: x = x * c2 + c3
        let c2 = self.emit_constant_i32(668265263); // golden ratio * 2^30
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

        // Reshape to target shape
        let result_ty = StableHLOType::f32_tensor(shape.to_vec());
        if shape.len() == 1 && shape[0] == total {
            (uniform, result_ty)
        } else {
            let result = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result.to_mlir(), uniform.to_mlir(),
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
    /// - Hash key to uniform float u in [0, 1)
    /// - Compute cumsum via tril(ones(K,K)) @ probs
    /// - mask = (cumsum >= u), count = sum(mask), index = K - count
    pub fn emit_choice(
        &mut self,
        key: &Register,
        key_ty: &StableHLOType,
        probs: &Register,
        probs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let k = probs_ty.shape()[0];

        // Hash key to uniform float
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

        // Compute cumulative sum via tril(ones(K,K)) @ probs
        let (ones_kk, ones_kk_ty) = self.emit_ones(&[k, k]);
        let (tril_kk, tril_kk_ty) = self.emit_tril(&ones_kk, &ones_kk_ty);
        let (cumsum, cumsum_ty) = self.emit_matmul(&tril_kk, probs, &tril_kk_ty, probs_ty);

        // mask = (cumsum >= u), broadcast u to match cumsum shape
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
