// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Dot product and einsum operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit a matrix multiply (dot_general)
    /// Follows NumPy @ semantics:
    ///   [K] @ [K]               -> scalar   (dot product)
    ///   [K] @ [K, N]            -> [N]      (vec-mat)
    ///   [M, K] @ [K]            -> [M]      (mat-vec)
    ///   [M, K] @ [K, N]         -> [M, N]   (matmul)
    ///   [..., M, K] @ [K, N]    -> [..., M, N]  (batched, rhs broadcast)
    ///   [..., M, K] @ [..., K, N] -> [..., M, N] (batched)
    pub fn emit_matmul(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();

        // Scalar operand: broadcast multiply instead of matmul
        if lhs_rank == 0 || rhs_rank == 0 {
            return self.emit_binop("*", lhs, rhs, lhs_ty, rhs_ty);
        }

        // Both 1D: dot product [K] @ [K] -> scalar
        if lhs_rank == 1 && rhs_rank == 1 {
            let result_ty = StableHLOType::f32_tensor(vec![]);
            let reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, contracting_dims = [0] x [0] : ({}, {}) -> {}",
                reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
                lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
            ));
            return (reg, result_ty);
        }

        // 1D lhs: [K] @ [K, N] -> [N] (vec-mat)
        if lhs_rank == 1 && rhs_rank >= 2 {
            let mut result_shape: Vec<i64> = rhs_shape[..rhs_rank - 2].to_vec();
            result_shape.extend_from_slice(&rhs_shape[rhs_rank - 1..]);
            let result_ty = StableHLOType::f32_tensor(result_shape);
            let reg = self.fresh_register();
            let rhs_contract = (rhs_rank as i64) - 2;
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, contracting_dims = [0] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
                rhs_contract,
                lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
            ));
            return (reg, result_ty);
        }

        // 1D rhs: [M, K] @ [K] -> [M] (mat-vec), or [..., M, K] @ [K] -> [..., M]
        if rhs_rank == 1 {
            let result_shape: Vec<i64> = lhs_shape[..lhs_rank - 1].to_vec();
            let result_ty = StableHLOType::f32_tensor(result_shape);
            let reg = self.fresh_register();
            let lhs_contract = lhs_rank as i64 - 1;
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [0] : ({}, {}) -> {}",
                reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
                lhs_contract,
                lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
            ));
            return (reg, result_ty);
        }

        // General case: both rank >= 2
        let n_batch = if rhs_rank <= 2 {
            0
        } else {
            lhs_rank.min(rhs_rank).saturating_sub(2)
        };
        let lhs_contract = lhs_rank as i64 - 1;
        let rhs_contract = n_batch as i64;

        let mut result_shape: Vec<i64> = lhs_shape[..n_batch].to_vec();
        result_shape.extend_from_slice(&lhs_shape[n_batch..lhs_rank - 1]);
        result_shape.extend_from_slice(&rhs_shape[n_batch + 1..]);

        let result_ty = StableHLOType::f32_tensor(result_shape);
        let reg = self.fresh_register();

        if n_batch == 0 {
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
                lhs_contract, rhs_contract,
                lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
            ));
        } else {
            let batch_dims: Vec<String> = (0..n_batch).map(|i| i.to_string()).collect();
            let batch_str = batch_dims.join(", ");
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, batching_dims = [{}] x [{}], contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
                batch_str, batch_str,
                lhs_contract, rhs_contract,
                lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
            ));
        }

        (reg, result_ty)
    }

    /// Emit einsum via dot_general + optional transpose.
    /// Parses Einstein summation notation (e.g. "...nd,d->...n") and lowers to
    /// stablehlo.dot_general with computed batching/contracting dimensions.
    pub fn emit_einsum(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
        spec: &str,
    ) -> Result<(Register, StableHLOType), String> {
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();

        // Parse spec: "lhs_labels,rhs_labels->out_labels"
        let spec_clean: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
        let arrow_pos = spec_clean
            .find("->")
            .ok_or_else(|| format!("einsum: missing '->' in spec '{}'", spec))?;
        let inputs_str = &spec_clean[..arrow_pos];
        let out_str = &spec_clean[arrow_pos + 2..];
        let comma_pos = inputs_str
            .find(',')
            .ok_or_else(|| format!("einsum: missing ',' in spec '{}'", spec))?;
        let lhs_str = &inputs_str[..comma_pos];
        let rhs_str = &inputs_str[comma_pos + 1..];

        // Parse labels, expanding ellipsis to concrete batch dims
        let parse_labels = |s: &str, shape: &[i64]| -> Result<Vec<char>, String> {
            if let Some(pos) = s.find("...") {
                let prefix = &s[..pos];
                let suffix = &s[pos + 3..];
                let n_explicit = prefix.chars().count() + suffix.chars().count();
                let n_batch = shape.len().saturating_sub(n_explicit);
                // Use uppercase batch labels to avoid collisions
                let batch_labels: Vec<char> = (0..n_batch).map(|i| (b'A' + i as u8) as char).collect();
                let mut labels: Vec<char> = prefix.chars().collect();
                labels.extend(batch_labels);
                labels.extend(suffix.chars());
                Ok(labels)
            } else {
                Ok(s.chars().collect())
            }
        };

        let lhs_labels = parse_labels(lhs_str, &lhs_shape)?;
        let rhs_labels = parse_labels(rhs_str, &rhs_shape)?;

        if lhs_labels.len() != lhs_shape.len() {
            return Err(format!(
                "einsum: lhs labels {:?} don't match shape {:?}",
                lhs_labels, lhs_shape
            ));
        }
        if rhs_labels.len() != rhs_shape.len() {
            return Err(format!(
                "einsum: rhs labels {:?} don't match shape {:?}",
                rhs_labels, rhs_shape
            ));
        }

        // Parse output labels with same ellipsis expansion
        // For output, infer batch count from lhs (which always has the batch dims)
        let n_batch_lhs = if lhs_str.contains("...") {
            let explicit_lhs = lhs_str.len() - 3;
            lhs_shape.len().saturating_sub(explicit_lhs)
        } else {
            0
        };
        let out_labels = if out_str.contains("...") {
            let pos = out_str.find("...").unwrap();
            let prefix = &out_str[..pos];
            let suffix = &out_str[pos + 3..];
            let batch_labels: Vec<char> =
                (0..n_batch_lhs).map(|i| (b'A' + i as u8) as char).collect();
            let mut labels: Vec<char> = prefix.chars().collect();
            labels.extend(batch_labels);
            labels.extend(suffix.chars());
            labels
        } else {
            out_str.chars().collect()
        };

        // Build label->dim index maps
        let mut lhs_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (i, &c) in lhs_labels.iter().enumerate() {
            lhs_map.insert(c, i);
        }
        let mut rhs_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (i, &c) in rhs_labels.iter().enumerate() {
            rhs_map.insert(c, i);
        }

        let out_set: std::collections::HashSet<char> = out_labels.iter().copied().collect();

        // Batching dims: in lhs AND rhs AND output
        let mut lhs_batch_dims = Vec::new();
        let mut rhs_batch_dims = Vec::new();
        // Contracting dims: in lhs AND rhs but NOT in output
        let mut lhs_contract_dims = Vec::new();
        let mut rhs_contract_dims = Vec::new();

        for (&c, &lhs_dim) in &lhs_map {
            if let Some(&rhs_dim) = rhs_map.get(&c) {
                if out_set.contains(&c) {
                    lhs_batch_dims.push(lhs_dim as i64);
                    rhs_batch_dims.push(rhs_dim as i64);
                } else {
                    lhs_contract_dims.push(lhs_dim as i64);
                    rhs_contract_dims.push(rhs_dim as i64);
                }
            }
        }

        // dot_general output order: batch dims (sorted by lhs order) + lhs remaining + rhs remaining
        let mut batch_pairs: Vec<(i64, i64, char)> = lhs_batch_dims
            .iter()
            .zip(rhs_batch_dims.iter())
            .map(|(&l, &r)| {
                let c = lhs_labels[l as usize];
                (l, r, c)
            })
            .collect();
        batch_pairs.sort_by_key(|&(l, _, _)| l);

        let lhs_batch_sorted: Vec<i64> = batch_pairs.iter().map(|&(l, _, _)| l).collect();
        let rhs_batch_sorted: Vec<i64> = batch_pairs.iter().map(|&(_, r, _)| r).collect();
        let batch_labels_sorted: Vec<char> = batch_pairs.iter().map(|&(_, _, c)| c).collect();

        let contract_set: std::collections::HashSet<char> = lhs_contract_dims
            .iter()
            .map(|&d| lhs_labels[d as usize])
            .collect();
        let batch_set: std::collections::HashSet<char> = batch_labels_sorted.iter().copied().collect();

        // lhs remaining dims (not batch, not contracting), in order
        let mut lhs_remaining_labels = Vec::new();
        for &c in lhs_labels.iter() {
            if !batch_set.contains(&c) && !contract_set.contains(&c) {
                lhs_remaining_labels.push(c);
            }
        }

        // rhs remaining dims (not batch, not contracting), in order
        let mut rhs_remaining_labels = Vec::new();
        for &c in rhs_labels.iter() {
            if !batch_set.contains(&c) && !contract_set.contains(&c) {
                rhs_remaining_labels.push(c);
            }
        }

        // dot_general native output label order
        let mut dot_output_labels: Vec<char> = Vec::new();
        dot_output_labels.extend(&batch_labels_sorted);
        dot_output_labels.extend(&lhs_remaining_labels);
        dot_output_labels.extend(&rhs_remaining_labels);

        // Compute result shape from label->size mapping
        let mut label_sizes: std::collections::HashMap<char, i64> =
            std::collections::HashMap::new();
        for (i, &c) in lhs_labels.iter().enumerate() {
            label_sizes.insert(c, lhs_shape[i]);
        }
        for (i, &c) in rhs_labels.iter().enumerate() {
            label_sizes.insert(c, rhs_shape[i]);
        }

        let dot_result_shape: Vec<i64> = dot_output_labels
            .iter()
            .map(|c| *label_sizes.get(c).unwrap())
            .collect();
        let dot_result_ty = StableHLOType::f32_tensor(dot_result_shape);

        // Format dimension arrays
        let fmt_dims = |dims: &[i64]| -> String {
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        let reg = self.fresh_register();

        if lhs_batch_sorted.is_empty() {
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(),
                lhs.to_mlir(),
                rhs.to_mlir(),
                fmt_dims(&lhs_contract_dims),
                fmt_dims(&rhs_contract_dims),
                lhs_ty.to_mlir(),
                rhs_ty.to_mlir(),
                dot_result_ty.to_mlir()
            ));
        } else {
            self.body.push(format!(
                "    {} = stablehlo.dot_general {}, {}, batching_dims = [{}] x [{}], contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(),
                lhs.to_mlir(),
                rhs.to_mlir(),
                fmt_dims(&lhs_batch_sorted),
                fmt_dims(&rhs_batch_sorted),
                fmt_dims(&lhs_contract_dims),
                fmt_dims(&rhs_contract_dims),
                lhs_ty.to_mlir(),
                rhs_ty.to_mlir(),
                dot_result_ty.to_mlir()
            ));
        }

        // Check if we need a transpose to match the desired output order
        if dot_output_labels == out_labels {
            Ok((reg, dot_result_ty))
        } else {
            // Build permutation: for each output label, find its position in dot_output_labels
            let perm: Vec<i64> = out_labels
                .iter()
                .map(|c| {
                    dot_output_labels
                        .iter()
                        .position(|d| d == c)
                        .unwrap() as i64
                })
                .collect();
            let (treg, tty) = self.emit_transpose(&reg, &dot_result_ty, &perm);
            Ok((treg, tty))
        }
    }
}
