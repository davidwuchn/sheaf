// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Lowers matrix multiplication and einsum to StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emits matrix multiplication with NumPy `@` behavior:
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

        // Multiplication handles scalar operands; dot_general does not.
        if lhs_rank == 0 || rhs_rank == 0 {
            return self.emit_binop("*", lhs, rhs, lhs_ty, rhs_ty);
        }

        // [K] @ [K] -> scalar
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

        // [K] @ [K, N] -> [N]
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

        // [..., M, K] @ [K] -> [..., M]
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

    // Remove leading batch dimensions of size one when the other operand is a
    // matrix. IREE handles the resulting two-dimensional multiplication faster.
    let lhs_leading_1 = if lhs_rank > 2 && rhs_rank <= 2 {
        let mut n = 0;
        for &d in &lhs_shape[..lhs_rank - 2] {
            if d == 1 { n += 1; } else { break; }
        }
        n
    } else {
        0
    };
    let rhs_leading_1 = if rhs_rank > 2 && lhs_rank <= 2 {
        let mut n = 0;
        for &d in &rhs_shape[..rhs_rank - 2] {
            if d == 1 { n += 1; } else { break; }
        }
        n
    } else {
        0
    };

    if lhs_leading_1 > 0 {
        let lhs_flat_shape: Vec<i64> = lhs_shape[lhs_leading_1..].to_vec();
        let (lhs_flat, lhs_flat_ty) = self.emit_reshape(lhs, lhs_ty, &lhs_flat_shape);
        let flat_lhs_contract = lhs_contract - lhs_leading_1 as i64;
        let flat_result_shape: Vec<i64> = result_shape[lhs_leading_1..].to_vec();
        let flat_result_ty = StableHLOType::f32_tensor(flat_result_shape.clone());
        let reg = self.fresh_register();
        self.body.push(format!(
            " {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
            reg.to_mlir(), lhs_flat.to_mlir(), rhs.to_mlir(),
            flat_lhs_contract, rhs_contract,
            lhs_flat_ty.to_mlir(), rhs_ty.to_mlir(), flat_result_ty.to_mlir()
        ));
        let result_ty = StableHLOType::f32_tensor(result_shape.clone());
        let (result_reg, _) = self.emit_reshape(&reg, &flat_result_ty, &result_shape);
        return (result_reg, result_ty);
    }

    if rhs_leading_1 > 0 {
        let rhs_flat_shape: Vec<i64> = rhs_shape[rhs_leading_1..].to_vec();
        let (rhs_flat, rhs_flat_ty) = self.emit_reshape(rhs, rhs_ty, &rhs_flat_shape);
        let flat_rhs_contract = rhs_contract - rhs_leading_1 as i64;
        let flat_result_shape: Vec<i64> = result_shape[rhs_leading_1..].to_vec();
        let flat_result_ty = StableHLOType::f32_tensor(flat_result_shape.clone());
        let reg = self.fresh_register();
        self.body.push(format!(
            " {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
            reg.to_mlir(), lhs.to_mlir(), rhs_flat.to_mlir(),
            lhs_contract, flat_rhs_contract,
            lhs_ty.to_mlir(), rhs_flat_ty.to_mlir(), flat_result_ty.to_mlir()
        ));
        let result_ty = StableHLOType::f32_tensor(result_shape.clone());
        let (result_reg, _) = self.emit_reshape(&reg, &flat_result_ty, &result_shape);
        return (result_reg, result_ty);
    }

    // Batch dimensions of size one can use a regular matrix multiplication.
    let all_batch_are_1 = n_batch > 0
        && lhs_shape[..n_batch].iter().all(|&d| d == 1)
        && rhs_shape[..n_batch].iter().all(|&d| d == 1);

    if all_batch_are_1 {
        let lhs_2d_shape = vec![lhs_shape[n_batch..lhs_rank - 1].iter().product::<i64>(), lhs_shape[lhs_rank - 1]];
        let rhs_2d_shape = vec![rhs_shape[n_batch], rhs_shape[n_batch + 1..].iter().product::<i64>()];
        let (lhs_flat, lhs_2d_ty) = self.emit_reshape(lhs, lhs_ty, &lhs_2d_shape);
        let (rhs_flat, rhs_2d_ty) = self.emit_reshape(rhs, rhs_ty, &rhs_2d_shape);
        let m = lhs_2d_shape[0];
        let n = rhs_2d_shape[1];
        let result_2d_shape = vec![m, n];
        let result_2d_ty = StableHLOType::f32_tensor(result_2d_shape.clone());
        let reg = self.fresh_register();
        self.body.push(format!(
            " {} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0] : ({}, {}) -> {}",
            reg.to_mlir(), lhs_flat.to_mlir(), rhs_flat.to_mlir(),
            lhs_2d_ty.to_mlir(), rhs_2d_ty.to_mlir(), result_2d_ty.to_mlir()
        ));
        let result_ty = StableHLOType::f32_tensor(result_shape.clone());
        let (result_reg, _) = self.emit_reshape(&reg, &result_2d_ty, &result_shape);
        return (result_reg, result_ty);
    }

    // Remove shared batch dimensions of size one before generating the loop.
    let n_peel = if n_batch > 0 {
        let mut n = 0;
        for i in 0..n_batch {
            if lhs_shape[i] == 1 && rhs_shape[i] == 1 {
                n += 1;
            } else {
                break;
            }
        }
        n
    } else {
        0
    };

    if n_peel > 0 {
        let lhs_flat_shape: Vec<i64> = lhs_shape[n_peel..].to_vec();
        let rhs_flat_shape: Vec<i64> = rhs_shape[n_peel..].to_vec();
        let (lhs_flat, lhs_flat_ty) = self.emit_reshape(lhs, lhs_ty, &lhs_flat_shape);
        let (rhs_flat, rhs_flat_ty) = self.emit_reshape(rhs, rhs_ty, &rhs_flat_shape);
        let lhs_flat_contract = lhs_contract - n_peel as i64;
        let rhs_flat_contract = rhs_contract - n_peel as i64;
        let flat_result_shape: Vec<i64> = result_shape[n_peel..].to_vec();
        let flat_result_ty = StableHLOType::f32_tensor(flat_result_shape.clone());
        let remaining_batch = n_batch - n_peel;
        let reg = self.fresh_register();
        if remaining_batch == 0 {
            self.body.push(format!(
                " {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(), lhs_flat.to_mlir(), rhs_flat.to_mlir(),
                lhs_flat_contract, rhs_flat_contract,
                lhs_flat_ty.to_mlir(), rhs_flat_ty.to_mlir(), flat_result_ty.to_mlir()
            ));
        } else {
            let batch_dims: Vec<String> = (0..remaining_batch).map(|i| i.to_string()).collect();
            self.body.push(format!(
                " {} = stablehlo.dot_general {}, {}, batching_dims = [{}] x [{}], contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                reg.to_mlir(), lhs_flat.to_mlir(), rhs_flat.to_mlir(),
                batch_dims.join(", "), batch_dims.join(", "),
                lhs_flat_contract, rhs_flat_contract,
                lhs_flat_ty.to_mlir(), rhs_flat_ty.to_mlir(), flat_result_ty.to_mlir()
            ));
        }
        let result_ty = StableHLOType::f32_tensor(result_shape.clone());
        let (result_reg, _) = self.emit_reshape(&reg, &flat_result_ty, &result_shape);
        return (result_reg, result_ty);
    }

    let result_ty = StableHLOType::f32_tensor(result_shape);
    let reg = self.fresh_register();

    if n_batch == 0 {
        self.body.push(format!(
            "  {} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
            reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
            lhs_contract, rhs_contract,
            lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
        ));
    } else {
        let batch_dims: Vec<String> = (0..n_batch).map(|i| i.to_string()).collect();
        let batch_str = batch_dims.join(", ");
        self.body.push(format!(
            "  {} = stablehlo.dot_general {}, {}, batching_dims = [{}] x [{}], contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
            reg.to_mlir(), lhs.to_mlir(), rhs.to_mlir(),
            batch_str, batch_str,
            lhs_contract, rhs_contract,
            lhs_ty.to_mlir(), rhs_ty.to_mlir(), result_ty.to_mlir()
        ));
    }

    (reg, result_ty)
    }

    /// Emits a two-input einsum using `dot_general` and, when needed, a transpose.
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

        // Give each dimension represented by `...` a temporary label.
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

        let lhs_labels = parse_labels(lhs_str, lhs_shape)?;
        let rhs_labels = parse_labels(rhs_str, rhs_shape)?;

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

        // The left input tells us how many dimensions `...` represents.
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

        let mut lhs_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (i, &c) in lhs_labels.iter().enumerate() {
            lhs_map.insert(c, i);
        }
        let mut rhs_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (i, &c) in rhs_labels.iter().enumerate() {
            rhs_map.insert(c, i);
        }

        let out_set: std::collections::HashSet<char> = out_labels.iter().copied().collect();

        // A label present in both inputs and the output is a batch dimension.
        let mut lhs_batch_dims = Vec::new();
        let mut rhs_batch_dims = Vec::new();
        // A label present in both inputs but not the output is summed over.
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

        // dot_general puts batch dimensions first, followed by each input's remaining dimensions.
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

        // Keep the remaining dimensions in their original order.
        let mut lhs_remaining_labels = Vec::new();
        for &c in lhs_labels.iter() {
            if !batch_set.contains(&c) && !contract_set.contains(&c) {
                lhs_remaining_labels.push(c);
            }
        }

        let mut rhs_remaining_labels = Vec::new();
        for &c in rhs_labels.iter() {
            if !batch_set.contains(&c) && !contract_set.contains(&c) {
                rhs_remaining_labels.push(c);
            }
        }

        let mut dot_output_labels: Vec<char> = Vec::new();
        dot_output_labels.extend(&batch_labels_sorted);
        dot_output_labels.extend(&lhs_remaining_labels);
        dot_output_labels.extend(&rhs_remaining_labels);

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
