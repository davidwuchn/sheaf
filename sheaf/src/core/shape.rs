// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shape operations

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastError<Dimension> {
    pub axis_from_right: usize,
    pub lhs: Dimension,
    pub rhs: Dimension,
}

pub fn broadcast_shapes<Dimension>(
    lhs: &[Dimension],
    rhs: &[Dimension],
) -> Result<Vec<Dimension>, BroadcastError<Dimension>>
where
    Dimension: Copy + Eq + From<u8>,
{
    let one = Dimension::from(1);
    let rank = lhs.len().max(rhs.len());
    let mut reversed = Vec::with_capacity(rank);

    for axis_from_right in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(axis_from_right + 1)
            .map(|index| lhs[index])
            .unwrap_or(one);
        let rhs_dim = rhs
            .len()
            .checked_sub(axis_from_right + 1)
            .map(|index| rhs[index])
            .unwrap_or(one);

        let result = if lhs_dim == rhs_dim || rhs_dim == one {
            lhs_dim
        } else if lhs_dim == one {
            rhs_dim
        } else {
            return Err(BroadcastError {
                axis_from_right,
                lhs: lhs_dim,
                rhs: rhs_dim,
            });
        };
        reversed.push(result);
    }

    reversed.reverse();
    Ok(reversed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumToShapeError<Dimension> {
    Rank { source: usize, target: usize },
    Dimension { axis_from_right: usize, source: Dimension, target: Dimension },
}

/// Validate a broadcast adjoint contraction and return axes in the source shape.
pub fn sum_to_shape_axes<Dimension>(
    source: &[Dimension],
    target: &[Dimension],
) -> Result<Vec<usize>, SumToShapeError<Dimension>>
where
    Dimension: Copy + Eq + From<u8>,
{
    if target.len() > source.len() {
        return Err(SumToShapeError::Rank { source: source.len(), target: target.len() });
    }
    let rank = source.len();
    let extra = rank - target.len();
    let mut axes: Vec<_> = (0..extra).collect();
    for (index, (&source, &target)) in source[extra..].iter().zip(target).enumerate() {
        if source == target {
            continue;
        }
        if target != Dimension::from(1) {
            return Err(SumToShapeError::Dimension {
                axis_from_right: rank - 1 - extra - index,
                source,
                target,
            });
        }
        axes.push(extra + index);
    }
    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::{BroadcastError, broadcast_shapes};

    #[test]
    fn adjoint_contraction_requires_a_broadcast_suffix() {
        use super::sum_to_shape_axes;
        assert_eq!(sum_to_shape_axes(&[2usize, 3, 4], &[1, 4]), Ok(vec![0, 1]));
        assert_eq!(sum_to_shape_axes(&[2usize, 3], &[]), Ok(vec![0, 1]));
        assert_eq!(sum_to_shape_axes(&[2usize, 1], &[2]), Err(super::SumToShapeError::Dimension {
            axis_from_right: 0, source: 1, target: 2,
        }));
        assert!(sum_to_shape_axes(&[2usize], &[1, 2]).is_err());
        assert_eq!(sum_to_shape_axes(&[0usize, 3], &[1, 3]), Ok(vec![0]));
    }

    #[test]
    fn broadcasts_trailing_dimensions() {
        assert_eq!(broadcast_shapes(&[2i64, 1], &[1, 3]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[] as &[usize], &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[4usize], &[2, 3, 4]), Ok(vec![2, 3, 4]));
    }

    #[test]
    fn reports_the_first_incompatible_trailing_axis() {
        assert_eq!(
            broadcast_shapes(&[2i64, 3], &[2, 4]),
            Err(BroadcastError {
                axis_from_right: 0,
                lhs: 3,
                rhs: 4,
            }),
        );
    }
}
