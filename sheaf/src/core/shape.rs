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

#[cfg(test)]
mod tests {
    use super::{BroadcastError, broadcast_shapes};

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
