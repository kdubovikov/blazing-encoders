use std::fmt::Debug;
use std::iter::Sum;

use fnv::FnvHashMap;
use itertools::Itertools;
use ndarray::{ShapeError, ViewRepr};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, ToPrimitive};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;

use crate::utils::ToOrderedFloat;

pub fn par_column_target_encoding<D, T>(data: &Array2<D>, target: &[T]) -> Result<Array2<D>, ShapeError>
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive + Float + Send,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive + Send {
    let mut result = Vec::new();
    data.axis_iter(Axis(1)).into_par_iter().enumerate().map(|(i, row)| {
        let mut owned_row = row.to_owned();
        let mut enc = Vec::from(owned_row.as_slice_mut().unwrap());
        let mut enc = enc.iter().map(|x| OrderedFloat::<D>::from(*x)).collect_vec();
        target_encoding(&mut enc, target);
        (i, enc)
    })
        .collect_into_vec(&mut result);

    let result = result.iter()
        .sorted_by_key(|(i, enc)| i).collect_vec();
    let result = result.iter()
        .map(|(i, enc)| enc)
        .flatten()
        .map(|x| x.0)
        .collect_vec();

    // .f() for fortran-order matrices
    Array2::from_shape_vec(data.raw_dim().f(), result)
}

pub fn target_encoding<D, T>(data: &mut Vec<OrderedFloat<D>>, target: &[T])
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive {
    let smoothing = 1.0;
    let min_samples_leaf = 1.;

    // group targets by each item in data
    let mut data_target: Vec<_> = data.iter().zip(target).collect(); // TODO array instead of vec
    data_target.par_sort_unstable_by_key(|x| *x.0);

    let d = data_target.iter().map(|(a, _)| *a).collect_vec();
    let num_groups: usize = d.into_iter().map(|x| *x).dedup().count();

    let sum: f64 = T::to_f64(&data_target.iter().map(|x| *x.1).sum::<T>()).unwrap();
    let prior: f64 = sum / target.len() as f64;

    let groups = data_target.into_iter().group_by(|x| *x.0);

    // calculate target encoding for each value in data
    // let mut encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>> = FnvHashMap::default();
    let mut encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>> = FnvHashMap::with_capacity_and_hasher(num_groups, Default::default());
    for (k, v) in &groups {
        let val: Array1<f64> = v.map(|x| {
            T::to_f64(x.1).unwrap()
        }).collect();
        let count: usize = val.len();

        if count == 1 {
            encodings.insert(k, OrderedFloat::from(D::from_f64(prior).unwrap()));
        } else {
            let encoding = compute_encoding(&val, count as f64, min_samples_leaf, smoothing, prior);
            encodings.insert(k, OrderedFloat::from(D::from_f64(encoding).unwrap()));
        }
    }

    // create encoded array
    for x in data.iter_mut() {
        *x = *encodings.get(x).unwrap();
    };
}

fn compute_encoding(val: &Array1<f64>, count: f64, min_samples_leaf: f64, smoothing: f64, prior: f64) -> f64 {
    let group_mean: f64 = &val.sum() / count as f64;
    let exp_count: f64 = -(count as f64 - min_samples_leaf as f64) / smoothing;
    let smoove: f64 = 1.0 / (1.0 + exp_count.exp());
    prior * (1.0 - smoove) + group_mean * smoove
}


#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use ndarray::{Array1, Zip};

    use assert_approx_eq::assert_approx_eq;

    use crate::target_encoding;

    use super::*;

    #[test]
    fn test_target_encoding() {
        let a = vec![0., 1., 1., 0., 3., 0., 1.];
        let mut a = a.to_ordered_float();
        let b = [1., 2., 2., 1., 0., 1., 2.];

        target_encoding(&mut a, &b);
        let expected = vec![1.0340579777206051, 1.9148550556984874, 1.9148550556984874, 1.0340579777206051, 1.2857142857142858, 1.0340579777206051, 1.9148550556984874];
        let a = a.iter().map(|x| x.0).collect_vec();
        expected.iter().zip(a.iter()).map(|(e, a): (&f64, &f64)| {
            assert_approx_eq!(e, a);
        });
    }

    #[test]
    fn test_par() {
        let mut a = Array2::<f64>::zeros((10, 7));
        let b = [1., 2., 2., 1., 0., 1., 2.];

        let result = par_column_target_encoding(&a, &b).unwrap();
        assert_eq!(result.shape(), a.shape());

        const RESULT: f64 = 1.2857142857142858;
        for r in result.iter() {
            assert_approx_eq!(r, RESULT);
        }
    }
}
