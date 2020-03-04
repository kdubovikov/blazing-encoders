use std::fmt::{Debug};

use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::{FromPrimitive, Num, ToPrimitive, Float, AsPrimitive};
use rand::prelude::*;
use rayon::prelude::*;
use std::iter::Sum;
use ordered_float::OrderedFloat;
use fnv::FnvHashMap;


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
    let prior: f64 =  sum / target.len() as f64;

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
    use super::*;
    use crate::target_encoding;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_target_encoding() {
        let a = vec![0., 1., 1., 0., 3., 0., 1.];
        let mut a = a.to_ordered_float();
        let b = [1., 2., 2., 1., 0., 1., 2.];
        // let mut a = Array1::from(a);
        // let b = Array1::from(b);

        let encodings = target_encoding(&mut a, &b);
        let expected = vec![1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0];

        assert_eq!(expected, a.iter().map(|x| x.0).collect_vec());
    }

    #[test]
    fn test_par() {
        let a = Array1::<i32>::from(vec![0, 1, 1, 0, 3, 0, 1]);

    }
}
