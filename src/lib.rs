use std::collections::HashMap;
use std::fmt::{Debug};

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::{FromPrimitive, Num, ToPrimitive, Float};
use rand::prelude::*;
use rayon::prelude::*;
use std::iter::Sum;
use ordered_float::OrderedFloat;

pub trait ToOrderedFloat<T>: where T: Float {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<T>>;
}

impl ToOrderedFloat<f32> for Vec<f32> {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<f32>> {
        self.iter().map(|x| OrderedFloat::<f32>::from(*x)).collect_vec()
    }
}

pub fn gen_ndarray<T, D>(n: usize, distr: &D) -> Array1<T>
    where
        T: Num + Clone,
        D: Distribution<T> {
    let mut random = thread_rng();
    let result = Array1::<T>::from_shape_fn(n, |_| random.sample(distr));
    result
}

pub fn gen_array<T, D>(n: usize, distr: &D) -> Vec<T>
    where
        T: Num + Clone,
        D: Distribution<T> {
    let mut random = thread_rng();
    let result : Vec<T> = (0..n).map(|_| {
        random.sample(distr)
    }).collect();
    result
}

pub fn target_encoding<D, T>(data: &mut Vec<OrderedFloat<D>>, target: &[T])
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive {
    let smoothing = 1.0;
    let min_samples_leaf = 1.;

    // group targets by each item in data
    let mut data_target: Vec<_> = data.iter().zip(target).collect(); // TODO array instead of vec
    data_target.par_sort_unstable_by_key(|x| {
            let idx = *x.0;
            idx
    });

    let sum: f64 = T::to_f64(&data_target.iter().map(|x| *x.1).sum::<T>()).unwrap();
    let prior: f64 =  sum / target.len() as f64;

    let groups = data_target.iter().group_by(|x| *x.0);
    // let groups: HashMap<D, Vec<T>> = data_target.into_iter().into_group_map();

    // calculate target encoding for each value in data
    let mut encodings: HashMap<OrderedFloat<D>, OrderedFloat<D>> = HashMap::new();
    for (k, v) in &groups {
        let val: Array1<T> = v.map(|x| {
            *x.1
        }).collect();
        let count: usize = val.len();

        if count == 1 {
            encodings.insert(k, OrderedFloat::from(D::from_f64(prior).unwrap()));
        } else {
            let group_mean: f64 = T::to_f64(&val.sum()).unwrap() / count as f64;
            let exp_count: f64 = -(count as f64 - min_samples_leaf) / smoothing;
            let smoove: f64 = 1.0 / (1.0 + exp_count.exp());
            let encoding = prior * (1.0 - smoove) + group_mean * smoove;
            // let encoding: T = T::from(val.sum() / T::from_usize(count).unwrap());
            // encodings.insert(k, encoding);
            encodings.insert(k, OrderedFloat::from(D::from_f64(encoding).unwrap()));
        }
    }

    // create encoded array
    for x in data.iter_mut() {
        *x = *encodings.get(x).unwrap();
    };
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

        let encodings = target_encoding(&mut a, &b);
        let expected = vec![1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0];

        assert_eq!(expected, a.iter().map(|x| x.0).collect_vec());
    }

    #[test]
    fn test_par() {
        let a = Array1::<i32>::from(vec![0, 1, 1, 0, 3, 0, 1]);

    }
}
