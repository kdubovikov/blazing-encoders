use std::fmt::{Debug};

use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::{FromPrimitive, Num, ToPrimitive, Float, AsPrimitive};
use rand::prelude::*;
use rayon::prelude::*;
use std::iter::Sum;
use ordered_float::OrderedFloat;
use fnv::FnvHashMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::io;
use std::fmt::Write as FmtWrite;

pub trait ToOrderedFloat<T>: where T: Float {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<T>>;
}

impl ToOrderedFloat<f32> for Vec<f32> {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<f32>> {
        self.iter().map(|x| OrderedFloat::<f32>::from(*x)).collect_vec()
    }
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

pub fn gen_array_f32<T, D>(n: usize, distr: &D) -> Vec<f32>
    where
        T: Num + Clone + AsPrimitive<f32>,
        D: Distribution<T> {
    gen_array(n, distr).iter().map(|x| x.as_()).collect_vec()
}

pub fn target_encoding<D, T>(data: &mut Vec<OrderedFloat<D>>, target: &[T])
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive + Send {
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
    // TODO pre-collect hashmap and execute computation in parallel
    let mut groups_map: FnvHashMap<OrderedFloat<D>, Array1<T>> = FnvHashMap::with_capacity_and_hasher(num_groups, Default::default());
    // let mut groups_map: HashMap<OrderedFloat<D>, Array1<T>> = HashMap::with_capacity_and_hasher(num_groups, Default::default());
    for (k, v) in &groups {
        let val: Array1<T> = v.map(|x| {
            *x.1
        }).collect();
        groups_map.insert(k, val);
    }

    // let mut encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>> = FnvHashMap::default();
    // pre-allocate hashmap for performance boost
    let mut encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>> = FnvHashMap::with_capacity_and_hasher(num_groups, Default::default());
    let enc = Arc::new(Mutex::new(encodings));
    let enc_c = enc.clone();

    groups_map.iter().par_bridge().for_each(|(k, val)| {
        let count: usize = val.len();
        let mut enc_locked = enc_c.lock().unwrap();

        if count == 1 {
            enc_locked.insert(*k, OrderedFloat::from(D::from_f64(prior).unwrap()));
        } else {
            let group_mean: f64 = T::to_f64(&val.sum()).unwrap() / count as f64;
            let exp_count: f64 = -(count as f64 - min_samples_leaf) / smoothing;
            let smoove: f64 = 1.0 / (1.0 + exp_count.exp());
            let encoding = prior * (1.0 - smoove) + group_mean * smoove;
            enc_locked.insert(*k, OrderedFloat::from(D::from_f64(encoding).unwrap()));
        }
    });

    let enc = enc.lock().unwrap();

    // create encoded array
    for x in data.iter_mut() {
        *x = *enc.get(x).unwrap();
    };
}


#[cfg(test)]
mod tests {
    use itertools::Itertools;
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
}
