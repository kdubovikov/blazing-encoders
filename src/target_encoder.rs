use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;

use fnv::FnvHashMap;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::{Float, FromPrimitive, Num, ToPrimitive};
use ordered_float::OrderedFloat;
// use rand::prelude::*;
use rayon::prelude::*;

use crate::utils::ToOrderedFloat;
use ndarray::IntoNdProducer;

pub struct TargetEncoder<D, T>
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive + Float + Send,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive + Send {
    encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>>,
    num_groups: usize,
    phantom_target: PhantomData<T>
}

impl<D, T> TargetEncoder<D, T>
    where
        T: Num + Copy + FromPrimitive + Sync + ToPrimitive + Debug + Sum + FromPrimitive + Float + Send,
        D: Float + Sync + Debug + Sum + ToPrimitive + FromPrimitive + Send {

    pub fn fit(data: &Array2<OrderedFloat<D>>, target: &[T]) -> TargetEncoder<D, T> {
        let mut encodings: Vec<(usize, TargetEncoder<D, T>)> = Vec::new();

        data.axis_iter(Axis(1)).into_par_iter().enumerate().map(|(i, row)| {
            let mut owned_row = row.to_owned();
            let mut enc = Vec::from(owned_row.as_slice_mut().unwrap());
            let mut enc = enc.iter().map(|x| OrderedFloat::<D>::from(*x)).collect_vec();
            let encoder = Self::fit_one_column(&mut enc, target);
            (i, encoder)
        }).collect_into_vec(&mut encodings);

        encodings.sort_unstable_by_key(|(i, encoder)| *i);
        let num_groups = encodings.iter().map(|(i, encoder)| encoder.num_groups).sum::<usize>();
        let num_groups= num_groups / encodings.len();
        let mut comb_encodings = FnvHashMap::with_capacity_and_hasher(num_groups, Default::default());

        for (_, enc) in encodings {
            let x = &enc.encodings;
            comb_encodings.extend(x);
        }

        TargetEncoder { encodings: comb_encodings, num_groups, phantom_target: PhantomData }

        // let result = result.iter()
        //     .sorted_by_key(|(i, enc)| i).collect_vec();
        // let result = result.iter()
        //     .map(|(i, enc)| enc)
        //     .flatten()
        //     .map(|x| x.0)
        //     .collect_vec();
        //
        // // .f() for fortran-order matrices
        // Array2::from_shape_vec(data.raw_dim().f(), result)
    }

    pub fn fit_one_column(data: &Vec<OrderedFloat<D>>, target: &[T]) -> TargetEncoder<D, T>
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
                let encoding = Self::compute_encoding(&val, count as f64, min_samples_leaf, smoothing, prior);
                encodings.insert(k, OrderedFloat::from(D::from_f64(encoding).unwrap()));
            }
        }

        TargetEncoder {encodings, num_groups, phantom_target: PhantomData }
    }

    /// create encoded array
    pub fn transform_vec(&self, data: &mut Vec<OrderedFloat<D>>) {
        for x in data.iter_mut() {
            *x = *self.encodings.get(x).unwrap();
        };
    }

    pub fn transform_arr(&self, data: &mut ArrayViewMut1<OrderedFloat<D>>) {
        data.map_mut(|x| *x = *self.encodings.get(x).unwrap());
    }

    pub fn transform_mat(&self, data: &mut Array2<OrderedFloat<D>>) {
        for mut row in data.outer_iter_mut() {
            self.transform_arr(&mut row);
        }
    }

    fn compute_encoding(val: &Array1<f64>, count: f64, min_samples_leaf: f64, smoothing: f64, prior: f64) -> f64 {
        let group_mean: f64 = &val.sum() / count as f64;
        let exp_count: f64 = -(count as f64 - min_samples_leaf as f64) / smoothing;
        let smoove: f64 = 1.0 / (1.0 + exp_count.exp());
        prior * (1.0 - smoove) + group_mean * smoove
    }
}



#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    #[test]
    fn test_target_encoding() {
        let a = vec![0., 1., 1., 0., 3., 0., 1.];
        let mut a = a.to_ordered_float();
        let b = [1., 2., 2., 1., 0., 1., 2.];

        let encoder = TargetEncoder::fit_one_column(&a, &b);
        encoder.transform_vec(&mut a);
        // target_encoding(&mut a, &b);
        let expected = vec![1.0340579777206051, 1.9148550556984874, 1.9148550556984874, 1.0340579777206051, 1.2857142857142858, 1.0340579777206051, 1.9148550556984874];
        let a = a.iter().map(|x| x.0).collect_vec();
        expected.iter().zip(a.iter()).map(|(e, a): (&f64, &f64)| {
            assert_approx_eq!(e, a);
        });
    }

    #[test]
    fn test_par() {
        let mut a = Array2::<f64>::zeros((10, 7)).mapv(OrderedFloat::from);
        let b = [1., 2., 2., 1., 0., 1., 2.];

        let encoder = TargetEncoder::fit(&mut a, &b);
        encoder.transform_mat(&mut a);

        for r in a.iter() {
            assert_approx_eq!(r.0, 1.2857142857142858);
        }
    }
}
