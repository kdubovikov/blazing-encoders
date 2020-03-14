use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;

use fnv::FnvHashMap;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::{Float, FromPrimitive, Num, ToPrimitive};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::utils::ToOrderedFloat;

/// Target encoding for multiple columns
pub struct TargetEncoder<D, T> where D: Float, T: Float {
    encodings: Vec<ColumnTargetEncoder<D, T>>,
    phantom_target: PhantomData<T>,
}

/// Target encoding on single column.
///
/// The research paper describing the algorithm can be found here: [A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems](https://dl.acm.org/doi/10.1145/507533.507538)
pub struct ColumnTargetEncoder<D, T> where D: Float, T: Float {
    encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>>,
    phantom_target: PhantomData<T>,
}

impl<D, T> TargetEncoder<D, T>
    where
        T: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send,
        D: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send {
    /// Create a new `TargetEncoder` and compute target encodings for all columns.
    /// This function does not transform the original dataset. See [`transform`](TargetEncoder::transform)
    pub fn fit(data: &Array2<OrderedFloat<D>>, target: &[T]) -> TargetEncoder<D, T> {
        let mut encodings: Vec<ColumnTargetEncoder<D, T>> = Vec::with_capacity(data.len_of(Axis(1)));

        data.axis_iter(Axis(1)).into_par_iter().enumerate().map(|(_, row)| {
            let mut owned_row = row.to_owned();
            let enc = Vec::from(owned_row.as_slice_mut().unwrap());
            let mut enc = enc.iter().map(|x| OrderedFloat::<D>::from(*x)).collect_vec();
            let encoder = ColumnTargetEncoder::fit(&mut enc, target);
            encoder
        }).collect_into_vec(&mut encodings);

        TargetEncoder { encodings, phantom_target: PhantomData }
    }

    /// Performs target encoding on provided `data`
    pub fn transform(&self, data: &mut Array2<OrderedFloat<D>>) {
        for (i, mut row) in data.axis_iter_mut(Axis(1)).enumerate() {
            self.encodings[i].transform_arr(&mut row);
        }
    }
}

impl<D, T> ColumnTargetEncoder<D, T>
    where
        T: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send,
        D: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send {
    /// Create new `ColumnTargetEncoder` and compute target encodings for a single column.
    /// This function does not transform the original dataset. See [`transform`](ColumnTargetEncoder::transform)
    pub fn fit(data: &Vec<OrderedFloat<D>>, target: &[T]) -> ColumnTargetEncoder<D, T> {
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

        ColumnTargetEncoder { encodings, phantom_target: PhantomData }
    }

    /// Encode provided `data`. If you need to transform an `ndarray` see [transform_arr](ColumnTargetEncoder::transform_arr).
    pub fn transform(&self, data: &mut Vec<OrderedFloat<D>>) {
        for x in data.iter_mut() {
            *x = *self.encodings.get(x).unwrap();
        };
    }

    /// Encode provided `data`
    pub fn transform_arr(&self, data: &mut ArrayViewMut1<OrderedFloat<D>>) {
        data.map_mut(|x| *x = *self.encodings.get(x).unwrap());
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
    use ndarray::Zip;
    use numpy::npyffi::array;

    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_fit_one_column() {
        let x = vec![0., 1., 1., 0., 3., 0., 1.];
        let mut x = x.to_ordered_float();
        let y = [1., 2., 2., 1., 0., 1., 2.];

        let encoder = ColumnTargetEncoder::fit(&x, &y);
        encoder.transform(&mut x);
        // target_encoding(&mut a, &b);
        let expected = vec![
            1.0340579777206051, 1.9148550556984874,
            1.9148550556984874, 1.0340579777206051,
            1.2857142857142858, 1.0340579777206051,
            1.9148550556984874];
        let actual = x.iter().map(|x| x.0).collect_vec();
        expected.iter().zip(actual.iter()).for_each(|(expected, actual): (&f64, &f64)| {
            assert_approx_eq!(expected, actual);
        });
    }

    #[test]
    fn test_fit_one_category() {
        let mut x = Array2::<f64>::zeros((10, 7)).mapv(OrderedFloat::from);
        let y = [1., 2., 2., 1., 0., 1., 2.];

        let encoder = TargetEncoder::fit(&x, &y);
        encoder.transform(&mut x);

        for r in x.iter() {
            assert_approx_eq!(r.0, 1.2857142857142858);
        }
    }

    #[test]
    fn test_fit_compare_category_encoders() {
        let a = array![[2., 6., 3., 5., 4.],
                       [3., 2., 2., 5., 3.],
                       [8., 4., 5., 3., 1.],
                       [5., 0., 2., 4., 9.],
                       [9., 5., 2., 0., 7.]];

        let mut a = a.mapv(OrderedFloat::from);

        let b = [0.48263811, 0.16705367, 0.32397016, 0.10172379, 0.54362169];
        let expected = array![[0.32380149, 0.32380149, 0.32380149, 0.32456501, 0.32380149],
                              [0.32380149, 0.32380149, 0.27711768, 0.32456501, 0.32380149],
                              [0.32380149, 0.32380149, 0.32380149, 0.32380149, 0.32380149],
                              [0.32380149, 0.32380149, 0.27711768, 0.32380149, 0.32380149],
                              [0.32380149, 0.32380149, 0.27711768, 0.32380149, 0.32380149]];

        let encoder = TargetEncoder::fit(&mut a, &b);
        encoder.transform(&mut a);

        Zip::from(&a).and(&expected).apply(|&a, &expected| {
            assert_approx_eq!(a.0, expected);
        });
    }
}
