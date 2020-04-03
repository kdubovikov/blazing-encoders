use std::iter::Sum;
use std::marker::PhantomData;

use fnv::FnvHashMap;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::stats::SummaryStatistics;
use crate::utils::ToOrderedFloat;
use std::fmt::Debug;

// pub trait Encoder<T>
// where
//     T: Float + FromPrimitive,
// {
//     fn compute_encoding(&self) -> T;
// }

/// Target encoding for multiple columns
pub struct MatrixEncoder<D, T>
where
    D: Float,
    T: Float,
{
    encodings: Vec<ColumnTargetEncoder<D, T>>,
    phantom_target: PhantomData<T>,
}

/// Target encoding on single column.
///
/// The research paper describing the algorithm can be found here: [A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems](https://dl.acm.org/doi/10.1145/507533.507538)
pub struct ColumnTargetEncoder<D, T>
where
    D: Float,
    T: Float,
{
    encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>>,
    phantom_target: PhantomData<T>
}

// impl<'a, T: Float + FromPrimitive, D: Float> Encoder<T> for ColumnTargetEncoder<'a, T, D> {
//     fn compute_encoding(&self) -> T {
//         let group_mean = self.val.mean().unwrap();
//         let exp_count = -(self.count - T::from(self.min_samples_leaf).unwrap()) / self.smoothing;
//         let smoove = T::one() / (T::one() + exp_count.exp());
//         self.prior * (T::one() - smoove) + group_mean * smoove
//     }
// }

// enum GlobalStats<T: Float> {
//     TargetEncoder { prior: T },
// }

// fn compute_global_stats<D: Float, T: Float>(
//     encoder: &Encoders<T>,
//     data: &Array1<T>,
//     target: &Array1<T>,
// ) -> GlobalStats<T> {
//     match encoder {
//         Encoders::TargetEncoder {
//             min_samples_leaf: usize,
//             smoothing: T,
//         } => {
//             let prior = D::from(target.sum()) / D::from(target.len());
//             GlobalStats::TargetEncoder { prior }
//         }
//     }
// }

pub struct TargetStats<T: Float + FromPrimitive> {
    prior: T
}

impl<T: Float + FromPrimitive> TargetStats<T> {
    fn new(target: &Array1<T>) -> TargetStats<T> {
        let prior = target.sum() / T::from_usize(target.len()).unwrap();
        TargetStats { prior }
    }
}

#[derive(Debug)]
pub enum Encoders<T>
where
    T: Float + FromPrimitive,
{
    TargetEncoder {
        min_samples_leaf: usize,
        smoothing: T,
        prior: Option<T>
    },
    JamesSteinEncoder {
        sigma: T
    },
}

impl<T: Float + FromPrimitive> Encoders<T> {
    pub fn new_target_encoder(min_samples_leaf: usize, smoothing: T) -> Encoders<T> {
        Encoders::TargetEncoder { min_samples_leaf, smoothing, prior: None }
    }

    pub fn new_james_stein_encoder(sigma: T) -> Encoders<T> {
        Encoders::JamesSteinEncoder { sigma }
    }

    pub fn compute_stats(&mut self, target: &Array1<T>) {
        match self {
            Encoders::TargetEncoder {min_samples_leaf, smoothing, prior} => {
                let a = target.mean().unwrap();
                *prior = Some(a);
            },
            Encoders::JamesSteinEncoder { sigma } => {
                unimplemented!()
            }
        }
    }
}

fn compute_encoding<T>(encoder: &Encoders<T>, val: &Array1<T>) -> T where T: Float + FromPrimitive {
    match *encoder {
        Encoders::TargetEncoder {min_samples_leaf, smoothing, prior} => {
            const err_msg: &str = "Target statistics were not computed prior to encoding computation";
            let prior = prior.expect(err_msg);
            let col_len =  T::from_usize(val.len()).unwrap();

            let group_mean = val.mean().unwrap();
            let exp_count = -(col_len - T::from_usize(min_samples_leaf).unwrap()) / T::from(smoothing).unwrap();
            let smoove = T::one() / (T::one() + exp_count.exp());
            prior * (T::one() - smoove) + group_mean * smoove
        },

        Encoders::JamesSteinEncoder {sigma} => {
            let var = val.var_axis(Axis(0), T::one());
            unimplemented!()
        }
    }
}

impl<D, T> MatrixEncoder<D, T>
where
    T: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send,
    D: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send,
{
    /// Create a new `TargetEncoder` and compute target encodings for all columns.
    /// This function does not transform the original dataset. See [`transform`](TargetEncoder::transform)
    pub fn fit(
        data: &Array2<OrderedFloat<D>>,
        target: &Array1<T>,
        encoder: &mut Encoders<D>,
    ) -> MatrixEncoder<D, T> {
        encoder.compute_stats(&target.map(|x| D::from(*x).unwrap()));
        let mut encodings: Vec<ColumnTargetEncoder<D, T>> =
            Vec::with_capacity(data.len_of(Axis(1)));

        data.axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .map(|(_, row)| {
                let mut owned_row = row.to_owned();
                let mut enc = owned_row
                    .map(|x| OrderedFloat::<D>::from(*x));
                ColumnTargetEncoder::fit(&mut enc, target, encoder)
            })
            .collect_into_vec(&mut encodings);

        MatrixEncoder {
            encodings,
            phantom_target: PhantomData,
        }
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
    D: Float + Sum + FromPrimitive + ToPrimitive + Sync + Send,
{
    /// Create new `ColumnTargetEncoder` and compute target encodings for a single column.
    /// This function does not transform the original dataset. See [`transform`](ColumnTargetEncoder::transform)
    pub fn fit(
        data: &Array1<OrderedFloat<D>>,
        target: &Array1<T>,
        encoder: &Encoders<D>,
    ) -> ColumnTargetEncoder<D, T> {
        // group targets by each item in data
        let mut data_target: Vec<_> = data.iter().zip(target).collect(); // TODO array instead of vec
        data_target.par_sort_unstable_by_key(|x| *x.0);

        let d = data_target.iter().map(|(a, _)| *a).collect_vec();
        let num_groups: usize = d.into_iter().map(|x| *x).dedup().count();

        let sum = D::from(data_target.iter().map(|x| *x.1).sum::<T>()).unwrap();
        let prior = sum / D::from(target.len()).unwrap(); // TODO this can be pushed up to MatrixEncoder

        let groups = data_target.into_iter().group_by(|x| *x.0);

        // calculate target encoding for each value in data
        let mut encodings: FnvHashMap<OrderedFloat<D>, OrderedFloat<D>> =
            FnvHashMap::with_capacity_and_hasher(num_groups, Default::default());

        for (k, v) in &groups {
            let val: Array1<D> = v.map(|x| D::from(*x.1).unwrap()).collect();
            let count: usize = val.len();

            if count == 1 {
                encodings.insert(k, OrderedFloat::from(prior));
            } else {
                // let encoding = compute_encoding(encoder, &val, target, D::from_usize(count).unwrap(), prior);
                let encoding = compute_encoding(encoder, &val);
                encodings.insert(k, OrderedFloat::from(encoding));
            }
        }

        ColumnTargetEncoder {
            encodings,
            phantom_target: PhantomData,
        }
    }

    /// Encode provided `data`. If you need to transform an `ndarray` see [transform_arr](ColumnTargetEncoder::transform_arr).
    pub fn transform(&self, data: &mut Array1<OrderedFloat<D>>) {
        for x in data.iter_mut() {
            *x = *self.encodings.get(x).unwrap();
        }
    }

    /// Encode provided `data`
    pub fn transform_arr(&self, data: &mut ArrayViewMut1<OrderedFloat<D>>) {
        data.map_mut(|x| *x = *self.encodings.get(x).unwrap());
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
        let x = array![0., 1., 1., 0., 3., 0., 1.];
        let mut x = x.to_ordered_float();
        let y = array![1., 2., 2., 1., 0., 1., 2.];

        let mut encoder = Encoders::new_target_encoder(1, 1.0);
        encoder.compute_stats(&y);

        let encoder = ColumnTargetEncoder::fit(
            &x,
            &y,
            &mut encoder,
        );
        encoder.transform(&mut x);
        // target_encoding(&mut a, &b);
        let expected = array![
            1.0340579777206051,
            1.9148550556984874,
            1.9148550556984874,
            1.0340579777206051,
            1.2857142857142858,
            1.0340579777206051,
            1.9148550556984874,
        ];
        let actual = x.iter().map(|x| x.0).collect_vec();
        expected
            .iter()
            .zip(actual.iter())
            .for_each(|(expected, actual): (&f64, &f64)| {
                assert_approx_eq!(expected, actual);
            });
    }

    #[test]
    fn test_fit_one_category() {
        let mut x = Array2::<f64>::zeros((10, 7)).mapv(OrderedFloat::from);
        let y = array![1., 2., 2., 1., 0., 1., 2.];

        let mut encoder = Encoders::new_target_encoder(1, 1.0);
        encoder.compute_stats(&y);

        let encoder = MatrixEncoder::fit(
            &x,
            &y,
            &mut encoder,
        );
        encoder.transform(&mut x);

        for r in x.iter() {
            assert_approx_eq!(r.0, 1.2857142857142858);
        }
    }

    #[test]
    fn test_fit_compare_category_encoders() {
        let a = array![
            [2., 6., 3., 5., 4.],
            [3., 2., 2., 5., 3.],
            [8., 4., 5., 3., 1.],
            [5., 0., 2., 4., 9.],
            [9., 5., 2., 0., 7.]
        ];

        let mut a = a.mapv(OrderedFloat::from);

        let b = array![0.48263811, 0.16705367, 0.32397016, 0.10172379, 0.54362169];
        let expected = array![
            [0.32380149, 0.32380149, 0.32380149, 0.32456501, 0.32380149],
            [0.32380149, 0.32380149, 0.27711768, 0.32456501, 0.32380149],
            [0.32380149, 0.32380149, 0.32380149, 0.32380149, 0.32380149],
            [0.32380149, 0.32380149, 0.27711768, 0.32380149, 0.32380149],
            [0.32380149, 0.32380149, 0.27711768, 0.32380149, 0.32380149]
        ];

        let encoder = MatrixEncoder::fit(
            &mut a,
            &b,
            &mut Encoders::new_target_encoder(1, 1.0),
        );
        encoder.transform(&mut a);

        Zip::from(&a).and(&expected).apply(|&a, &expected| {
            assert_approx_eq!(a.0, expected);
        });
    }
}
