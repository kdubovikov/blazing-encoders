use num_traits::{Float, Num, AsPrimitive};
use ordered_float::OrderedFloat;
use rand::prelude::Distribution;
use rand::thread_rng;
use rand::Rng;
use itertools::Itertools;

pub trait ToOrderedFloat<T>: where T: Float {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<T>>;
}

impl ToOrderedFloat<f32> for Vec<f32> {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<f32>> {
        self.iter().map(|x| OrderedFloat::<f32>::from(*x)).collect_vec()
    }
}

impl ToOrderedFloat<f64> for Vec<f64> {
    fn to_ordered_float(&self) -> Vec<OrderedFloat<f64>> {
        self.iter().map(|x| OrderedFloat::<f64>::from(*x)).collect_vec()
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

