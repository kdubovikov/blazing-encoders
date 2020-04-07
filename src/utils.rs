use itertools::Itertools;
use ndarray::{Array2, Array1};
use num_traits::{AsPrimitive, Float, Num};
use ordered_float::OrderedFloat;
use rand::prelude::Distribution;
use rand::Rng;
use rand::thread_rng;

pub trait ToOrderedFloat<T>: where T: Float {
    /// Covert a vector of floats to a vector of OrderedFloat
    fn to_ordered_float(&self) -> Array1<OrderedFloat<T>>;
}

impl ToOrderedFloat<f32> for Array1<f32> {
    /// Covert a vector of floats to a vector of OrderedFloat
    /// # Example
    /// ```
    /// use blazing_encoders::utils::ToOrderedFloat;
    /// use ordered_float::OrderedFloat;
    /// use ndarray::{Array1, array};
    /// let v = array![1., 2., 3.];
    /// let v_ordered: Array1<OrderedFloat<f32>> = v.to_ordered_float();
    /// ```
    fn to_ordered_float(&self) -> Array1<OrderedFloat<f32>> {
        self.map(|x| OrderedFloat(*x))
    }
}

impl ToOrderedFloat<f64> for Array1<f64> {
    /// Covert a vector of floats to a vector of OrderedFloat
    /// # Example
    /// ```
    /// use blazing_encoders::utils::ToOrderedFloat;
    /// use ordered_float::OrderedFloat;
    /// use ndarray::{Array1, array};
    /// use numpy::npyffi::array;
    /// let v = array![1., 2., 3.];
    /// let v_ordered: Array1<OrderedFloat<f64>> = v.to_ordered_float();
    /// ```
    fn to_ordered_float(&self) -> Array1<OrderedFloat<f64>> {
        self.map(|x| OrderedFloat(*x))
    }
}

pub fn gen_array<T, D>(n: usize, distr: &D) -> Vec<T>
    where
        T: Num + Clone,
        D: Distribution<T> {
    let mut random = thread_rng();
    (0..n).map(|_| {
        random.sample(distr)
    }).collect()
}

/// Generate an array using random elements from distribution `distr`
/// # Example
/// ```
/// # use blazing_encoders::utils::gen_array_f32;
/// use rand::distributions::Uniform;
/// let arr = gen_array_f32(10, &Uniform::new(0, 10));
/// # assert_eq!(arr.len(), 10)
/// ```
pub fn gen_array_f32<T, D>(n: usize, distr: &D) -> Vec<f32>
    where
        T: Num + Clone + AsPrimitive<f32>,
        D: Distribution<T> {
    gen_array(n, distr).iter().map(|x| x.as_()).collect_vec()
}


/// Generate an [`Array2`](ndarray::Array2) matrix using random elements from distribution `distr`
/// # Example
/// ```
/// # use blazing_encoders::utils::gen_array2_f32;
/// use rand::distributions::Uniform;
/// let arr = gen_array2_f32((5, 5), &Uniform::new(0, 10));
/// # assert_eq!(arr.shape(), &[5, 5])
/// ```
pub fn gen_array2_f32<T, D>(shape: (usize, usize), distr: &D) -> Array2<f32>
    where
        T: Num + Clone + AsPrimitive<f32>,
        D: Distribution<T> {
    let mut random = thread_rng();
    Array2::from_shape_fn(shape, |_| random.sample(distr).as_())
}

