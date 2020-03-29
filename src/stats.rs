use num_traits::{Float, FromPrimitive};
use std::iter::Sum;
use ndarray::{Array1};

trait SummaryStatistics<T: Float + Sum + FromPrimitive> {
    // fn mean(&self) -> Option<T>;
    fn var(&self, ddof: T) -> T;
    fn std(&self, ddof: T) -> T;
}

impl<T: Float + Sum + FromPrimitive> SummaryStatistics<T> for Array1<T> {
    fn var(&self, ddof: T) -> T { 
        let zero = T::from_usize(0).expect("Converting 0 to `T` must not fail.");
        let n = T::from_usize(self.len()).expect("Converting length to `T` must not fail.");
        assert!(
            !(ddof < zero || ddof > n),
            "`ddof` must not be less than zero or greater than the length of \
             the axis",
        );
        let dof = n - ddof;
        let mut mean = T::zero();
        let mut sum_sq = T::zero();
        for (i, &x) in self.into_iter().enumerate() {
            let count = T::from_usize(i + 1).expect("Converting index to `T` must not fail.");
            let delta = x - mean;
            mean = mean + delta / count;
            sum_sq = (x - mean).mul_add(delta, sum_sq);
        }
        sum_sq / dof
    }

    fn std(&self, ddof: T) -> T { 
        self.var(ddof).sqrt()
     }
}


#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;
    use ndarray::array;

    #[test]
    fn test_var() {
        let data: Array1<_> = array![1., -4.32, 1.14, 0.32];
        let var = data.var(1.);
        assert_approx_eq!(var, 6.7331, 1e-4);
    }

    #[test]
    fn test_std() {
        let a = array![1., -4.32, 1.14, 0.32];
        let stddev = a.std(1.);
       assert_approx_eq!(stddev, 2.59483, 1e-4);
    }
}
