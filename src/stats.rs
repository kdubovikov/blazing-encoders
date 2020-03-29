use num_traits::{Float, FromPrimitive};
use std::iter::Sum;
use ndarray::{Array1};

// trait StatFloat = Float + Sum + FromPrimitive;

trait SummaryStatistics<T: Float + Sum + FromPrimitive> {
    // fn mean(&self) -> Option<T>;
    fn var(&self, ddof: T) -> T;
    fn std(&self, ddof: T) -> T;
}

// impl<T: Float + Sum + FromPrimitive> SummaryStatistics<T> for Vec<T> {
//     fn mean(&self) -> Option<T> {
//         let sum: T = self.iter().map(|&x| x).sum();
//         let count = T::from_usize(self.len()).unwrap();

//         match count {
//             positive if positive > T::zero() => Some(sum / count),
//             _ => None,
//         }
//     }

//     fn var(&self, ddof: T) -> Option<T> {
//         match (Self::mean(self), self.len()) {
//             (Some(data_mean), count) if count > 0 => {
//                 let variance = self.iter().map(|value| {
//                     let diff = data_mean - (*value);
//                     diff * diff
//                 }).sum::<T>() / T::from_usize(count).unwrap();
//                 Some(variance)
//             },
//             _ => None
//         }
//     }

//     fn std(&self, ddof: T) -> Option<T> {
//         match Self::var(self, ddof) {
//             Some(variance) => {
//                 Some(variance.sqrt())
//             },
//             _ => None
//         }
//     }
// }

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

    // #[test]
    // fn test_mean() {
    //     let data = vec![0., 1., 1.25, 52.];
    //     let mean = data.mean().unwrap();
    //     assert_approx_eq!(mean, 13.5625);
    // }

    // #[test]
    // fn test_std() {
    //     let data = vec![0., 1., 1.25, 52.];
    //     let std = data.std(0.).unwrap();
    //     assert_approx_eq!(std, 22.196829024660257)
    // }

    // #[test]
    // fn test_var() {
    //     let data = vec![0., 1., 1.25, 52.];
    //     let var = data.var(0.).unwrap();
    //     assert_approx_eq!(var, 492.69921875);
    // }
}
