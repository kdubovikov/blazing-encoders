use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, s, Zip};
use num_traits::{FromPrimitive, Num};

pub fn target_encoding<D, T>(data: &Array1<D>, target: &Array1<T>) -> Array1<T>
    where
        T: Num + Copy + FromPrimitive,
        D: Num + Copy + PartialOrd + Eq + Hash {

    // group targets by each item in data
    let groups = data.iter()
        .zip(target)
        .sorted_by(|a, b| {
            let idx = *a.0;
            idx.partial_cmp(b.0).unwrap()
        })
        .group_by(|x| *x.0);

    // calculate target encoding for each value in data
    let mut encodings = HashMap::new();
    for (k, v) in &groups {
        let val: Array1<T> =  v.map(|x| {
            *x.1
        }).collect();
        let count = val.len();
        let encoding: T = T::from(val.sum() / T::from_usize(count).unwrap());
        encodings.insert(k, encoding);
    }

    // create encoded array
    let data_len = data.len();
    let mut result = Array1::<T>::from_shape_fn(data_len, |idx| {
       *encodings.get(&data[idx]).unwrap()
    });

    return result;
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use ndarray::{Array1, Zip};

    use crate::target_encoding;

    #[test]
    fn test_target_encoding() {
        let a = Array1::<i32>::from(vec![0, 1, 1, 0, 3, 0, 1]);
        let b = Array1::<f32>::from(vec![1., 2., 2., 1., 0., 1., 2.]);
        let encodings = target_encoding( &a, &b);
        let expected = Array1::from(vec![1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
        assert_eq!(expected, encodings);
    }
}
