use ndarray::{ArrayD, ArrayViewD, Array1, Zip};
use num_traits::{Num};
use itertools::Itertools;

fn target_encoding<T>(data: &mut Array1<T>, target: Array1<T>) where T: Num + Ord + Copy {
    let groups = data.iter_mut()
        .zip(&target)
        .sorted_by_key(|x| *x.0)
        .group_by(|x| *x.0);

    for (k, v) in &groups {
        let val: Array1<T> =  v.map(|x| {
            *x.0 = T::one(); // TODO move into anoher map and target encode original features
            *x.1
        }).collect();
        let count = val.len();

        // val.sum() / T::into(count);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Zip};
    use itertools::Itertools;
    use rayon::iter::IntoParallelRefIterator;
    use crate::target_encoding;

    #[test]
    fn it_works() {
        let mut a = Array1::<i32>::from(vec![0, 1, 1, 0, 3, 0, 1]);
        let b = Array1::<i32>::from(vec![1, 2, 2, 1, 0, 1, 2]);
        target_encoding(&mut a, b);
        println!("{:?}", a);
    }
}
