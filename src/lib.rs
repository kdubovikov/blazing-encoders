mod target_encoder;
mod utils;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Dim, Axis};
use numpy::{IntoPyArray, PyArrayDyn, PyArray, PyArray2, PyArray1};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use target_encoder::target_encoding;
use utils::ToOrderedFloat;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use crate::target_encoder::par_column_target_encoding;

#[pymodule]
fn blazing_encoders(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "target_encoding")]
    fn target_encoding_py(
        py: Python,
        data: &PyArrayDyn<f64>,
        target: &PyArrayDyn<f64>,
    ) -> Py<PyArray<f64, Dim<[usize; 1]>>> {
        let mut data = data.as_slice_mut().unwrap();
        let mut data = data.iter().map(|x| OrderedFloat::from(*x)).collect_vec();
        let target = target.as_slice().unwrap();
        target_encoding(&mut data, target);
        let mut d = data.iter().map(|x| x.0).collect_vec();
        d.into_pyarray(py).to_owned()
    }

    #[pyfn(m, "par_column_target_encoding")]
    fn par_column_target_encoding_py(py: Python, data: &PyArray2<f64>, target: &PyArray1<f64>) -> Py<PyArray2<f64>> {
        let data = data.as_array_mut();
        let target = target.as_slice().unwrap();
        let result = par_column_target_encoding(&data.to_owned(), &target).unwrap();
        result.into_pyarray(py).to_owned()
    }
    Ok(())
}
