pub mod target_encoder;
pub mod utils;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Dim, Axis, Array2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray, PyArray2, PyArray1};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use utils::ToOrderedFloat;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use crate::target_encoder::TargetEncoder;

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

        let encoder = TargetEncoder::fit_one_column(&data, target);
        encoder.transform_vec(&mut data);
        let mut d = data.iter().map(|x| x.0).collect_vec();
        d.into_pyarray(py).to_owned()
    }

    #[pyfn(m, "par_column_target_encoding")]
    fn par_column_target_encoding_py(py: Python, data: &PyArray2<f64>, target: &PyArray1<f64>) -> Py<PyArray2<f64>> {
        let mut data = data.as_array_mut().mapv::<OrderedFloat<f64>, _>(OrderedFloat::from);
        let target = target.as_slice().unwrap();
        let data = py.allow_threads(move || {
            let encoder = TargetEncoder::fit(&data, target);
            encoder.transform_mat(&mut data);
            data
        });
        Array2::from(data).map(|x| x.0).into_pyarray(py).to_owned()
    }
    Ok(())
}
