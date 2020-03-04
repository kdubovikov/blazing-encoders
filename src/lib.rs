mod target_encoder;
mod utils;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Dim};
use numpy::{IntoPyArray, PyArrayDyn, PyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use target_encoder::target_encoding;
use utils::ToOrderedFloat;
use itertools::Itertools;

#[pymodule]
fn blazing_encoders(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "target_encoding")]
    fn target_encoding_py(
        py: Python,
        data: &PyArrayDyn<f64>,
        target: &PyArrayDyn<f64>,
    ) -> Py<PyArray<f64, Dim<[usize; 1]>>> {
        let mut data = data.as_slice_mut().unwrap();
        let target = target.as_slice().unwrap();
        let mut d = Vec::from(data).to_ordered_float();
        target_encoding(&mut d, target);
        let mut d = d.iter().map(|x| x.0).collect_vec();
        d.into_pyarray(py).to_owned()
    }
    Ok(())
}
