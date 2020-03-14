use itertools::Itertools;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayDyn};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use ndarray::Dim;

use crate::target_encoder::{ColumnTargetEncoder, TargetEncoder};

pub mod target_encoder;
pub mod utils;

macro_rules! create_target_encoder_class {
    ($name:ident, $type:ty) => {
        #[allow(non_camel_case_types)]
        #[pyclass]
        struct $name {
            encoder: TargetEncoder<$type, $type>
        }

        #[pymethods]
        impl $name {
            #[staticmethod]
            fn fit(py: Python, data: &PyArray2<$type>, target: &PyArray1<$type>) -> Self {
                let data = data.as_array_mut().mapv::<OrderedFloat<$type>, _>(OrderedFloat::from);
                let target = target.as_slice().unwrap();
                let encoder = py.allow_threads(move || {
                    TargetEncoder::fit(&data, target)
                });

                $name { encoder }
            }

            fn transform(&self, py: Python, data: &PyArray2<$type>) -> Py<PyArray2<$type>> {
                let mut data = data.as_array_mut().mapv::<OrderedFloat<$type>, _>(OrderedFloat::from);
                self.encoder.transform(&mut data);
                Array2::from(data).map(|x| x.0).into_pyarray(py).to_owned()
            }
        }

    }
}

create_target_encoder_class!(TargetEncoder_f64, f64);
create_target_encoder_class!(TargetEncoder_f32, f32);

#[pymodule]
fn blazing_encoders(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TargetEncoder_f64>().expect("Error adding class to python module");
    m.add_class::<TargetEncoder_f32>().expect("Error adding class to python module");

    #[pyfn(m, "target_encoding")]
    fn target_encoding_py(
        py: Python,
        data: &PyArrayDyn<f64>,
        target: &PyArrayDyn<f64>,
    ) -> Py<PyArray<f64, Dim<[usize; 1]>>> {
        let data = data.as_slice_mut().unwrap();
        let mut data = data.iter().map(|x| OrderedFloat::from(*x)).collect_vec();
        let target = target.as_slice().unwrap();

        let encoder = ColumnTargetEncoder::fit(&data, target);
        encoder.transform(&mut data);
        let d = data.iter().map(|x| x.0).collect_vec();
        d.into_pyarray(py).to_owned()
    }

    #[pyfn(m, "par_column_target_encoding")]
    fn par_column_target_encoding_py(py: Python, data: &PyArray2<f64>, target: &PyArray1<f64>) -> Py<PyArray2<f64>> {
        let mut data = data.as_array_mut().mapv::<OrderedFloat<f64>, _>(OrderedFloat::from);
        let target = target.as_slice().unwrap();
        let data = py.allow_threads(move || {
            let encoder = TargetEncoder::fit(&data, target);
            encoder.transform(&mut data);
            data
        });
        Array2::from(data).map(|x| x.0).into_pyarray(py).to_owned()
    }
    Ok(())
}
