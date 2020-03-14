//! A parallel implementation of target encoding algorithm for categorical variables.
//! # About
//! The algorithm and implementation is highly inspored by
//! [category_encoders](https://contrib.scikit-learn.org/categorical-encoding/) Python library.
//! This Rust implementation uses parallel operations to perform target encoding.
//! It is faster and more memory efficient than Python implementation.
//! `blazing_encoders` also present a Python API similar to `category_encoders`.
//!
//! # Installation
//! ## Rust
//! Add `blazing_ecnoders` to your `Cargo.toml`
//! ```toml
//! [dependencies]
//! blazing_encoders = "0.1.0"
//! ```
//!
//! ## Python
//! You can build the Python wheel from source using [maturin](https://github.com/PyO3/maturin):
//! ```bash
//! maturin build --release
//! pip install -U ./target/wheels/wheel_file_name.whl
//! ```
//!
//! # Usage
//! ## Rust API
//! ```rust
//! use numpy::npyffi::array;
//! use blazing_encoders::target_encoder::TargetEncoder;
//! let data = array![[2., 6., 3., 5., 4.],
//!                   [3., 2., 2., 5., 3.],
//!                   [8., 4., 5., 3., 1.],
//!                   [5., 0., 2., 4., 9.],
//!                   [9., 5., 2., 0., 7.]];
//!
//! let mut data = a.mapv(OrderedFloat::from);
//! let target = [0.48263811, 0.16705367, 0.32397016, 0.10172379, 0.54362169];
//!
//! let encoder = TargetEncoder::fit(&mut a, &b);
//! encoder.transform(&mut a);
//! ```
//!
//! ## Python API
//! Currently, the Python API supports only float32 and float64 data and targets,
//! so you might need to convert your matrices before the encoding.
//!
//! ```python
//! import blazing_encoders as be
//! import numpy as np
//!
//! data = np.random.randint(0, 10, (5,5))
//! target = np.random.rand(5)
//!
//! encoder = be.TargetEncoder_f64.fit(data, target) # you can use TargetEncoder_f32 for float32 data
//! encoded_data = encoder.transform(a)
//! ```
//!
//! # Limitations
//! * Currently, only the basic target encoding is implemented
//! * The library supports only continuous target variables. As a result, it can be used only in regression settings.
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
