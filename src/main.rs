pub mod lib;
use blazing_encoders::{gen_array, target_encoding};
use rand::distributions::Uniform;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use crate::lib::ToOrderedFloat;

fn main() {
    let data = gen_array::<f32, _>(1000000, &Uniform::new(0., 3000.));
    let mut data = data.to_ordered_float();
    let target = gen_array::<f32, _>(1000000, &Uniform::new(0., 1000.));
    target_encoding(&mut data, &target);
}