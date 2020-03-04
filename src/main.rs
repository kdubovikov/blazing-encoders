pub mod lib;
use blazing_encoders::{gen_array_f32, gen_array, target_encoding};
use rand::distributions::Uniform;
use crate::lib::ToOrderedFloat;
use std::time::Instant;

fn main() {
    let data: Vec<f32> = gen_array_f32::<i32, _>(15000000, &Uniform::new(0, 3000));
    let mut data = data.to_ordered_float();
    let target = gen_array::<f32, _>(15000000, &Uniform::new(0., 1000.));

    let now = Instant::now();
    target_encoding(&mut data, &target);
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}