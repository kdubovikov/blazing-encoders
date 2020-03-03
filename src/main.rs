pub mod lib;
use blazing_encoders::{gen_array, target_encoding};
use rand::distributions::Uniform;

fn main() {
    let data = gen_array::<i32, _>(15000000, &Uniform::new(0, 3000));
    let target = gen_array::<f32, _>(15000000, &Uniform::new(0.0, 1000.0));
    let encodings = target_encoding(&data, &target);
    println!("{:?}", encodings)
}