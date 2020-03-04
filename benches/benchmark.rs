use criterion::{BenchmarkId, black_box, Criterion, criterion_group, criterion_main, Throughput, Bencher};
use ndarray::Array1;
use num_traits::Num;

use blazing_encoders::{target_encoding, gen_array, gen_array_f32};
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::UniformInt;
use ordered_float::OrderedFloat;
use itertools::Itertools;
use blazing_encoders::ToOrderedFloat;


pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("array size");
    const TARGET_SIZE: usize = 1000000;

    for size in [10, 100, 1000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let mut data = gen_array_f32::<i32, _>(TARGET_SIZE, &Uniform::new(0, *size));
        let target = gen_array::<f32, _>(TARGET_SIZE, &Uniform::new(0.0, 1000.0));
        let data = data.iter().map(|x| OrderedFloat::from(*x)).collect_vec();
        let input = (&data, &target);
        // group.bench_with_input(BenchmarkId::from_parameter(size), &input,
        //                    |b, i| b.iter(|| {
        //                        target_encoding(&data, i.1)
        //                    }));
    }
    group.finish();

}

pub fn benchmark(c: &mut Criterion) {
    const SIZE: usize = 100000;
    let mut data = gen_array_f32::<i32, _>(SIZE, &Uniform::new(0, 3000));
    let mut data = data.iter().map(|x| OrderedFloat::<f32>::from(*x)).collect_vec();
    let target = gen_array::<f32, _>(SIZE, &Uniform::new(0.0, 1000.0));
    c.bench_function("target encoding",
                     |b|
                         b.iter(|| target_encoding(&mut black_box(data.clone()), black_box(target.as_slice()))));
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
