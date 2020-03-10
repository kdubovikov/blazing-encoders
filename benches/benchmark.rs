use criterion::{Bencher, BenchmarkId, black_box, Criterion, criterion_group, criterion_main, Throughput};
use itertools::Itertools;
use ndarray::Array1;
use num_traits::Num;
use ordered_float::OrderedFloat;
use rand::{Rng, thread_rng};
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::UniformInt;

use blazing_encoders::target_encoder::target_encoding;
use blazing_encoders::utils::{gen_array, gen_array_f32, ToOrderedFloat};

pub fn benchmark_cat_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("array size");
    const CAT_SIZE: usize = 3000;

    for size in [10, 100, 1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let mut data = gen_array_f32::<i32, _>(size as usize, &Uniform::new(0, CAT_SIZE));
        let target = gen_array::<f32, _>(size as usize, &Uniform::new(0.0, 1000.0));
        let data = data.iter().map(|x| OrderedFloat::from(*x)).collect_vec();
        let input = (&data, &target);
        group.bench_with_input(BenchmarkId::from_parameter(size), &input,
                               |b, i| b.iter(|| {
                                   target_encoding(&mut data.clone(), i.1)
                               }));
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

criterion_group!(benches, benchmark, benchmark_cat_size);
criterion_main!(benches);
