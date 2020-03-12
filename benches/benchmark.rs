use criterion::{Bencher, BenchmarkId, black_box, Criterion, criterion_group, criterion_main, Throughput};
use itertools::Itertools;
use ndarray::Array1;
use num_traits::Num;
use ordered_float::OrderedFloat;
use rand::{Rng, thread_rng};
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::UniformInt;

use blazing_encoders::target_encoder::{TargetEncoder, ColumnTargetEncoder};
use blazing_encoders::utils::{gen_array, gen_array_f32, ToOrderedFloat};

pub fn benchmark_column_target_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("array size");

    for (size, cat_size) in [(10, 3), (100, 3), (1000, 10), (10000, 50), (100000, 100), (1000000, 3000)].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let mut data = gen_array_f32::<i32, _>(*size as usize, &Uniform::new(0, *cat_size));
        let target = gen_array::<f32, _>(*size as usize, &Uniform::new(0.0, 1000.0));
        let data = data.iter().map(|x| OrderedFloat::from(*x)).collect_vec();
        let input = (&data, &target);
        group.bench_with_input(BenchmarkId::from_parameter(size), &input,
                               |b, i| b.iter(|| {
                                   let encoder = ColumnTargetEncoder::fit(&data, &target);
                                   encoder.transform(&mut data.clone());
                               }));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = benchmark_column_target_encoder
}

criterion_main!(benches);
