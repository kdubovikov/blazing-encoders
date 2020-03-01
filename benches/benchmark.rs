use criterion::{BenchmarkId, black_box, Criterion, criterion_group, criterion_main, Throughput};
use ndarray::Array1;
use num_traits::Num;

use blazing_encoders::{target_encoding, gen_array};
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::UniformInt;


pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("array size");
    const TARGET_SIZE: usize = 10000000;

    for size in [10, 100, 1000, 100000, 1000000, 10000000].iter() {
        group.throughput(Throughput::Elements(*size));
        let data = gen_array::<i32, _>(*size as usize, &Uniform::new(0, 3000));
        let target = gen_array::<f32, _>(TARGET_SIZE, &Uniform::new(0.0, 1000.0));
        let input = (&data, &target);
        group.bench_with_input(BenchmarkId::from_parameter(size), &input,
                           |b, &i| b.iter(|| target_encoding(i.0, i.1)));
    }
    group.finish();

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
