import blazing_encoders as be
import numpy as np
import time
from category_encoders.target_encoder import TargetEncoder
import argparse


def compare_encoders(a, b, cols, blazing_encoder_fun, flatten_te=False):
    start = time.time()
    encoding_be = blazing_encoder_fun(a, b)
    time_blazing = time.time() - start
    print(f"{'blazing_encoders':<20}{time_blazing:>15.3f}")

    start = time.time()
    te = TargetEncoder(cols=cols)
    encoding_te = te.fit_transform(a, b).to_numpy()
    if flatten_te:
        encoding_te = encoding_te.reshape(-1)

    time_category_encoders = time.time() - start
    print(f"{'category_encoders':<20}{time_category_encoders:>15.3f}")

    speed_ratio = time_blazing / time_category_encoders
    if speed_ratio < 1:
        print(f"blazing encoders are {1 / speed_ratio:.2f} times faster ⬆️")
    else:
        print(f"blazing encoders are {speed_ratio:.2f} times slower ⬇️")

    print("Checking that blazing_encoders and category_encoders results match")
    assert np.allclose(encoding_te, encoding_be)
    print("Results match 👍🏻")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="blazing_encoders benchmark")
    parser.add_argument('n_rows', type=int, help="number of rows to benchmark on")
    parser.add_argument('n_cols', type=int, help="number of columns to benchmark on")
    parser.add_argument('n_cats', type=int, help="number of categories to benchmark on")
    args = parser.parse_args()

    print("‍Benchmarking target encoding for 1 column 💨")
    a = np.random.randint(0, args.n_cats, args.n_rows).astype('float')
    b = np.random.rand(args.n_rows)

    compare_encoders(a, b, [0], be.target_encoding, flatten_te=True)

    print(f"Benchmarking target encoding for {args.n_cols} columns 💨")
    matrix_size = (args.n_rows, args.n_cols)
    a = np.random.randint(0, args.n_cats, matrix_size).astype('float')
    b = np.random.rand(args.n_rows)

    compare_encoders(a, b, range(0, args.n_cols), be.par_column_target_encoding)


