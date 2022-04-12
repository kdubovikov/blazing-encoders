# 🔥 blazing-encoders
![build](https://github.com/kdubovikov/blazing-encoders/workflows/Rust/badge.svg)

Blazing-fast categorical feature encoding.

This is a Python library written in Rust that allows you to encode categorical variables using smoothed mean target. In particular, this algorithm is used: [A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems](https://dl.acm.org/citation.cfm?id=507538).

## Installation
`pip install blazing_encoders`

## Usage

```Python
from blazing_encoders import TargetEncoder_f64
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.randint(0, 10, (5, 5)).astype('float')
    target = np.random.rand(5)

    encoder = TargetEncoder_f64.fit(data, target, smoothing=1.0, min_samples_leaf=1)
    encoded_data = encoder.transform(data)
    print(encoded_data)
```

You can use two of the available classes: `TargetEncoder_f64`, and `TargetEncoder_f32` to control the balance between memory usage and numerical precision of your target encoding process.

Underneath, the library will share as much memory as possible so that overhead should be minimal. Also, it will parallelize target encoding computation so that the overall process will complete much faster.

## Documentation and examples
You can find out how to use the library [here](https://github.com/kdubovikov/blazing-encoders/blob/master/examples/example.py). Also, you can read more at [this blog post](https://blog.kdubovikov.ml/articles/datascience/blazing-encoders).
