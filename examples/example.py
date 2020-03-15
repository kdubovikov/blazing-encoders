from blazing_encoders import TargetEncoder_f64
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.randint(0, 10, (5, 5)).astype('float')
    target = np.random.rand(5)

    encoder = TargetEncoder_f64.fit(data, target, smoothing=1.0, min_samples_leaf=1)
    encoded_data = encoder.transform(data)
    print(encoded_data)
