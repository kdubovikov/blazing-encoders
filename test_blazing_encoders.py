import blazing_encoders as be
import numpy as np
import time
from category_encoders.target_encoder import TargetEncoder
import sys

if __name__ == '__main__':
    size = sys.argv[1]
    b = np.random.rand(int(size))
    a = np.random.randint(0, 10000, int(size)).astype('float')

    start = time.time()
    encoding = be.target_encoding(a, b)
    end = time.time() - start
    print("--- %s seconds ---" % end)

    start = time.time()
    te = TargetEncoder(cols=[0])
    encoding = te.fit_transform(a, b)
    end = time.time() - start
    print("--- %s seconds ---" % end)
