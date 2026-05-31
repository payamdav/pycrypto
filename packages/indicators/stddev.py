import numba as nb
import numpy as np


@nb.njit(inline='always')
def stddev(array, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window - 1, n):
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += array[j]
        mean /= window
        var = 0.0
        for j in range(i - window + 1, i + 1):
            diff = array[j] - mean
            var += diff * diff
        var /= window
        out[i] = var ** 0.5
    return out
