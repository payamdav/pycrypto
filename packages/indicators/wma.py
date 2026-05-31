import numba as nb
import numpy as np


@nb.njit(inline='always')
def wma(array, weights, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    w_sum = 0.0
    for k in range(window):
        w_sum += weights[k]
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += array[i - window + 1 + j] * weights[j]
        out[i] = s / w_sum
    return out
