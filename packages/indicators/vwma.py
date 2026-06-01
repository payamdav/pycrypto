import numba as nb
import numpy as np


@nb.njit(inline='always')
def vwma(array, volume, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window - 1, n):
        s = 0.0
        v_sum = 0.0
        for j in range(i - window + 1, i + 1):
            s += array[j] * volume[j]
            v_sum += volume[j]
        out[i] = s / v_sum
    return out
