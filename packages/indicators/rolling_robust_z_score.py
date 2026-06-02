import numba as nb
import numpy as np


@nb.njit(inline='always')
def rolling_robust_z_score(array, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)

    if n < window:
        return out

    # Allocate sorted buffer once
    buf = np.empty(window, dtype=np.float64)

    # --- First valid window: copy and insertion-sort ---
    for k in range(window):
        buf[k] = array[k]

    for k in range(1, window):
        key = buf[k]
        j = k - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = key

    # Compute and store result for i == window - 1
    i = window - 1
    if window % 2 == 1:
        median = buf[window // 2]
    else:
        median = (buf[window // 2 - 1] + buf[window // 2]) / 2.0
    q1 = buf[window // 4]
    q3 = buf[3 * window // 4]
    iqr = q3 - q1
    if iqr == 0.0:
        out[i] = 0.0
    else:
        out[i] = (array[i] - median) / iqr

    # --- Subsequent windows: incremental update ---
    for i in range(window, n):
        old_val = array[i - window]
        new_val = array[i]

        # Find old_val in sorted buffer (linear scan)
        k = 0
        while k < window and buf[k] != old_val:
            k += 1

        # Overwrite with new value and bubble into sorted position
        buf[k] = new_val
        while k > 0 and buf[k] < buf[k - 1]:
            tmp = buf[k]
            buf[k] = buf[k - 1]
            buf[k - 1] = tmp
            k -= 1
        while k < window - 1 and buf[k] > buf[k + 1]:
            tmp = buf[k]
            buf[k] = buf[k + 1]
            buf[k + 1] = tmp
            k += 1

        if window % 2 == 1:
            median = buf[window // 2]
        else:
            median = (buf[window // 2 - 1] + buf[window // 2]) / 2.0
        q1 = buf[window // 4]
        q3 = buf[3 * window // 4]
        iqr = q3 - q1
        if iqr == 0.0:
            out[i] = 0.0
        else:
            out[i] = (array[i] - median) / iqr

    return out
