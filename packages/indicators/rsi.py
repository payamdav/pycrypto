import numba as nb
import numpy as np


@nb.njit(inline='always')
def rsi_1_1(array, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    if n < window + 1:
        return out

    # Seed avg_gain and avg_loss using the first `window` price changes
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, window + 1):
        delta = array[i] - array[i - 1]
        if delta > 0.0:
            avg_gain += delta
        else:
            avg_loss -= delta
    avg_gain /= window
    avg_loss /= window

    # Compute RSI for index `window`
    if avg_loss == 0.0:
        out[window] = 1.0
    else:
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        out[window] = (rsi - 50.0) / 50.0

    # Wilder's smoothing for remaining indices
    alpha = 1.0 / window
    for i in range(window + 1, n):
        delta = array[i] - array[i - 1]
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        avg_gain = avg_gain * (1.0 - alpha) + gain * alpha
        avg_loss = avg_loss * (1.0 - alpha) + loss * alpha
        if avg_loss == 0.0:
            out[i] = 1.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            out[i] = (rsi - 50.0) / 50.0

    return out
