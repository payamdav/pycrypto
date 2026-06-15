"""Volume-weighted histogram over a fixed range (numba-jitted).

Reproduces ``np.histogram(values, bins=bins, range=(range_min, range_max),
weights=weights)[0]`` with explicit loops, per the KDE building block in
look_back_look_ahead.ipynb (cell 5).
"""

import numpy as np
import numba as nb


@nb.njit
def weighted_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int = 200,
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> np.ndarray:
    """Weighted histogram of ``values`` over ``[range_min, range_max]``.

    Reproduces ``np.histogram(values, bins=bins, range=(range_min, range_max),
    weights=weights)[0]``.

    For each (v, w): values outside ``[range_min, range_max]`` are skipped;
    otherwise ``idx = int((v - range_min) / bin_width)`` and the boundary case
    ``v == range_max`` (where ``idx == bins``) is clamped to ``bins - 1``.

    Returns
    -------
    np.ndarray
        Newly allocated ``np.float64`` array of length ``bins`` holding the
        summed weights per bin.
    """
    counts = np.zeros(bins, dtype=np.float64)
    bin_width = (range_max - range_min) / bins

    for i in range(values.shape[0]):
        v = values[i]
        if v < range_min or v > range_max:
            continue
        idx = int((v - range_min) / bin_width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += weights[i]

    return counts
