"""Kernel construction for volume-weighted KDE smoothing.

Reproduces the notebook's `make_kernel` (look_back_look_ahead.ipynb, cell 5).
The numeric core runs jitted with an explicit loop; the public `make_kernel`
is a thin Python wrapper that maps the kernel-type string to an integer code.
"""

import numpy as np
import numba as nb

# Kernel type codes used by the jitted core.
_TRIANGULAR = 0
_EPANECHNIKOV = 1
_UNIFORM = 2

_KERNEL_CODES = {
    "Triangular": _TRIANGULAR,
    "Epanechnikov": _EPANECHNIKOV,
    "Uniform": _UNIFORM,
}


@nb.njit
def _make_kernel_core(kernel_code: int, bandwidth: int) -> np.ndarray:
    """Build and normalize the kernel for a given integer code and bandwidth.

    x runs over arange(-bandwidth, bandwidth + 1), length 2*bandwidth + 1.
    The output is normalized so it sums to 1.0.
    """
    n = 2 * bandwidth + 1
    k = np.empty(n, dtype=np.float64)
    bw = float(bandwidth)

    for i in range(n):
        x = float(i - bandwidth)
        if kernel_code == _TRIANGULAR:
            val = 1.0 - abs(x) / bw
            if val < 0.0:
                val = 0.0
        elif kernel_code == _EPANECHNIKOV:
            ratio = x / bw
            val = 1.0 - ratio * ratio
            if val < 0.0:
                val = 0.0
        else:  # _UNIFORM
            val = 1.0
        k[i] = val

    total = 0.0
    for i in range(n):
        total += k[i]
    for i in range(n):
        k[i] = k[i] / total

    return k


def make_kernel(kernel_type: str = "Triangular", bandwidth: int = 5) -> np.ndarray:
    """Return a normalized 1-D kernel of length ``2 * bandwidth + 1``.

    Parameters
    ----------
    kernel_type : {"Triangular", "Epanechnikov", "Uniform"}
        Kernel shape over ``x = arange(-bandwidth, bandwidth + 1)``:
          - "Triangular"   -> max(1 - |x| / bandwidth, 0)
          - "Epanechnikov" -> max(1 - (x / bandwidth) ** 2, 0)
          - "Uniform"      -> all ones
    bandwidth : int
        Half-width of the kernel (>= 1).

    Returns
    -------
    np.ndarray
        Newly allocated ``np.float64`` array of length ``2 * bandwidth + 1``,
        normalized so it sums to ``1.0``.

    Raises
    ------
    ValueError
        If ``kernel_type`` is not one of the three supported kernels.
    """
    if kernel_type not in _KERNEL_CODES:
        raise ValueError(f"Unknown kernel: {kernel_type!r}")
    return _make_kernel_core(_KERNEL_CODES[kernel_type], int(bandwidth))
