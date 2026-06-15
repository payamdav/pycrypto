"""KDE tools: volume-weighted KDE construction and peak finding.

Reproduces the KDE construction and peak-finding logic from
notebooks/tests/look_back_look_ahead.ipynb (cells 5 and 6).
"""

from packages.kde_tools.kernels import make_kernel
from packages.kde_tools.histogram import weighted_histogram
from packages.kde_tools.kde import convolve_same, compute_kde
from packages.kde_tools.peaks import top_kde_peaks, kde_peaks_above_below

__all__ = [
    "make_kernel",
    "weighted_histogram",
    "convolve_same",
    "compute_kde",
    "top_kde_peaks",
    "kde_peaks_above_below",
]
