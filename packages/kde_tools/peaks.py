"""KDE peak finding (scipy-based).

Reproduces the notebook's ``top_kde_peaks`` and the above/below split
(look_back_look_ahead.ipynb, cell 6). ``find_peaks`` / ``peak_prominences``
stay in scipy so their semantics match the notebook bit-for-bit.
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences


def top_kde_peaks(
    kde_series: np.ndarray,
    prices: np.ndarray,
    distance: float,
    n: int = 3,
):
    """Return the ``n`` highest-prominence peaks of ``kde_series``.

    Parameters
    ----------
    kde_series : np.ndarray
        KDE values to find peaks in.
    prices : np.ndarray
        Bin-center prices aligned with ``kde_series``.
    distance : float
        Minimum horizontal distance between peaks (``find_peaks`` ``distance``).
    n : int
        Maximum number of peaks to return (top-n by descending prominence).

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(peak_prices, peak_proms)``; both empty when no peaks are found.
    """
    peaks, _ = find_peaks(kde_series, distance=distance)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    proms = peak_prominences(kde_series, peaks)[0]
    order = np.argsort(proms)[::-1][:n]
    return prices[peaks[order]], proms[order]


def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    distance: float = 5,
    n: int = 3,
    split_at: float = 0.0,
):
    """Top-``n`` KDE peaks above and below the current price.

    ``split_at`` is the current price in normalized space (defaults to ``0.0``
    because ``price_l`` normalizes to ``0.0``). "Above" = ``bin_centers >=
    split_at``, "below" = ``bin_centers < split_at`` (matching the notebook's
    ``pos_mask`` / ``neg_mask``).

    Returns
    -------
    dict
        ``{"above_prices", "above_proms", "below_prices", "below_proms"}``.
    """
    pos_mask = bin_centers >= split_at
    neg_mask = bin_centers < split_at

    pos_kde = kde[pos_mask]
    pos_prices = bin_centers[pos_mask]
    neg_kde = kde[neg_mask]
    neg_prices = bin_centers[neg_mask]

    above_prices, above_proms = top_kde_peaks(pos_kde, pos_prices, distance=distance, n=n)
    below_prices, below_proms = top_kde_peaks(neg_kde, neg_prices, distance=distance, n=n)

    return {
        "above_prices": above_prices,
        "above_proms": above_proms,
        "below_prices": below_prices,
        "below_proms": below_proms,
    }
