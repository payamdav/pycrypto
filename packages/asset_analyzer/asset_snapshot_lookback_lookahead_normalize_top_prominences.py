"""Look-back/look-ahead normalize variant with KDE top-prominence peak features.

Builds the existing 60-element ``fl`` vector via the njit
``asset_snapshot_lookback_lookahead_normalize_prepare`` (reused unchanged) and
appends 6 KDE top-prominence peak features at indices 60-65, producing a
66-element vector.

The 6 features are the normalized-vwap price positions of the top-3
highest-prominence KDE peaks above the current price (indices 60-62, default
``+1.0``) and the top-3 below (indices 63-65, default ``-1.0``), computed via
``packages.kde_tools.compute_kde`` + ``kde_peaks_above_below``.

This file is plain Python (NOT njit): ``kde_peaks_above_below`` relies on scipy
(``find_peaks`` / ``peak_prominences``), which cannot run inside ``@nb.njit`` /
``nb.prange``. The heavy inner pieces stay jitted internally (the 60-vector and
the KDE histogram/convolution).
"""

import numpy as np

from .asset_snapshot_lookback_lookahead_normalize import asset_snapshot_lookback_lookahead_normalize_prepare
from packages.kde_tools import compute_kde, kde_peaks_above_below


def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare(
    arr: np.ndarray,
    look_back: int,
    look_ahead: int,
    k_scaler: float,
    kde_bins: int = 200,
    kde_bandwidth: int = 5,
    kde_kernel: str = "Triangular",
    kde_ignore_borders: bool = True,
    kde_peak_distance: float = 5,
) -> np.ndarray:
    """Build the 66-element fl vector (60 base + 6 KDE peak features).

    ``arr`` is a slice of shape ``(look_back + look_ahead, 12)``. Indices 0-59
    are produced by the existing njit prepare function and remain byte-for-byte
    identical; indices 60-65 hold the KDE top-prominence peak prices.
    """
    # 1. Reuse the existing njit 60-vector unchanged.
    base = asset_snapshot_lookback_lookahead_normalize_prepare(arr, look_back, look_ahead, k_scaler)

    # 2. Allocate the enlarged 66-wide vector and copy the base.
    fl = np.zeros(66, dtype=np.float64)
    fl[:60] = base

    # 3. KDE input construction over the look-back slice (see spec 3.1).
    price_l = arr[look_back - 1, 4]                                   # vwap anchor (price_l)
    vwap_lb = arr[:look_back, 4]
    scaled_lb = np.clip(k_scaler * (vwap_lb - price_l) / price_l, -1.0, 1.0)

    v_lb = arr[:look_back, 2]                                         # volume column
    v_lb_mean = v_lb.mean()
    v_lb_norm = v_lb / v_lb_mean if v_lb_mean > 0 else v_lb           # daily-avg-normalized weights

    # 4. KDE + above/below top-3 peaks via kde_tools (see spec 3.2).
    kde_res = compute_kde(
        scaled_lb,
        v_lb_norm,
        bins=kde_bins,
        kernel_type=kde_kernel,
        bandwidth=kde_bandwidth,
        range_min=-1.0,
        range_max=1.0,
        ignore_borders=kde_ignore_borders,
    )

    peaks = kde_peaks_above_below(
        kde_res["kde"],
        kde_res["bin_centers"],
        distance=kde_peak_distance,
        n=3,
        split_at=0.0,                                                # current price normalizes to 0.0
    )

    # 5. Fill indices 60-65 (see spec 3.3): above defaults +1.0, below defaults -1.0.
    fl[60] = fl[61] = fl[62] = 1.0
    fl[63] = fl[64] = fl[65] = -1.0

    for j, p in enumerate(peaks["above_prices"][:3]):
        fl[60 + j] = p
    for j, p in enumerate(peaks["below_prices"][:3]):
        fl[63 + j] = p

    return fl


def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_single_by_index(
    arr: np.ndarray,
    look_back: int,
    look_ahead: int,
    k_scaler: float,
    index: int,
    **kde_kwargs,
) -> np.ndarray:
    """Prepare the 66-vector for the observation anchored at ``index``."""
    return asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare(
        arr[index - look_back + 1 : index + look_ahead + 1],
        look_back,
        look_ahead,
        k_scaler,
        **kde_kwargs,
    )


def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_single_by_date_string(
    arr: np.ndarray,
    look_back: int,
    look_ahead: int,
    k_scaler: float,
    date_string: str,
    **kde_kwargs,
) -> np.ndarray:
    """Prepare the 66-vector for the observation whose ts matches ``date_string``.

    ``date_string`` format is "YYYY-MM-DD HH:MM:SS". It is converted to a
    millisecond timestamp; a binary search over the (ascending) ``ts`` column
    finds the matching candle, then ``_single_by_index`` is called. Raises if no
    exact match is found.
    """
    target_timestamp = int(np.datetime64(date_string, 'ms').astype(np.int64))
    left = 0
    right = arr.shape[0] - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid, 0] == target_timestamp:
            return asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_single_by_index(
                arr, look_back, look_ahead, k_scaler, mid, **kde_kwargs
            )
        elif arr[mid, 0] < target_timestamp:
            left = mid + 1
        else:
            right = mid - 1
    raise ValueError(f"Timestamp {date_string} not found in the array")


def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_all(
    arr: np.ndarray,
    look_back: int,
    look_ahead: int,
    k_scaler: float,
    **kde_kwargs,
) -> np.ndarray:
    """Build the 66-wide fl matrix for every valid observation.

    Plain Python loop (NOT ``nb.prange``): scipy peak-finding runs once per
    observation, so this is much slower than the njit/prange original (~1M
    observations for btcusdt). A simple loop is acceptable per spec. Optional
    parallelism (e.g. chunked ``concurrent.futures``/``joblib``) MAY be added so
    long as it does not change results; the default path is a straightforward
    loop.
    """
    n_obs = arr.shape[0] - look_back - look_ahead + 1
    result = np.zeros((n_obs, 66), dtype=np.float64)
    for i in range(look_back - 1, arr.shape[0] - look_ahead):
        result[i - (look_back - 1)] = asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare(
            arr[i - look_back + 1 : i + look_ahead + 1],
            look_back,
            look_ahead,
            k_scaler,
            **kde_kwargs,
        )
    return result
