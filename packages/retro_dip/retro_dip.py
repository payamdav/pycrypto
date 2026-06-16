import numba as nb
import numpy as np


@nb.njit(cache=True)
def scan_retro_dip_signals(
    prices,
    max_lookback_window=1440,
    target_past_premium=0.0100,
    min_interim_drawdown=0.0020,
    max_interim_drawdown=0.0050,
):
    """Scan a 1-D price array for retro-dip mean-reversion signals.

    For every anchor index ``i`` (starting at ``max_lookback_window``) the
    function scans backward up to ``max_lookback_window`` candles tracking the
    lowest price seen. A signal is recorded when, before the running interim
    drawdown exceeds ``max_interim_drawdown``, a past candle is found whose
    premium over the current price reaches ``target_past_premium`` while the
    interim drawdown is at least ``min_interim_drawdown``.

    Parameters
    ----------
    prices : np.ndarray
        1-D ``float64`` array of candle VWAP values, ordered oldest -> newest.
        Index position acts as the (1-minute) timestamp.
    max_lookback_window : int
        Maximum depth to search backward in minutes/candles.
    target_past_premium : float
        Required premium of the past anchor price (e.g. ``0.0100`` = 100 bps).
    min_interim_drawdown : float
        Minimum drop required below the current price (e.g. ``0.0020`` = 20 bps).
    max_interim_drawdown : float
        Maximum allowed drop below the current price (e.g. ``0.0050`` = 50 bps).

    Returns
    -------
    np.ndarray
        2-D ``float64`` array, one row per signal with columns:
        ``[current_index, target_past_premium_index, interim_drawdown]``.
    """
    n = prices.shape[0]

    # Pre-allocate to the maximum possible number of signals (one per anchor).
    capacity = n - max_lookback_window
    if capacity < 0:
        capacity = 0
    out = np.empty((capacity, 3), dtype=np.float64)
    count = 0

    for i in range(max_lookback_window, n):
        current_vwap = prices[i]
        if current_vwap <= 0.0:
            continue

        lowest_vwap_seen = current_vwap
        limit = i - max_lookback_window  # inclusive lower bound for the scan

        for j in range(i - 1, limit - 1, -1):
            past_vwap = prices[j]

            # Track the lowest VWAP encountered so far in this backward scan.
            if past_vwap < lowest_vwap_seen:
                lowest_vwap_seen = past_vwap

            interim_drawdown = (current_vwap - lowest_vwap_seen) / current_vwap

            # Early exit: dropped too far below the current price.
            if interim_drawdown > max_interim_drawdown:
                break

            # Target condition: past candle is sufficiently above current price.
            past_premium = (past_vwap - current_vwap) / current_vwap
            if past_premium >= target_past_premium:
                if interim_drawdown >= min_interim_drawdown:
                    out[count, 0] = np.float64(i)
                    out[count, 1] = np.float64(j)
                    out[count, 2] = interim_drawdown
                    count += 1
                    break

    return out[:count]


@nb.njit(cache=True)
def cluster_retro_dip_signals(signals):
    """Merge strictly contiguous signals into clusters.

    Signals occurring in consecutive minutes (consecutive indices) are merged
    into a single cluster. Any gap greater than 1 between consecutive signal
    indices starts a new cluster. The earliest signal of each cluster is kept
    and the number of merged points is recorded.

    Parameters
    ----------
    signals : np.ndarray
        2-D ``float64`` array from :func:`scan_retro_dip_signals`, ordered by
        increasing current index, with columns
        ``[current_index, target_past_premium_index, interim_drawdown]``.

    Returns
    -------
    np.ndarray
        2-D ``float64`` array, one row per cluster with columns:
        ``[current_index, target_past_premium_index, interim_drawdown,
        cluster_length]``.
    """
    m = signals.shape[0]
    out = np.empty((m, 4), dtype=np.float64)
    count = 0

    if m == 0:
        return out[:count]

    # Initialize the first cluster with the first signal.
    cluster_current = signals[0, 0]
    cluster_anchor = signals[0, 1]
    cluster_drawdown = signals[0, 2]
    cluster_length = 1.0
    prev_index = signals[0, 0]

    for k in range(1, m):
        cur_index = signals[k, 0]
        # Strict contiguity: a gap larger than 1 minute starts a new cluster.
        if cur_index - prev_index == 1.0:
            cluster_length += 1.0
        else:
            out[count, 0] = cluster_current
            out[count, 1] = cluster_anchor
            out[count, 2] = cluster_drawdown
            out[count, 3] = cluster_length
            count += 1

            cluster_current = signals[k, 0]
            cluster_anchor = signals[k, 1]
            cluster_drawdown = signals[k, 2]
            cluster_length = 1.0

        prev_index = cur_index

    # Flush the final cluster.
    out[count, 0] = cluster_current
    out[count, 1] = cluster_anchor
    out[count, 2] = cluster_drawdown
    out[count, 3] = cluster_length
    count += 1

    return out[:count]


def retro_dip(
    prices,
    max_lookback_window=1440,
    target_past_premium=0.0100,
    min_interim_drawdown=0.0020,
    max_interim_drawdown=0.0050,
):
    """Detect and cluster retro-dip signals over a 1-D price array.

    This is the high-level entry point: it scans ``prices`` for signals and
    merges strictly contiguous signals into structural clusters.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of candle VWAP values, ordered oldest -> newest. Index
        position acts as the (1-minute) timestamp. Coerced to ``float64``.
    max_lookback_window : int
        Maximum depth to search backward in minutes/candles (default 1440).
    target_past_premium : float
        Required premium of the past anchor price (default 0.0100).
    min_interim_drawdown : float
        Minimum drop required below the current price (default 0.0020).
    max_interim_drawdown : float
        Maximum allowed drop below the current price (default 0.0050).

    Returns
    -------
    np.ndarray
        2-D ``float64`` array, one row per non-contiguous market opportunity
        with columns:
        ``[current_index, target_past_premium_index, interim_drawdown,
        cluster_length]``.
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)

    signals = scan_retro_dip_signals(
        prices,
        max_lookback_window,
        target_past_premium,
        min_interim_drawdown,
        max_interim_drawdown,
    )
    return cluster_retro_dip_signals(signals)
