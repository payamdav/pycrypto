from .asset_snapshot_lookback_lookahead_normalize import asset_snapshot_lookback_lookahead_normalize_prepare, asset_snapshot_lookback_lookahead_normalize_prepare_no_x_axis
import numba as nb
import numpy as np


@nb.njit(inline='always')
def asset_snapshot_lookback_lookahead_normalize_prepare_single_by_index(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float, index: int):
    return asset_snapshot_lookback_lookahead_normalize_prepare(arr[index - look_back + 1 : index + look_ahead + 1], look_back, look_ahead, k_scaler)


def asset_snapshot_lookback_lookahead_normalize_prepare_single_by_date_string(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float, date_string: str):
    # date_string format is "YYYY-MM-DD HH:MM:SS"
    # we convert it to timestamp ms and look for ts equals to that; if we find the exact equal timestamp we use that index, otherwise raise error. we assume that arr is sorted by timestamp in ascending order and we can use binary search to find the closest index
    target_timestamp = int(np.datetime64(date_string, 'ms').astype(np.int64))
    left = 0
    right = arr.shape[0] - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid, 0] == target_timestamp:
            return asset_snapshot_lookback_lookahead_normalize_prepare_single_by_index(arr, look_back, look_ahead, k_scaler, mid)
        elif arr[mid, 0] < target_timestamp:
            left = mid + 1
        else:
            right = mid - 1
    raise ValueError(f"Timestamp {date_string} not found in the array")


@nb.njit(parallel=True)
def asset_snapshot_lookback_lookahead_normalize_prepare_all(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float):
    for i in nb.prange(look_back - 1, arr.shape[0] - look_ahead):
        asset_snapshot_lookback_lookahead_normalize_prepare(arr[i - look_back + 1 : i + look_ahead + 1], look_back, look_ahead, k_scaler)


@nb.njit(parallel=True)
def asset_snapshot_lookback_lookahead_normalize_prepare_all_no_x_axis(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float):
    for i in nb.prange(look_back - 1, arr.shape[0] - look_ahead):
        asset_snapshot_lookback_lookahead_normalize_prepare_no_x_axis(arr[i - look_back + 1 : i + look_ahead + 1], look_back, look_ahead, k_scaler)
