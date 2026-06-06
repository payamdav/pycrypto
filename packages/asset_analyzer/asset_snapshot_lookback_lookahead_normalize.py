import numba as nb
import numpy as np


@nb.njit(inline='always')
def asset_snapshot_lookback_lookahead_normalize_prepare(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float):
    # columns of arr: ts=0, c=1, v=2, q=3, vwap=4, vb=5, vs=6, v_rzs=7, v_ma=8, rsi_1_1_7=9, rsi_1_1_14=10, rsi_1_1_60=11
    # ts: timestamp ms, c: close price, v: volume, q: quote volume, vwap: volume weighted average price, vb: aggressive buy volume, vs: aggressive sell volume, v_rzs: volume robust z-score in last week, v_ma: moving average of volume last week, rsi_1_1_7: RSI with period 7, rsi_1_1_14: RSI with period 14, rsi_1_1_60: RSI with period 60 rsi_1_1 is scaled to be between -1 and 1, where 0 is the middle point, -1 is the lowest point and 1 is the highest point in the look_back period
    # arr is already sliced and its shape is (look_back + look_ahead, 12)
    # we assume that we are at the end of the look_back period and the look_ahead period is in the future
    # when we think of a price of candle we normally think about vwap

    candle_time_frame_ms = arr[1, 0] - arr[0, 0]
    last_candle_timestamp = arr[look_back - 1, 0]
    current_timestamp = last_candle_timestamp + candle_time_frame_ms
    last_candle_price = arr[look_back - 1, 4]  # vwap of the last candle in the look_back period
    current_price = arr[look_back - 1, 1]  # close price of the last candle in the look_back period - the latest price we know without looking into the future
    # we normalize price with scaled relative change to the last candle price, and we scale it with k_scaler to focus more on recent price changes and clip all out of -1, 1 range to be -1 or 1
    # for x axis ( time axis ) we use fixed 0 to 1 range where 1 is the current time and 0 is the time of the first candle in the look_back period
    lb_x_axis = np.arange(look_back) / look_back
    lb_normalized_price = k_scaler * (arr[:look_back, 4] - last_candle_price) / last_candle_price  # normalize vwap price in the look_back period
    lb_normalized_price = np.clip(lb_normalized_price, -1, 1)
    la_x_axis = np.arange(look_ahead) / look_back
    la_normalized_price = k_scaler * (arr[look_back:, 4] - last_candle_price) / last_candle_price  # normalize vwap price in the look_ahead period
    la_normalized_price = np.clip(la_normalized_price, -1, 1)
    return


@nb.njit(inline='always')
def asset_snapshot_lookback_lookahead_normalize_prepare_no_x_axis(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float):
    # columns of arr: ts=0, c=1, v=2, q=3, vwap=4, vb=5, vs=6, v_rzs=7, v_ma=8, rsi_1_1_7=9, rsi_1_1_14=10, rsi_1_1_60=11
    # ts: timestamp ms, c: close price, v: volume, q: quote volume, vwap: volume weighted average price, vb: aggressive buy volume, vs: aggressive sell volume, v_rzs: volume robust z-score in last week, v_ma: moving average of volume last week, rsi_1_1_7: RSI with period 7, rsi_1_1_14: RSI with period 14, rsi_1_1_60: RSI with period 60 rsi_1_1 is scaled to be between -1 and 1, where 0 is the middle point, -1 is the lowest point and 1 is the highest point in the look_back period
    # arr is already sliced and its shape is (look_back + look_ahead, 12)
    # we assume that we are at the end of the look_back period and the look_ahead period is in the future
    # when we think of a price of candle we normally think about vwap

    candle_time_frame_ms = arr[1, 0] - arr[0, 0]
    last_candle_timestamp = arr[look_back - 1, 0]
    current_timestamp = last_candle_timestamp + candle_time_frame_ms
    last_candle_price = arr[look_back - 1, 4]  # vwap of the last candle in the look_back period
    current_price = arr[look_back - 1, 1]  # close price of the last candle in the look_back period - the latest price we know without looking into the future
    # we normalize price with scaled relative change to the last candle price, and we scale it with k_scaler to focus more on recent price changes and clip all out of -1, 1 range to be -1 or 1
    # for x axis ( time axis ) we use fixed 0 to 1 range where 1 is the current time and 0 is the time of the first candle in the look_back period
    # lb_x_axis = np.arange(look_back) / look_back
    lb_normalized_price = k_scaler * (arr[:look_back, 4] - last_candle_price) / last_candle_price  # normalize vwap price in the look_back period
    lb_normalized_price = np.clip(lb_normalized_price, -1, 1)
    # la_x_axis = np.arange(look_ahead) / look_back
    la_normalized_price = k_scaler * (arr[look_back:, 4] - last_candle_price) / last_candle_price  # normalize vwap price in the look_ahead period
    la_normalized_price = np.clip(la_normalized_price, -1, 1)
    return


