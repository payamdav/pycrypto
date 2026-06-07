import numba as nb
import numpy as np


@nb.njit(inline='always')
def asset_snapshot_lookback_lookahead_normalize_prepare(arr: np.ndarray, look_back: int, look_ahead: int, k_scaler: float):
    # columns of arr: ts=0, c=1, v=2, q=3, vwap=4, vb=5, vs=6, v_rzs=7, v_ma=8, rsi_1_1_period_1=9, rsi_1_1_period_2=10, rsi_1_1_period_3=11
    # ts: timestamp ms, c: close price, v: volume, q: quote volume, vwap: volume weighted average price, vb: aggressive buy volume, vs: aggressive sell volume, v_rzs: volume robust z-score in last week, v_ma: moving average of volume last week, rsi_1_1_period_1: RSI with period 7, rsi_1_1_period_2: RSI with period 14, rsi_1_1_period_3: RSI with period 60 rsi_1_1 is scaled to be between -1 and 1, where 0 is the middle point, -1 is the lowest point and 1 is the highest point in the look_back period
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
    la_x_axis = (np.arange(look_ahead) / look_back) + 1  # we scale look_ahead x axis with look_back to have the same scale as look_back x axis, and we shift it by 1 to start from current time
    la_normalized_price = k_scaler * (arr[look_back:, 4] - last_candle_price) / last_candle_price  # normalize vwap price in the look_ahead period
    la_normalized_price = np.clip(la_normalized_price, -1, 1)

    # Features and labels vector preparation
    # features and labels vector. holds all in 1D vector. related items like sequences will store as a chunk on this vector. The following index helpers can be used to access different parts of the vector
    # f for feature. l for label. e for expanding window. si for start index. ei for end index. i for index. n for normalized

    f_e_vwap_si, f_e_vwap_ei = 0, 23
    f_e_n_imbalances_si, f_e_n_imbalances_ei = 24, 47
    f_rsi_1_1_period_1_i, f_rsi_1_1_period_2_i, f_rsi_1_1_period_3_i = 48, 49, 50
    l_e_vwap_si, l_e_vwap_ei = 51, 54
    l_e_n_imbalances_si, l_e_n_imbalances_ei = 55, 58
    
    fl = np.zeros(58, dtype=np.float64)  # features and labels vector of size 58

    fl[f_rsi_1_1_period_1_i] = arr[look_back - 1, 9]  # rsi_1_1_period_1 of the last candle in the look_back period
    fl[f_rsi_1_1_period_2_i] = arr[look_back - 1, 10]  # rsi_1_1_period_2 of the last candle in the look_back period
    fl[f_rsi_1_1_period_3_i] = arr[look_back - 1, 11]  # rsi_1

    # right anchored expanding window over arr[:look_back] with step 60 and count 24
    _volume_acc = 0
    _quote_acc = 0
    _vb_acc = 0
    _vs_acc = 0

    _rew_step = 60
    _rew_count = 24
    _rew_i = look_back - 1
    for _rew_j in range(_rew_count):
        _rew_out_idx = _rew_count - 1 - _rew_j  # fills output[-1] first, then output[-2], ...
        for _rew_s in range(_rew_step):
            # accumulate arr[_rew_i]
            _volume_acc += arr[_rew_i, 2]  # volume
            _quote_acc += arr[_rew_i, 3]   # quote volume
            _vb_acc += arr[_rew_i, 5]      # aggressive buy volume
            _vs_acc += arr[_rew_i, 6]      # aggressive sell volume

            _rew_i -= 1
        _vwap = _quote_acc / _volume_acc if _volume_acc > 0 else 0
        # vwap must be normalized as other prices above
        _vwap = k_scaler * (_vwap - last_candle_price) / last_candle_price  # normalize vwap price in the look_back period
        _vwap = np.clip(_vwap, -1, 1)
        fl[f_e_vwap_si + _rew_out_idx] = _vwap
        fl[f_e_n_imbalances_si + _rew_out_idx] = (_vb_acc - _vs_acc) / (_vb_acc + _vs_acc) if (_vb_acc + _vs_acc) > 0 else 0

    # left anchored expanding window over arr[look_back:] with step 60 and count 4
    _volume_acc = 0
    _quote_acc = 0
    _vb_acc = 0
    _vs_acc = 0
    
    _lew_step = 60
    _lew_count = 4
    _lew_i = look_back
    for _lew_j in range(_lew_count):
        for _lew_s in range(_lew_step):
            # accumulate arr[_lew_i]
            _volume_acc += arr[_lew_i, 2]  # volume
            _quote_acc += arr[_lew_i, 3]   # quote volume
            _vb_acc += arr[_lew_i, 5]      # aggressive buy volume
            _vs_acc += arr[_lew_i, 6]      # aggressive sell volume
            _lew_i += 1
        _vwap = _quote_acc / _volume_acc if _volume_acc > 0 else 0
        _vwap = k_scaler * (_vwap - last_candle_price) / last_candle_price  # normalize vwap price in the look_ahead period
        _vwap = np.clip(_vwap, -1, 1)
        fl[l_e_vwap_si + _lew_j] = _vwap
        fl[l_e_n_imbalances_si + _lew_j] = (_vb_acc - _vs_acc) / (_vb_acc + _vs_acc) if (_vb_acc + _vs_acc) > 0 else 0




    return fl
