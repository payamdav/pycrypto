import io
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

# Column indices in the output array
NC = SimpleNamespace(
    ts=0,   # open_time (ms)
    o=1,    # open
    h=2,    # high
    l=3,    # low
    c=4,    # close
    v=5,    # volume
    q=6,    # quote_volume
    n=7,    # count (number of trades)
    vwap=8, # quote_volume / volume
    vb=9,   # taker_buy_volume
    vs=10,  # volume - taker_buy_volume
)

# CSV column indices to load (skip close_time=6, taker_buy_quote_volume=10, ignore=11)
_CSV_COLS = (0, 1, 2, 3, 4, 5, 7, 8, 9)


def load_numpy_candles_from_binance_file(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as response:
            raw = response.read()
        data = io.BytesIO(raw)
    else:
        data = path_or_url

    loaded = np.genfromtxt(
        data,
        delimiter=",",
        skip_header=1,
        usecols=_CSV_COLS,
        dtype=np.float64,
    )

    n_rows = loaded.shape[0]
    out = np.empty((n_rows, 11), dtype=np.float64)

    # ts, o, h, l, c, v, q, n
    out[:, NC.ts] = loaded[:, 0]
    out[:, NC.o]  = loaded[:, 1]
    out[:, NC.h]  = loaded[:, 2]
    out[:, NC.l]  = loaded[:, 3]
    out[:, NC.c]  = loaded[:, 4]
    out[:, NC.v]  = loaded[:, 5]
    out[:, NC.q]  = loaded[:, 6]
    out[:, NC.n]  = loaded[:, 7]

    # vwap = quote_volume / volume; forward-fill candles with zero volume or quote
    # to avoid discontinuities (nan/zero) in the vwap series
    v_col, q_col = loaded[:, 5], loaded[:, 6]
    invalid = (v_col == 0) | (q_col == 0)
    safe_v = np.where(invalid, 1.0, v_col)
    vwap = np.where(invalid, np.nan, q_col / safe_v)
    # forward-fill: each invalid index inherits the last valid index's value
    fwd_idx = np.arange(n_rows)
    fwd_idx[invalid] = 0
    np.maximum.accumulate(fwd_idx, out=fwd_idx)
    out[:, NC.vwap] = vwap[fwd_idx]

    # vb = taker_buy_volume
    out[:, NC.vb] = loaded[:, 8]

    # vs = volume - taker_buy_volume
    out[:, NC.vs] = loaded[:, 5] - loaded[:, 8]

    return out


def numpy_candles_filter_date(arr: np.ndarray, start_date=None, end_date=None, count=None) -> np.ndarray:
    def _to_ms(s):
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000

    ts = arr[:, NC.ts]
    mask = np.ones(len(ts), dtype=bool)

    if start_date is not None:
        mask &= ts >= _to_ms(start_date)
    if end_date is not None:
        mask &= ts <= _to_ms(end_date)

    result = arr[mask]
    if count is not None:
        result = result[:count]
    return result


def numpy_candles_info(arr: np.ndarray) -> None:
    ts = arr[:, NC.ts]
    tf_sec = int((ts[1] - ts[0]) / 1000)
    duration_ms = ts[-1] - ts[0]
    duration_min = duration_ms / 1000 / 60
    duration_days = duration_ms / 1000 / 60 / 60 / 24
    first_dt = datetime.fromtimestamp(ts[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    last_dt  = datetime.fromtimestamp(ts[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"shape={arr.shape}  tf={tf_sec}s  "
        f"duration={duration_min:.1f}min/{duration_days:.2f}days  "
        f"first={first_dt}  last={last_dt}"
    )


def numpy_candle_test(arr: np.ndarray) -> None:
    errors = []

    def _row_list(mask, limit=5):
        indices = np.where(mask)[0]
        sample = ", ".join(str(i) for i in indices[:limit])
        suffix = f" … ({len(indices)} total)" if len(indices) > limit else f" ({len(indices)} total)"
        return sample + suffix

    # 1. All values must be finite and >= 0
    bad_finite = ~np.isfinite(arr)
    if bad_finite.any():
        for col_idx, name in vars(NC).items():
            col_mask = bad_finite[:, name]
            if col_mask.any():
                errors.append(f"Non-finite values in column '{col_idx}': rows {_row_list(col_mask)}")

    bad_negative = arr < 0
    if bad_negative.any():
        for col_idx, name in vars(NC).items():
            col_mask = bad_negative[:, name]
            if col_mask.any():
                errors.append(f"Negative values in column '{col_idx}': rows {_row_list(col_mask)}")

    # 2. ts differences must all be equal (no missing candles)
    ts = arr[:, NC.ts]
    if len(ts) > 1:
        diffs = np.diff(ts)
        expected_diff = diffs[0]
        unequal = diffs != expected_diff
        if unequal.any():
            errors.append(
                f"Irregular timestamp gaps (expected {expected_diff:.0f} ms): "
                f"after rows {_row_list(unequal)}"
            )

    # 3. open and close must be within [low, high]
    for col_name, col_idx in (("open", NC.o), ("close", NC.c)):
        below = arr[:, col_idx] < arr[:, NC.l]
        above = arr[:, col_idx] > arr[:, NC.h]
        if below.any():
            errors.append(f"{col_name} below low: rows {_row_list(below)}")
        if above.any():
            errors.append(f"{col_name} above high: rows {_row_list(above)}")

    # 4. For candles with non-zero volume and quote, vwap must equal q / v
    active = (arr[:, NC.v] > 0) & (arr[:, NC.q] > 0)
    if active.any():
        expected_vwap = arr[active, NC.q] / arr[active, NC.v]
        bad_vwap = ~np.isclose(arr[active, NC.vwap], expected_vwap)
        if bad_vwap.any():
            active_indices = np.where(active)[0]
            bad_rows = active_indices[bad_vwap]
            errors.append(f"vwap != q/v for active candles: rows {_row_list(bad_rows > -1, limit=5)}")
            # reframe mask into full-array space for _row_list
            full_mask = np.zeros(len(arr), dtype=bool)
            full_mask[bad_rows] = True
            errors[-1] = f"vwap != q/v for active candles: rows {_row_list(full_mask)}"

    # 5. vs + vb must equal v for every candle
    bad_vol_split = ~np.isclose(arr[:, NC.vs] + arr[:, NC.vb], arr[:, NC.v])
    if bad_vol_split.any():
        errors.append(f"vs + vb != v: rows {_row_list(bad_vol_split)}")

    # 6. Candles with n > 0 must have volume > 0 and quote > 0
    has_trades = arr[:, NC.n] > 0
    if has_trades.any():
        zero_v = has_trades & (arr[:, NC.v] == 0)
        zero_q = has_trades & (arr[:, NC.q] == 0)
        if zero_v.any():
            errors.append(f"n > 0 but volume == 0: rows {_row_list(zero_v)}")
        if zero_q.any():
            errors.append(f"n > 0 but quote == 0: rows {_row_list(zero_q)}")

    if errors:
        for msg in errors:
            print(f"FAIL: {msg}")
    else:
        print("all tests passed")
