# Idea: Look-Back / Look-Ahead Windowing

## Purpose

A reusable data-preparation pattern for ML studies on candle data.
Every study that references **look_back_look_ahead** follows this specification exactly.

---

## Parameters

| Parameter     | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `asset`       | str     | —       | Lowercase asset name (e.g. `"btcusdt"`) |
| `start`       | str     | —       | Study start date `"YYYY-MM-DD"` |
| `end`         | str     | —       | Study end date `"YYYY-MM-DD"` |
| `look_back`   | int     | —       | Number of preceding candles per observation (includes current candle) |
| `look_ahead`  | int     | —       | Number of future candles per observation |
| `time_frame`  | int     | `1`     | Candle duration in minutes |
| `columns`     | list    | all     | Candle columns to load; if omitted load all 11 columns |

---

## Conventions

| Term             | Definition |
|------------------|------------|
| `last_candle`    | The most recent candle in the look-back window (index `look_back - 1` within the window) |
| `last_time`      | `last_candle.ts` — the **open time** of the last candle (millisecond epoch) |
| `current_time`   | `last_time + time_frame * 60_000` — the moment the last candle closed; "now" from the observation's perspective |
| `price`          | `vwap` of a candle unless stated otherwise |
| `current_price`  | `last_candle.c` — close price of the last candle |

> Candle `ts` is always the **open time**. A finished candle with `ts = T` covers the interval `[T, T + tf_ms)` where `tf_ms = time_frame * 60_000`.

---

## Observation Definition

For a single observation anchored at position `i` (0-indexed over the full loaded array, sorted by `ts`):

```
look-back window  : candles[ i - look_back + 1  …  i ]   (look_back rows, inclusive)
look-ahead window : candles[ i + 1              …  i + look_ahead ] (look_ahead rows)
last_candle       : candles[i]
```

A valid observation requires both windows to be fully populated.

---

## Inclusive vs Exclusive Date Boundaries

### Exclusive (default — silent means exclusive)

The agent loads **more data than the stated date range** so that every observation
inside `[start, end]` has complete look-back, look-ahead, and any historic-indicator data.

```
actual_load_start = start
                    - look_back      * time_frame minutes
                    - indicator_extra * time_frame minutes   ← see Historic Indicators
actual_load_end   = end + look_ahead * time_frame minutes
```

Valid observation indices inside the loaded array:

```python
valid_mask = (ts >= start_ms) & (ts <= end_ms)
# every row in valid_mask has full look_back behind it and full look_ahead ahead of it
```

### Inclusive

The date range `[start, end]` is treated as the safe observation window directly.
No extra boundary loading is needed for look-back/look-ahead.
The safe slice of the loaded array is:

```python
obs_indices = range(look_back - 1, len(arr) - look_ahead)
# arr is loaded exactly for [start, end] with no extensions
```

Use `[look_back : -look_ahead]` (or `[look_back-1 : len-look_ahead]`) as the
observation-anchor index range.

---

## Windowing Modes

### 1. Loop (iterator)

One observation at a time. Memory = O(look_back + look_ahead).

```python
for i in range(look_back - 1, len(arr) - look_ahead):
    lb_window = arr[i - look_back + 1 : i + 1]          # shape (look_back, cols)
    la_window = arr[i + 1            : i + 1 + look_ahead]  # shape (look_ahead, cols)
    last_candle   = lb_window[-1]
    current_price = last_candle["c"]
```

### 2. Vectorized

All observations at once as 2-D arrays. Fast but memory = O(items × look_back).

```python
from numpy.lib.stride_tricks import sliding_window_view

# lb_col: 1-D array of one column, shape (N,)
lb_2d = sliding_window_view(lb_col, look_back)          # shape (N - look_back + 1, look_back)
la_2d = sliding_window_view(la_col, look_ahead)         # shape (N - look_ahead + 1, look_ahead)

# Align: both arrays must start at the same anchor index
# anchor range: [look_back - 1 , N - look_ahead]
n_obs     = len(arr) - look_back - look_ahead + 1
lb_2d     = lb_2d[:n_obs]                               # shape (n_obs, look_back)
la_2d     = la_2d[look_back:][:n_obs]                   # shape (n_obs, look_ahead)
```

For multiple columns repeat per column or stack after:

```python
# Multi-column look-back: shape (n_obs, look_back, n_cols)
import numpy as np
lb_multi = np.stack(
    [sliding_window_view(arr[:, c], look_back) for c in range(n_cols)],
    axis=-1
)[:n_obs]
```

### 3. Chunked Vectorized

Vectorized inside a loop of `chunk_size` observations. Controls peak RAM.

```python
chunk_size = 512   # tune to available RAM

for start_idx in range(0, n_obs, chunk_size):
    end_idx   = min(start_idx + chunk_size, n_obs)
    lb_chunk  = lb_2d[start_idx:end_idx]    # shape (≤chunk_size, look_back)
    la_chunk  = la_2d[start_idx:end_idx]    # shape (≤chunk_size, look_ahead)
    # process chunk …
```

---

## Historic Indicators

Some studies require indicators computed over a **history window longer than look_back**
(e.g. a 10-day moving average of volume requires 10 × 1440 preceding minutes).

### Notation

```
indicator_extra = indicator_period_in_candles
                  e.g. 10-day MA of volume → indicator_extra = 10 * 1440 = 14_400
```

### Boundary Extension (exclusive mode — default)

```
actual_load_start = start
                    - (look_back + indicator_extra) * time_frame minutes
actual_load_end   = end + look_ahead * time_frame minutes
```

### Usage Pattern

```python
# arr is fully loaded including the indicator history prefix
# The indicator prefix occupies indices [0 : indicator_extra]
# Observations start at index: look_back - 1 + indicator_extra

for i in range(look_back - 1 + indicator_extra, len(arr) - look_ahead):
    lb_window       = arr[i - look_back + 1          : i + 1]
    la_window       = arr[i + 1                      : i + 1 + look_ahead]
    indicator_hist  = arr[i - look_back + 1 - indicator_extra : i - look_back + 1]
    ma_volume       = indicator_hist[:, col_v].mean()   # example: simple MA
```

---

## Data Loading

Use `agents/datasets/huggingface_candles.md → load_range()` with the extended boundaries.

```python
# Compute extended boundaries before calling load_range
from datetime import datetime, timedelta, timezone

tf_min           = time_frame                        # minutes per candle
indicator_extra  = 0                                 # set if historic indicators needed

dt_start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
dt_end   = datetime.fromisoformat(end  ).replace(tzinfo=timezone.utc)

load_start = dt_start - timedelta(minutes=(look_back + indicator_extra) * tf_min)
load_end   = dt_end   + timedelta(minutes= look_ahead                   * tf_min)

df = load_range(
    asset,
    load_start.strftime("%Y-%m-%d"),
    load_end  .strftime("%Y-%m-%d"),
    columns=columns,   # None = all
)
arr = df.sort_values("ts").to_numpy()   # shape (N, n_cols)
```

---

## Quick-Reference Cheat Sheet

```
look_back = 1440, look_ahead = 240, time_frame = 1 (default)

observation i:
  feature input : arr[i-1439 : i+1]      # 1440 rows
  label   input : arr[i+1   : i+241]     # 240  rows
  last_candle   : arr[i]
  last_time     : arr[i, col_ts]
  current_time  : arr[i, col_ts] + 60_000   (ms)
  current_price : arr[i, col_c]

exclusive (default) → extend load boundaries by look_back behind and look_ahead ahead
inclusive           → use arr[look_back-1 : -look_ahead] as anchor range directly

historic indicator (e.g. 10-day vol MA):
  indicator_extra = 10 * 1440 = 14_400
  extend load_start by an additional 14_400 minutes
  valid anchor starts at index: look_back - 1 + indicator_extra
```

---

## Column Index Reference

When working with numpy arrays (after `.to_numpy()`), column order matches the load
order. Always resolve indices from the DataFrame column list before converting:

```python
cols       = list(df.columns)          # preserves order
col_ts     = cols.index("ts")
col_o      = cols.index("o")
col_h      = cols.index("h")
col_l      = cols.index("l")
col_c      = cols.index("c")
col_v      = cols.index("v")
col_q      = cols.index("q")
col_n      = cols.index("n")
col_vwap   = cols.index("vwap")
col_vb     = cols.index("vb")
col_vs     = cols.index("vs")
```

Full column descriptions → see `agents/datasets/huggingface_candles.md`.
