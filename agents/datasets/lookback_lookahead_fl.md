# Lookback-Lookahead Feature/Label Dataset (`fl_data`)

## Identity

| Key          | Value |
|--------------|-------|
| Storage      | GCS bucket `payamdprojectbucket` |
| Key pattern  | `fl_data_{asset}` (e.g. `fl_data_btcusdt`) |
| Format       | NumPy `.npy` (saved via `np.save`, loaded via `np.load`) |
| dtype        | `float64` |
| Shape        | `(n_obs, 60)` |
| Parameters   | `LOOK_BACK=1440`, `LOOK_AHEAD=240`, `K=100` |

`n_obs = total_candles - LOOK_BACK - LOOK_AHEAD + 1`  
For btcusdt the source has ~1 042 561 candles → ~1 040 882 observations.

---

## Assets

Confirmed generated: **btcusdt**.  
Source candles (`lookback_lookahead_{asset}`) exist for all 7 assets (btcusdt, ethusdt, adausdt, xrpusdt, trumpusdt, vineusdt, dogeusdt); run the processing notebook to generate `fl_data_{asset}` for each.

---

## How to Load

```python
import io
import numpy as np
import sys
sys.path.insert(0, "pycrypto/packages/tools/google_cloud_storage_tools")
from gcs_tools import gcs_json_key_file, read_file

gcs_json_key_file()   # always call first

asset = "btcusdt"
data_bytes = read_file("payamdprojectbucket", f"fl_data_{asset}")
fl = np.load(io.BytesIO(data_bytes))   # shape (n_obs, 60), dtype float64
```

---

## Vector Layout (60 columns)

Each row is one observation anchored at `last_candle` (the final candle of the look-back window).

| Index | Name | Description |
|-------|------|-------------|
| 0 | `ts` | `current_timestamp` = `last_candle.ts + candle_timeframe_ms` (ms epoch) |
| 1–24 | `f_e_vwap[0..23]` | Right-anchored expanding VWAP features (24 values) |
| 25–48 | `f_e_n_imbalances[0..23]` | Right-anchored expanding normalized volume imbalances (24 values) |
| 49 | `f_rsi_1_1_period_1` | RSI (period 7), scaled to [−1, 1] |
| 50 | `f_rsi_1_1_period_2` | RSI (period 14), scaled to [−1, 1] |
| 51 | `f_rsi_1_1_period_3` | RSI (period 60), scaled to [−1, 1] |
| 52–55 | `l_e_vwap[0..3]` | Left-anchored expanding VWAP labels (4 values) |
| 56–59 | `l_e_n_imbalances[0..3]` | Left-anchored expanding normalized volume imbalance labels (4 values) |

---

## Feature Details

### `f_e_vwap` (indices 1–24) — Right-anchored, step=60, count=24

Right-anchored expanding VWAP over the look-back window, sampled every 60 candles.  
Each value = `clip(K * (vwap_acc - last_candle_price) / last_candle_price, -1, 1)`  
where `vwap_acc` is the VWAP of a cumulative slice ending at `last_candle`.

- `fl[1]`  = VWAP of oldest 60-min bucket (candles ~1440..1381 before current)
- `fl[24]` = VWAP of most recent 60-min bucket (last 60 candles)

Index increases toward the present: **`fl[1]` = oldest, `fl[24]` = newest**.

### `f_e_n_imbalances` (indices 25–48) — Right-anchored, step=60, count=24

Same windows as `f_e_vwap` but computes normalized volume imbalance:  
`(vb_acc - vs_acc) / (vb_acc + vs_acc)`, range `[-1, 1]`  
where `vb` = aggressive buy volume, `vs` = aggressive sell volume.

- `fl[25]` = oldest bucket, `fl[48]` = most recent bucket.

### `f_rsi_1_1_period_*` (indices 49–51)

RSI of the last candle in the look-back window, scaled to `[-1, 1]` (`(RSI - 50) / 50`).

---

## Label Details

### `l_e_vwap` (indices 52–55) — Left-anchored, step=60, count=4

Left-anchored **expanding** VWAP over the look-ahead window. Each value = VWAP over the first `(j+1)*60` candles after `last_candle`, normalized the same way as features.

| Index | Horizon | Candles used |
|-------|---------|--------------|
| 52 | 1 h | first 60 look-ahead candles |
| 53 | 2 h | first 120 look-ahead candles (expanding) |
| 54 | 3 h | first 180 look-ahead candles (expanding) |
| 55 | 4 h | all 240 look-ahead candles (expanding) |

Value = `clip(K * (vwap_acc - last_candle_price) / last_candle_price, -1, 1)`

### `l_e_n_imbalances` (indices 56–59) — Left-anchored, step=60, count=4

Same expanding windows as `l_e_vwap` but for volume imbalance `(vb_acc - vs_acc) / (vb_acc + vs_acc)`.

| Index | Horizon |
|-------|---------|
| 56 | 1 h |
| 57 | 2 h |
| 58 | 3 h |
| 59 | 4 h |

---

## Normalization Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `K` | 100 | Scale factor; a 1% price move → normalized value of 1.0 |
| `last_candle_price` | `arr[look_back-1, 4]` (vwap) | Normalization anchor per observation |
| Clip | hard `[-1, 1]` | Moves ≥1% from anchor are clamped |

---

## Index Constants (use in code)

```python
TS_I                  = 0
F_E_VWAP_SI           = 1;  F_E_VWAP_EI           = 24   # inclusive
F_E_N_IMBALANCES_SI   = 25; F_E_N_IMBALANCES_EI   = 48   # inclusive
F_RSI_P1_I            = 49  # period 7
F_RSI_P2_I            = 50  # period 14
F_RSI_P3_I            = 51  # period 60
L_E_VWAP_SI           = 52; L_E_VWAP_EI           = 55   # inclusive
L_E_N_IMBALANCES_SI   = 56; L_E_N_IMBALANCES_EI   = 59   # inclusive
```

---

## Source Data & Processing

Source candles are stored at `gs://payamdprojectbucket/lookback_lookahead_{asset}` (numpy array, shape `(N, 12)`).

Source columns: `ts=0, c=1, v=2, q=3, vwap=4, vb=5, vs=6, v_rzs=7, v_ma=8, rsi_p7=9, rsi_p14=10, rsi_p60=11`

Processing function: `asset_snapshot_lookback_lookahead_normalize_prepare_all` in  
`packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize_caller.py`

To regenerate: run `notebooks/studies/lookback_lookahead_features_labels/lookback_lookahead_data_process.ipynb` with the desired `ASSET`.
