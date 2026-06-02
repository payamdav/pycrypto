# Indicators Package

## Package Location

```
packages/indicators/
```

## Import Path

```python
from packages.indicators import ma, wma, vwma, rsi_1_1, stddev
```

Or import individual functions directly:

```python
from packages.indicators.ma import ma
from packages.indicators.wma import wma
from packages.indicators.vwma import vwma
from packages.indicators.rsi import rsi_1_1
from packages.indicators.stddev import stddev
```

---

## Common Conventions

- All functions are decorated with `@nb.njit(inline='always')` for maximum numba performance.
- Input: 1-D `numpy.ndarray` with `dtype=np.float64`.
- Output: newly allocated 1-D `numpy.ndarray` with the same shape and `dtype=np.float64`.
- The `window` parameter is an integer defaulting to `60`.
- Padding: indices where the full window is not yet available are filled with `0.0`.
- Explicit `for` loops are used inside jitted functions (no numpy high-level calls inside `@nb.njit`).

---

## Available Indicators

### `ma(array, window=60)` — Moving Average

**File:** `packages/indicators/ma.py`

**Signature:**
```python
def ma(array: np.ndarray, window: int = 60) -> np.ndarray
```

**Behavior:**
- Computes the simple moving average over the trailing `window` elements.
- `output[i] = mean(array[i - window + 1 : i + 1])` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`

---

### `wma(array, weights, window=60)` — Weighted Moving Average

**File:** `packages/indicators/wma.py`

**Signature:**
```python
def wma(array: np.ndarray, weights: np.ndarray, window: int = 60) -> np.ndarray
```

**Parameters:**
- `weights`: 1-D `np.float64` array of length `window`.

**Behavior:**
- Computes a weighted average over the trailing `window` elements.
- `output[i] = sum(array[i - window + 1 : i + 1] * weights) / sum(weights)` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`

---

### `vwma(array, volume, window=60)` — Volume Weighted Moving Average

**File:** `packages/indicators/vwma.py`

**Signature:**
```python
def vwma(array: np.ndarray, volume: np.ndarray, window: int = 60) -> np.ndarray
```

**Parameters:**
- `volume`: 1-D `np.float64` array with the same shape as `array`.

**Behavior:**
- Computes a volume-weighted moving average over the trailing `window` elements.
- `output[i] = sum(array[i - window + 1 : i + 1] * volume[i - window + 1 : i + 1]) / sum(volume[i - window + 1 : i + 1])` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`

---

### `rsi_1_1(array, window=60)` — RSI Scaled to [-1, 1]

**File:** `packages/indicators/rsi.py`

**Signature:**
```python
def rsi_1_1(array: np.ndarray, window: int = 60) -> np.ndarray
```

**Behavior:**
- Computes standard RSI using Wilder's smoothing method (exponential moving average with `alpha = 1/window`).
- Seeds avg_gain and avg_loss from the first `window` price changes, then applies EMA smoothing.
- Scales from [0, 100] to [-1, 1]: `output = (RSI - 50) / 50`
  - RSI 0 → -1, RSI 50 → 0, RSI 100 → +1
- Edge case: when `avg_loss == 0`, output is `1.0` (RSI = 100).
- `output[i] = 0.0` for `i < window` (insufficient data for initial seeding).

---

### `stddev(array, window=60)` — Standard Deviation

**File:** `packages/indicators/stddev.py`

**Signature:**
```python
def stddev(array: np.ndarray, window: int = 60) -> np.ndarray
```

**Behavior:**
- Computes the **population** standard deviation (divides by N, not N-1) over the trailing `window` elements.
- `output[i] = std(array[i - window + 1 : i + 1])` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`

---

### `rolling_robust_z_score(array, window=60)` — Rolling Robust Z-Score

**File:** `packages/indicators/rolling_robust_z_score.py`

**Signature:**
```python
def rolling_robust_z_score(array: np.ndarray, window: int = 60) -> np.ndarray
```

**Behavior:**
- Computes a robust z-score using median and IQR over the trailing `window` elements.
- `output[i] = (array[i] - median) / IQR` for `i >= window - 1`, where `IQR = Q3 - Q1`.
- `Q1 = sorted[W // 4]`, `Q3 = sorted[3 * W // 4]` from the sorted window of size `W`.
- `output[i] = 0.0` for `i < window - 1` (zero-padding) or when `IQR == 0`.
- Uses an incremental sort strategy: the first valid window is fully sorted once; subsequent windows update the sorted buffer in O(W) by replacing the outgoing value with the incoming value and bubbling it into position.

---

## Usage Example

```python
import numpy as np
from packages.indicators import ma, wma, rsi_1_1, stddev

prices = np.random.randn(200).astype(np.float64).cumsum() + 100.0

ma_out    = ma(prices, window=20)
rsi_out   = rsi_1_1(prices, window=14)
std_out   = stddev(prices, window=20)

weights   = np.arange(1, 21, dtype=np.float64)   # length must equal window
wma_out   = wma(prices, weights, window=20)

volume    = np.random.rand(200).astype(np.float64) * 1000.0
vwma_out  = vwma(prices, volume, window=20)
```
