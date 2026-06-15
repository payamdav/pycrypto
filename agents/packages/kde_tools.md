# KDE Tools Package

## Identity

| Key          | Value                       |
|--------------|-----------------------------|
| Package path | `packages/kde_tools/`       |
| Purpose      | Volume-weighted kernel density estimate (KDE) construction and KDE peak finding over normalized look-back prices, extracted from `notebooks/tests/look_back_look_ahead.ipynb` (cells 5 and 6). |

| Exported function       | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `make_kernel`           | Build a normalized smoothing kernel (`Triangular`, `Epanechnikov`, `Uniform`) |
| `weighted_histogram`    | Numba-jitted volume-weighted histogram over a fixed range                   |
| `convolve_same`         | Numba-jitted `np.convolve(..., mode="same")` reimplementation               |
| `compute_kde`           | Orchestrator: border filter + weighted histogram + kernel convolution       |
| `top_kde_peaks`         | Top-`n` highest-prominence peaks via `scipy.signal`                         |
| `kde_peaks_above_below` | Top-`n` KDE peaks above and below the current price (primary entry point)   |

This package extracts **only** the KDE construction and peak-finding logic. It does
**not** load data, normalize prices, compute DVR oscillators, or plot — those remain
the caller's responsibility (the notebook does them upstream of cell 5).

---

## Normalized-Space Convention

The KDE operates on look-back prices already normalized to `[-1, 1]` around the last
candle's price (`price_l`), per
`agents/ideas/idea_normalize_based_on_last_price_clip.md`. In this normalized space:

- The **current price** (`price_l`, the last look-back candle's VWAP) maps to **`0.0`**.
- Prices **above** the current price are **positive** (`>= 0.0`).
- Prices **below** the current price are **negative** (`< 0.0`).
- Clipped extremes sit at `±1.0`.

`kde_peaks_above_below` splits at `split_at` (default `0.0`): "above" is
`bin_centers >= split_at`, "below" is `bin_centers < split_at`.

---

## Setup

The package directory is import-safe and lives under `packages/`. Import via dot
notation from the repository root:

```python
from packages.kde_tools import (
    make_kernel,
    weighted_histogram,
    convolve_same,
    compute_kde,
    top_kde_peaks,
    kde_peaks_above_below,
)
```

In a notebook, install the dependencies inline and clone the repo first (see
`agents/general/rules.md`):

```python
%pip install numpy numba scipy
```

Dependencies are listed in `packages/kde_tools/requirements.txt`
(`numpy`, `numba`, `scipy`).

---

## `make_kernel`

```python
def make_kernel(kernel_type: str = "Triangular", bandwidth: int = 5) -> np.ndarray:
```

Returns a 1-D `np.float64` kernel of length `2 * bandwidth + 1`, normalized so it sums
to `1.0`. With `x = arange(-bandwidth, bandwidth + 1)`:

- `"Triangular"`   → `max(1 - |x| / bandwidth, 0)`
- `"Epanechnikov"` → `max(1 - (x / bandwidth) ** 2, 0)`
- `"Uniform"`      → all ones

**Raises:** `ValueError` if `kernel_type` is not one of the three supported kernels.

The numeric core (`_make_kernel_core`) is `@nb.njit` and takes an integer kernel code
(`0=Triangular, 1=Epanechnikov, 2=Uniform`); `make_kernel` is a thin Python wrapper
that maps the string to the code and calls the jitted core.

---

## `weighted_histogram`

```python
@nb.njit
def weighted_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int = 200,
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> np.ndarray:
```

Reproduces `np.histogram(values, bins=bins, range=(range_min, range_max), weights=weights)[0]`.

- `bin_width = (range_max - range_min) / bins`.
- For each `(v, w)`: skip if `v < range_min` or `v > range_max`; otherwise
  `idx = int((v - range_min) / bin_width)`, clamping the boundary case `v == range_max`
  (`idx == bins`) down to `bins - 1`. Accumulate `counts[idx] += w`.

**Returns:** newly allocated `np.float64` array of length `bins`.

> Bin geometry helpers: `bin_width = (range_max - range_min) / bins` and
> `bin_centers = range_min + (arange(bins) + 0.5) * bin_width`, equivalent to the
> notebook's `(edges[:-1] + edges[1:]) / 2`. `compute_kde` returns both.

---

## `convolve_same`

```python
@nb.njit
def convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
```

Reproduces `np.convolve(signal, kernel, mode="same")` exactly. Output length equals
`len(signal)`. Implementation computes the full convolution (length `N + M - 1`) with
explicit loops and returns the centered slice `full[(M - 1) // 2 : (M - 1) // 2 + N]`,
where `N = len(signal)`, `M = len(kernel)`.

**Returns:** newly allocated `np.float64` array of length `len(signal)`.

---

## `compute_kde`

```python
def compute_kde(
    scaled_prices: np.ndarray,
    weights: np.ndarray,
    bins: int = 200,
    kernel_type: str = "Triangular",
    bandwidth: int = 5,
    range_min: float = -1.0,
    range_max: float = 1.0,
    ignore_borders: bool = True,
) -> dict:
```

Volume-weighted KDE over normalized look-back prices, matching the notebook exactly:

1. **Border filter** (only when `ignore_borders` is `True`): keep entries with
   `range_min < scaled_prices < range_max` (**strict** on both sides → drops values
   exactly at `±1.0`); `n_excluded` counts the dropped entries. When `ignore_borders`
   is `False`, use all entries and `n_excluded = 0`.
2. `counts = weighted_histogram(kde_prices, kde_weights, bins, range_min, range_max)`.
3. `bin_width = (range_max - range_min) / bins`;
   `bin_centers = range_min + (arange(bins) + 0.5) * bin_width`.
4. `kernel_arr = make_kernel(kernel_type, bandwidth)`.
5. `kde = convolve_same(counts, kernel_arr)`.

**Returns** a dict:

| Key           | Type / shape              | Description                                  |
|---------------|---------------------------|----------------------------------------------|
| `kde`         | `np.float64`, `(bins,)`   | Smoothed density (convolved counts)          |
| `counts`      | `np.float64`, `(bins,)`   | Raw weighted histogram                       |
| `bin_centers` | `np.float64`, `(bins,)`   | Bin-center prices in normalized space        |
| `bin_width`   | `float`                   | Bin width                                    |
| `kernel`      | `np.float64`, `(2*bandwidth+1,)` | Normalized smoothing kernel           |
| `n_excluded`  | `int`                     | Candles dropped by the border filter         |

---

## `top_kde_peaks`

```python
def top_kde_peaks(
    kde_series: np.ndarray,
    prices: np.ndarray,
    distance: float,
    n: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
```

Reproduces the notebook's `top_kde_peaks` exactly:

- `peaks, _ = scipy.signal.find_peaks(kde_series, distance=distance)`
- if no peaks → return two empty arrays.
- `proms = scipy.signal.peak_prominences(kde_series, peaks)[0]`
- `order = np.argsort(proms)[::-1][:n]` (highest prominence first; ties keep `argsort`'s order).
- returns `(prices[peaks[order]], proms[order])`.

`find_peaks` and `peak_prominences` are kept in scipy so their semantics
(local-maxima/plateau handling, distance-based priority filtering, prominence
base-finding) match the notebook bit-for-bit.

---

## `kde_peaks_above_below`

```python
def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    distance: float = 5,
    n: int = 3,
    split_at: float = 0.0,
) -> dict:
```

Primary entry point for "3 highest-prominence peaks above and 3 below the current
price." `split_at` is the current price in normalized space (defaults to `0.0`). Splits
`kde`/`bin_centers` into above (`bin_centers >= split_at`) and below
(`bin_centers < split_at`) halves, calls `top_kde_peaks` on each with the given
`distance` and `n`, and returns:

```python
{
    "above_prices": ...,  "above_proms": ...,   # >= split_at, top-n by prominence
    "below_prices": ...,  "below_proms": ...,   # <  split_at, top-n by prominence
}
```

---

## Numba Strategy

- **Jitted (explicit loops, clear speedups):** `weighted_histogram`, `convolve_same`,
  and the kernel-building core behind `make_kernel`. They run over per-observation
  arrays and benefit from jitting across many sliding windows. Per
  `agents/general/indicators.md`, they use explicit `for` loops — no `np.histogram`,
  `np.convolve`, `np.maximum`, `np.argsort`, etc. inside `@nb.njit` — and return newly
  allocated `np.float64` arrays.
- **Kept in scipy (not reimplemented):** `find_peaks` and `peak_prominences`. Their
  exact semantics must match the notebook bit-for-bit, and the per-half KDE has only
  ~`bins/2` points, so reimplementing them in numba would risk behavioral drift for no
  meaningful speedup. `top_kde_peaks` / `kde_peaks_above_below` stay plain Python
  wrappers around scipy.

---

## End-to-End Usage Example

```python
import numpy as np
from packages.kde_tools import compute_kde, kde_peaks_above_below

# scaled_lb : look-back VWAP normalized to [-1, 1] (current price -> 0.0)
# v_lb_norm : per-candle normalized volume weights (same length)
scaled_lb = np.clip(np.random.randn(1440) * 0.2, -1.0, 1.0)
v_lb_norm = np.random.rand(1440)

# 1. Build the volume-weighted KDE (notebook defaults)
result = compute_kde(
    scaled_lb,
    v_lb_norm,
    bins=200,
    kernel_type="Triangular",
    bandwidth=5,
    ignore_borders=True,
)
kde         = result["kde"]
bin_centers = result["bin_centers"]

# 2. Top-3 most prominent peaks above and below the current price (0.0)
peaks = kde_peaks_above_below(kde, bin_centers, distance=5, n=3, split_at=0.0)
print("Above:", peaks["above_prices"], peaks["above_proms"])
print("Below:", peaks["below_prices"], peaks["below_proms"])
```

---

## Notes for Agents

- Mirror the notebook's logic precisely; do not "improve" the algorithm.
- The border filter uses **strict** `>` and `<` so values exactly at `±1.0` are excluded
  when `ignore_borders=True`.
- For `convolve_same`, the centered-slice offset is `(M - 1) // 2` with `M = len(kernel)`.
- `np.argsort(proms)[::-1][:n]` is a descending sort keeping numpy's tie order — do not
  switch to a stable/different sort.
- The current price corresponds to normalized `0.0`; that is the default `split_at`.
- The package itself relies on `requirements.txt`; the inline `%pip install` note above
  applies only to notebooks.
