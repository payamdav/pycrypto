# KDE Tools Package Specification

## Task Summary

Create a reusable package at `packages/kde_tools/` that reproduces **exactly** the
KDE (kernel density estimate) construction and KDE peak-finding logic currently
embedded in `notebooks/tests/look_back_look_ahead.ipynb` (cells 5 and 6).

The package must provide:

1. **Kernel creation** — `make_kernel(...)` for `"Triangular"`, `"Epanechnikov"`,
   `"Uniform"` kernels.
2. **Volume-weighted histogram** — a numba-jitted weighted histogram over a fixed
   range (the building block of the KDE).
3. **KDE computation** — border filtering + weighted histogram + kernel
   convolution, returning the smoothed density and its bin centers.
4. **Peak finding** — find the **3 highest-prominence peaks above** and the
   **3 highest-prominence peaks below** the current price, exactly as the notebook
   does with `scipy.signal.find_peaks` + `peak_prominences`.

Use **numba `@nb.njit`** for every numeric core where it speeds up the work
(weighted histogram, convolution, kernel construction). The peak-detection step
keeps `scipy.signal` (see *Numba Strategy* below).

Also create an agent reference document at `agents/packages/kde_tools.md`.

---

## Background and Context

In the referenced notebook the look-back/look-ahead window is loaded and the
look-back VWAP is normalized to `[-1, 1]` around the last candle's price
(`price_l`) per `agents/ideas/idea_normalize_based_on_last_price_clip.md`. In
this normalized space:

- The **current price** (`price_l`, the last look-back candle's VWAP) maps to
  **`0.0`**.
- Prices **above** the current price are **positive** (`scaled >= 0`).
- Prices **below** the current price are **negative** (`scaled < 0`).
- Clipped extremes sit at `±1.0`.

A **volume-weighted KDE** of the normalized look-back prices reveals where trading
volume concentrated (support/resistance-like price magnets). The notebook then
finds the most prominent KDE peaks above and below the current price.

This package extracts **only** the KDE construction and peak-finding logic. It does
**not** load data, normalize prices, compute DVR oscillators, or plot — those remain
the caller's responsibility (the notebook already does them upstream).

---

## Repository Conventions (from /agents)

- Reusable packages go in `packages/<name>/` (`agents/general/paths_and_files.md`).
- Any Python file with external dependencies must ship a `requirements.txt` in the
  same folder (`agents/general/rules.md`).
- Numba conventions follow `agents/general/indicators.md`: explicit `for` loops
  inside jitted functions (no numpy high-level calls inside `@nb.njit`), 1-D
  `np.float64` arrays, newly allocated outputs.
- Package agent-reference docs live under `agents/packages/` (e.g.
  `agents/packages/gcs_tools.md`).

---

## Reference Logic (from the notebook — must be reproduced exactly)

### KDE construction (notebook cell 5)

```python
# scaled_lb : normalized look-back prices in [-1, 1]  (price_l -> 0.0)
# v_lb_norm : per-candle normalized volume weights (same length as scaled_lb)

# 1. Optional border exclusion: drop clipped prices at the extremes
if KDE_IGNORE_BORDERS:
    border_mask = (scaled_lb > -1.0) & (scaled_lb < 1.0)   # STRICT inequalities
    kde_prices  = scaled_lb[border_mask]
    kde_weights = v_lb_norm[border_mask]
    n_excluded  = (~border_mask).sum()
else:
    kde_prices  = scaled_lb
    kde_weights = v_lb_norm
    n_excluded  = 0

# 2. Volume-weighted histogram over the fixed range [-1, 1]
counts, bin_edges = np.histogram(
    kde_prices, bins=BINS_COUNT, range=(-1.0, 1.0), weights=kde_weights,
)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width   = bin_edges[1] - bin_edges[0]

# 3. Kernel
def make_kernel(kernel_type, bandwidth):
    x = np.arange(-bandwidth, bandwidth + 1, dtype=float)
    if kernel_type == "Triangular":
        k = np.maximum(1.0 - np.abs(x) / bandwidth, 0.0)
    elif kernel_type == "Epanechnikov":
        k = np.maximum(1.0 - (x / bandwidth) ** 2, 0.0)
    elif kernel_type == "Uniform":
        k = np.ones(len(x))
    else:
        raise ValueError(f"Unknown kernel: {kernel_type!r}")
    k /= k.sum()           # normalized to sum to 1
    return k

kernel_arr = make_kernel(KERNEL, BANDWIDTH)

# 4. Smooth by convolution (same-length output)
kde = np.convolve(counts, kernel_arr, mode="same")
```

### Peak finding (notebook cell 6)

```python
from scipy.signal import find_peaks, peak_prominences

pos_mask   = bin_centers >= 0      # at / above current price
neg_mask   = bin_centers <  0      # below current price
pos_kde, pos_prices = kde[pos_mask], bin_centers[pos_mask]
neg_kde, neg_prices = kde[neg_mask], bin_centers[neg_mask]

def top_kde_peaks(kde_series, prices, distance, n=3):
    peaks, _ = find_peaks(kde_series, distance=distance)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    proms = peak_prominences(kde_series, peaks)[0]
    order = np.argsort(proms)[::-1][:n]          # highest prominence first
    return prices[peaks[order]], proms[order]

pos_peak_prices, pos_peak_proms = top_kde_peaks(pos_kde, pos_prices, distance=BANDWIDTH)
neg_peak_prices, neg_peak_proms = top_kde_peaks(neg_kde, neg_prices, distance=BANDWIDTH)
```

Notebook parameter values for reference: `BINS_COUNT = 200`, `BANDWIDTH = 5`,
`KERNEL = "Triangular"`, `KDE_IGNORE_BORDERS = True`. The `distance` argument to
`find_peaks` is `BANDWIDTH` for both the above and below halves.

---

## Functional Requirements

### Package Structure

```
packages/kde_tools/
├── __init__.py          # re-exports the public API
├── kernels.py           # make_kernel
├── histogram.py         # weighted_histogram (numba)
├── kde.py               # convolve_same (numba) + compute_kde (orchestrator)
├── peaks.py             # top_kde_peaks + kde_peaks_above_below
└── requirements.txt     # numpy, numba, scipy
```

### Common Rules

1. All numeric arrays are 1-D `numpy.ndarray` with `dtype=np.float64`; jitted
   functions return newly allocated arrays.
2. `import numba as nb`, `import numpy as np`.
3. Inside `@nb.njit` functions use explicit `for` loops — no `np.histogram`,
   `np.convolve`, `np.maximum`, `np.argsort`, etc.
4. Public results must match the notebook's numpy/scipy results to floating-point
   tolerance (`rtol=1e-9`/`atol=1e-9` for the KDE arrays; identical peak prices and
   prominences for the peak functions).

---

### 1. `make_kernel(kernel_type, bandwidth)` — `kernels.py`

```python
def make_kernel(kernel_type: str = "Triangular", bandwidth: int = 5) -> np.ndarray
```

- Returns a 1-D `np.float64` kernel of length `2 * bandwidth + 1`, normalized so it
  sums to `1.0`.
- `x = arange(-bandwidth, bandwidth + 1)`:
  - `"Triangular"`  → `max(1 - |x| / bandwidth, 0)`
  - `"Epanechnikov"`→ `max(1 - (x / bandwidth)**2, 0)`
  - `"Uniform"`     → all ones
- Raise `ValueError` for an unknown `kernel_type`.
- **Numba:** implement the numeric core in an `@nb.njit` helper that takes an
  **integer kernel code** (`0=Triangular, 1=Epanechnikov, 2=Uniform`) and the
  bandwidth, builds the kernel with an explicit loop, and normalizes it. The public
  `make_kernel` is a thin Python wrapper that maps the string to the code (raising
  `ValueError` on unknown strings) and calls the jitted helper. This keeps the
  string-friendly API while the loop runs jitted.

### 2. `weighted_histogram(values, weights, bins, range_min, range_max)` — `histogram.py`

```python
@nb.njit
def weighted_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int = 200,
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> np.ndarray   # returns counts, shape (bins,)
```

- Reproduces `np.histogram(values, bins=bins, range=(range_min, range_max), weights=weights)[0]`.
- `bin_width = (range_max - range_min) / bins`.
- For each `v, w`: skip if `v < range_min` or `v > range_max`; otherwise
  `idx = int((v - range_min) / bin_width)`, and clamp `idx == bins` (the case
  `v == range_max`) down to `bins - 1`. Accumulate `counts[idx] += w`.
- Returns a newly allocated `np.float64` array of length `bins`.

> Provide a tiny non-jitted helper (or document the formula) for `bin_edges` /
> `bin_centers` / `bin_width` so callers reproduce the notebook's
> `bin_centers = (edges[:-1] + edges[1:]) / 2`. `compute_kde` (below) returns these,
> so a standalone helper is optional.

### 3. `convolve_same(signal, kernel)` — `kde.py`

```python
@nb.njit
def convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray
```

- Reproduces `np.convolve(signal, kernel, mode="same")` exactly.
- Output length = `len(signal)` (the KDE always has `len(signal) >= len(kernel)`).
- Implementation: compute the full convolution (length `N + M - 1`) conceptually and
  return the centered slice `full[(M - 1) // 2 : (M - 1) // 2 + N]`, where
  `N = len(signal)`, `M = len(kernel)`. Use explicit loops; do **not** call
  `np.convolve`. Validate against `np.convolve(..., mode="same")` in the smoke test.

### 4. `compute_kde(...)` — `kde.py` (orchestrator)

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
):
    ...
    return {
        "kde":         kde,          # np.float64, shape (bins,)
        "counts":      counts,       # raw weighted histogram, shape (bins,)
        "bin_centers": bin_centers,  # shape (bins,)
        "bin_width":   bin_width,    # float
        "kernel":      kernel_arr,   # shape (2*bandwidth+1,)
        "n_excluded":  n_excluded,   # int, candles dropped by border filter
    }
```

Steps, matching the notebook exactly:

1. **Border filter** (only when `ignore_borders` is `True`): keep entries with
   `range_min < scaled_prices < range_max` (**strict** on both sides → drops values
   sitting exactly at `±1.0`); `n_excluded` counts the dropped entries. When
   `ignore_borders` is `False`, use all entries and `n_excluded = 0`.
2. `counts = weighted_histogram(kde_prices, kde_weights, bins, range_min, range_max)`.
3. `bin_width = (range_max - range_min) / bins`;
   `bin_centers = range_min + (arange(bins) + 0.5) * bin_width`
   (equivalent to the notebook's `(edges[:-1] + edges[1:]) / 2`).
4. `kernel_arr = make_kernel(kernel_type, bandwidth)`.
5. `kde = convolve_same(counts, kernel_arr)`.

> The border filter, edge math, and dict assembly are light Python glue (run once
> per observation, tiny arrays); they need not be jitted. The heavy numeric cores
> (histogram, convolution, kernel) are jitted as specified above.

### 5. `top_kde_peaks(kde_series, prices, distance, n)` — `peaks.py`

```python
def top_kde_peaks(
    kde_series: np.ndarray,
    prices: np.ndarray,
    distance: float,
    n: int = 3,
):
    return peak_prices, peak_proms   # both np.ndarray, length <= n
```

- Reproduces the notebook's `top_kde_peaks` **exactly**:
  - `peaks, _ = scipy.signal.find_peaks(kde_series, distance=distance)`
  - if no peaks → return two empty arrays.
  - `proms = scipy.signal.peak_prominences(kde_series, peaks)[0]`
  - `order = np.argsort(proms)[::-1][:n]` (highest prominence first; ties keep
    `argsort`'s order).
  - return `prices[peaks[order]]`, `proms[order]`.

### 6. `kde_peaks_above_below(kde, bin_centers, distance, n, split_at)` — `peaks.py`

```python
def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    distance: float = 5,
    n: int = 3,
    split_at: float = 0.0,
):
    return {
        "above_prices": ...,  "above_proms": ...,   # >= split_at, top-n by prominence
        "below_prices": ...,  "below_proms": ...,   # <  split_at, top-n by prominence
    }
```

- `split_at` is the **current price in normalized space** — defaults to `0.0`
  (because `price_l` normalizes to `0.0`). "Above" = `bin_centers >= split_at`,
  "below" = `bin_centers < split_at`, matching the notebook's `pos_mask`/`neg_mask`.
- Split `kde` and `bin_centers` by the masks, call `top_kde_peaks` on each half with
  the given `distance` and `n`, and return the four arrays.
- This is the package's primary entry point for "3 highest-prominence peaks above and
  3 below the current price."

### `__init__.py`

Re-export the public API:

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

### `requirements.txt`

```
numpy
numba
scipy
```

### Agent Reference Document — `agents/packages/kde_tools.md`

Create a reference doc (same style as `agents/packages/gcs_tools.md`) covering:
package location and import path; each function's signature, parameters, return
value, and behavior; the normalized-space convention (current price = `0.0`, above =
positive, below = negative); the numba strategy; and a short end-to-end usage example
that mirrors the notebook (compute KDE from normalized prices + volume weights, then
get top-3 peaks above/below).

---

## Numba Strategy

- **Jit (clear speedups, explicit loops):** `weighted_histogram`, `convolve_same`,
  and the kernel-building core behind `make_kernel`. These run over per-observation
  arrays and benefit from jitting, especially when the package is called across many
  sliding windows.
- **Keep in scipy (do NOT reimplement):** `find_peaks` and `peak_prominences`. Their
  exact semantics (local-maxima/plateau handling, distance-based priority filtering,
  prominence base-finding) must match the notebook bit-for-bit; reimplementing them
  in numba would risk behavioral drift for no meaningful speedup (the per-half KDE has
  ~`bins/2` points). `top_kde_peaks` / `kde_peaks_above_below` therefore stay plain
  Python wrappers around scipy.
- Follow `agents/general/indicators.md`: no numpy high-level calls inside
  `@nb.njit`; build outputs with explicit loops.

---

## Non-Goals / Out of Scope

- No data loading, look-back/look-ahead windowing, price/time normalization, DVR
  oscillators, or volume normalization — the caller supplies `scaled_prices` and
  `weights` already prepared (as the notebook does upstream of cell 5).
- No plotting / matplotlib / Plotly.
- No CLI, no notebook edits, no GCS export.
- No support for 2-D arrays or non-`float64` dtypes.

---

## Assumptions

- `numpy`, `numba`, and `scipy` are available in the environment.
- `scaled_prices` and `weights` are equal-length 1-D arrays; `weights` are the
  per-candle normalized volumes (`v_lb_norm` in the notebook).
- Normalization to `[-1, 1]` with the current price at `0.0` has already been applied
  by the caller; the KDE range defaults to `(-1.0, 1.0)`.
- `bandwidth >= 1` and `bins >= 1`.

---

## Acceptance Criteria

1. `packages/kde_tools/` exists with `__init__.py`, `kernels.py`, `histogram.py`,
   `kde.py`, `peaks.py`, and `requirements.txt`.
2. `from packages.kde_tools import make_kernel, weighted_histogram, convolve_same,
   compute_kde, top_kde_peaks, kde_peaks_above_below` works.
3. `make_kernel` returns a normalized (sum `= 1`) kernel of length `2*bandwidth+1`
   for each of the three kernel types and raises `ValueError` otherwise.
4. `weighted_histogram` matches `np.histogram(..., range=(-1,1), weights=...)[0]` to
   `atol=1e-9` on random inputs; `convolve_same` matches
   `np.convolve(..., mode="same")` to `atol=1e-9`.
5. `compute_kde` reproduces the notebook's `kde`, `bin_centers`, `bin_width`, and
   `n_excluded` for the same inputs/parameters (`bins=200`, `bandwidth=5`,
   `kernel="Triangular"`, `ignore_borders=True`), including the strict `±1.0` border
   exclusion.
6. `kde_peaks_above_below` returns the same peak prices and prominences (top-3 above
   `0.0` and top-3 below `0.0`, by descending prominence, `distance=bandwidth`) as the
   notebook's cell 6.
7. The jitted functions (`weighted_histogram`, `convolve_same`, kernel core) carry
   `@nb.njit` and use explicit loops (no numpy high-level calls inside).
8. All files pass `python -m py_compile`.
9. `agents/packages/kde_tools.md` exists and documents the package.

---

## Open Questions

None — the notebook fully determines the required behavior.

---

## Notes for the Downstream Coding Agent

- Mirror the notebook's logic precisely; do not "improve" the algorithm.
- Border filter uses **strict** `>` and `<` so values exactly at `±1.0` are excluded
  when `ignore_borders=True`.
- For `convolve_same`, the centered-slice offset is `(M - 1) // 2` with
  `M = len(kernel)`; verify against `np.convolve(..., mode="same")` before finishing.
- `np.argsort(proms)[::-1][:n]` is a descending sort that keeps numpy's tie order —
  replicate it exactly (do not switch to a stable/different sort).
- The current price corresponds to normalized `0.0`; that is the default `split_at`
  separating "above" from "below". Expose it as a parameter but default to `0.0`.
- Keep modules minimal: jitted numeric cores in their files, thin Python orchestration
  for `compute_kde` and the peak wrappers.
- Add an inline `%pip install`/import-path note only in docs, not in package code;
  the package itself relies on `requirements.txt`.
