# Indicators Package Specification

## Task Summary

Create a reusable indicators package at `packages/indicators/` containing numba-optimized technical indicator functions. Each indicator lives in its own file. All functions are decorated with `@nb.njit(inline='always')` for maximum performance. Additionally, create an agent reference document at `agents/general/indicators.md`.

---

## Background and Context

The repository already has a `packages/` folder for reusable libraries (e.g., `packages/numpy_candles/`). Per `agents/general/paths_and_files.md`, reusable libraries belong in `packages/<meaningful_name>/`. This new package follows the same pattern.

All indicators operate on 1-D numpy arrays of dtype `float64` and return a newly allocated array of the same shape and dtype.

---

## Repository Conventions (from /agents)

- Reusable packages go in `packages/<name>/` (`agents/general/paths_and_files.md`)
- General specifications/rules for agents go in `agents/general/` (`agents/general/paths_and_files.md`)
- The project uses numpy with float64 dtype arrays as a standard data representation

---

## Functional Requirements

### Package Structure

```
packages/indicators/
├── __init__.py        # imports all indicator functions for convenient access
├── ma.py             # Moving Average
├── wma.py            # Weighted Moving Average
├── rsi.py            # RSI scaled to [-1, 1]
└── stddev.py         # Standard Deviation
```

### Common Rules for All Indicators

1. Each function must be decorated with `@nb.njit(inline='always')`
2. Input array is a 1-D `numpy.ndarray` with `dtype=np.float64`
3. Output is a **newly allocated** `numpy.ndarray` with the same shape and dtype as input
4. The `window` parameter is an integer defaulting to `60`
5. For indices where the full window is not yet available (i.e., indices `0` to `window - 2`), the output value should be `0.0` (padding with zero)
6. Use explicit `for` loops inside the numba-jitted function (as fast as vectorized in numba)
7. Import numba as `nb` and numpy as `np`

### Indicator Definitions

#### 1. `ma(array, window=60)` — Moving Average

- File: `packages/indicators/ma.py`
- Computes the simple moving average over the trailing `window` elements
- `output[i] = mean(array[i - window + 1 : i + 1])` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`

#### 2. `wma(array, weights, window=60)` — Weighted Moving Average

- File: `packages/indicators/wma.py`
- `weights` is a 1-D numpy array of dtype `float64` with length equal to `window`
- Computes a weighted average: `output[i] = sum(array[i - window + 1 : i + 1] * weights) / sum(weights)` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`
- The function signature is `wma(array, weights, window=60)`

#### 3. `rsi_1_1(array, window=60)` — RSI scaled to [-1, 1]

- File: `packages/indicators/rsi.py`
- Compute standard RSI (Relative Strength Index) using the classic smoothed/exponential method:
  - Calculate price changes: `delta[i] = array[i] - array[i-1]`
  - Separate gains and losses
  - Use exponential moving average (Wilder's smoothing) with the given window for avg_gain and avg_loss
  - `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`
- Then scale from [0, 100] to [-1, 1]: `output = (RSI - 50) / 50`
  - This maps RSI 0 → -1, RSI 50 → 0, RSI 100 → +1
- `output[i] = 0.0` for `i < window` (insufficient data for calculation)

#### 4. `stddev(array, window=60)` — Standard Deviation

- File: `packages/indicators/stddev.py`
- Computes the population standard deviation over the trailing `window` elements
- `output[i] = std(array[i - window + 1 : i + 1])` for `i >= window - 1`
- `output[i] = 0.0` for `i < window - 1`
- Use population std (divide by N, not N-1)

### `__init__.py`

- Import all indicator functions so they can be accessed as:
  ```python
  from packages.indicators import ma, wma, rsi_1_1, stddev
  ```

### Agent Reference Document

Create `agents/general/indicators.md` containing:
- Package location and import path
- List of all available indicators with their signatures, parameter descriptions, and behavior notes
- Common conventions (output shape, dtype, padding, numba decoration)
- This file serves as the authoritative reference for AI agents to prevent redeveloping existing functionality

---

## Non-Goals / Out of Scope

- No CLI interface or script entry points
- No visualization or plotting
- No unit test files (unless explicitly requested later)
- No support for 2-D arrays or non-float64 dtypes at this time
- No caching or state management between calls

---

## Assumptions

- `numba` and `numpy` are already available in the project environment
- The caller is responsible for passing correctly shaped/typed arrays
- The `weights` array for `wma` will always have length equal to `window`
- Population standard deviation (N divisor) is used, not sample std (N-1)
- RSI uses Wilder's smoothing method (exponential moving average with alpha = 1/window)

---

## Acceptance Criteria

1. All four indicator files exist under `packages/indicators/` and each contains a single public function decorated with `@nb.njit(inline='always')`
2. `packages/indicators/__init__.py` exports `ma`, `wma`, `rsi_1_1`, `stddev`
3. Each function accepts the specified parameters and returns a new numpy array of the same shape/dtype
4. Padding positions (before window is filled) contain `0.0`
5. `agents/general/indicators.md` exists and documents all indicators with signatures and usage notes
6. All files pass `python -m py_compile` without errors
7. A quick smoke test (e.g., calling each function with a small array) should execute without errors

---

## Open Questions

None — the requirements are clear.

---

## Notes for the Downstream Coding Agent

- Use `import numba as nb` and `import numpy as np` at the top of each indicator file
- Inside `@nb.njit` functions you cannot use numpy high-level functions like `np.mean()` or `np.std()` — implement them with explicit loops
- The `inline='always'` parameter is critical for performance when indicators are composed
- Keep each file minimal — one function per file, no classes
- For the `__init__.py`, simply import from each submodule
- For `rsi_1_1`, handle the edge case where `avg_loss == 0` (RSI = 100, so output = 1.0)
- Remember to create **both** the package files and the `agents/general/indicators.md` reference document
- Verify files compile with: `python -m py_compile packages/indicators/<file>.py`
