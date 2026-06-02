# Rolling Robust Z-Score Indicator

## Task Summary

Implement a new indicator `rolling_robust_z_score` in the `packages/indicators/` package. The function computes a robust z-score using median and IQR over a rolling window, with an optimized incremental sort strategy compatible with Numba's `@nb.njit`.

---

## Background and Context

The indicators package (`packages/indicators/`) contains Numba-jitted rolling-window functions that operate on 1-D `np.float64` arrays. Each indicator lives in its own file, is decorated with `@nb.njit(inline='always')`, uses explicit loops (no high-level NumPy calls inside jitted code), and zero-pads output positions where the full window is unavailable.

The robust z-score replaces the mean/stddev of a standard z-score with the median and IQR, making it resistant to outliers:

```
robust_z_score = (x - median) / IQR
```

where `IQR = Q3 - Q1` (75th percentile minus 25th percentile).

---

## Repository Conventions (from `/agents/general/indicators.md`)

- Decorator: `@nb.njit(inline='always')`
- Input: 1-D `numpy.ndarray`, `dtype=np.float64`
- Output: newly allocated 1-D `numpy.ndarray`, same shape, `dtype=np.float64`
- `window` parameter: integer, default `60`
- Padding: indices `< window - 1` filled with `0.0`
- Explicit `for` loops inside jitted functions (no NumPy high-level calls)
- Each indicator in its own `.py` file under `packages/indicators/`
- Exported via `packages/indicators/__init__.py`

---

## Functional Requirements

### Signature

```python
@nb.njit(inline='always')
def rolling_robust_z_score(array, window=60):
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `array` | `np.ndarray` (1-D, float64) | Input data |
| `window` | `int` (default 60) | Lookback size **including** the current element |

### Output

- `np.ndarray` (1-D, float64), same length as `array`.

### Behaviour

1. **Zero-padding:** For indices `i < window - 1`, set `output[i] = 0.0`.
2. **Rolling window:** For each index `i >= window - 1`, consider the window `array[i - window + 1 : i + 1]` (total of `window` elements — the current element plus `window - 1` preceding elements).
3. **Sorted window maintenance (incremental sort):**
   - For the **first valid window** (`i == window - 1`): copy the window into a working sorted buffer and perform a full sort (e.g., insertion sort or any O(W log W) sort suitable for Numba).
   - For subsequent windows: one value drops out (the element leaving the window) and one new value enters. Find the old value in the sorted buffer, overwrite it with the new value, and "bubble" it into its correct sorted position. This is O(W) per step and requires **zero new memory allocation** inside the loop.
4. **Median:** From the sorted buffer of size `W`:
   - If `W` is odd: `median = sorted[W // 2]`
   - If `W` is even: `median = (sorted[W // 2 - 1] + sorted[W // 2]) / 2.0`
5. **IQR (Interquartile Range):**
   - `Q1 = sorted[W // 4]`
   - `Q3 = sorted[3 * W // 4]`
   - `IQR = Q3 - Q1`
6. **Robust Z-Score:**
   - If `IQR == 0.0`: `output[i] = 0.0` (avoid division by zero)
   - Otherwise: `output[i] = (array[i] - median) / IQR`

### Edge Cases

- `IQR == 0` (all window values identical or near-identical quartiles): output `0.0`.
- `window <= 0` or `window > len(array)`: behaviour is undefined (may assume valid input per existing conventions).

---

## Non-Goals / Out of Scope

- No support for 2-D or higher-dimensional arrays.
- No support for NaN handling.
- No weighted or exponential variant.
- No unit test file creation (unless the downstream agent decides to add one).

---

## Assumptions

- Input arrays are contiguous C-order float64 arrays with no NaN/Inf values.
- `window` is always a positive integer ≤ `len(array)`.
- The sorted buffer is allocated once (before the loop) as a 1-D array of size `window` and reused throughout.

---

## Acceptance Criteria

1. **File created:** `packages/indicators/rolling_robust_z_score.py` containing the `rolling_robust_z_score` function.
2. **Numba-compatible:** Function is decorated with `@nb.njit(inline='always')` and compiles/runs without error.
3. **Export added:** `packages/indicators/__init__.py` updated to import and expose `rolling_robust_z_score`.
4. **Agents doc updated:** `/agents/general/indicators.md` updated with a new section documenting `rolling_robust_z_score` following the existing format.
5. **Zero-padding:** First `window - 1` output values are `0.0`.
6. **Correctness:** For a known input, output matches the expected robust z-score computed via naive method.
7. **Incremental sort:** The implementation does NOT re-sort the full window each iteration (only the first window is fully sorted; subsequent windows use the bubble/insert strategy).
8. **No extra allocation:** No new arrays or lists are created inside the main loop.

---

## Open Questions

None — the specification is fully defined.

---

## Notes for the Downstream Coding Agent

- Follow the exact style of `packages/indicators/stddev.py` as a template for file structure and imports.
- The sorted buffer should be allocated with `np.empty(window, dtype=np.float64)` before the main loop.
- For the first window, copy `array[0:window]` into the buffer and sort it in-place (a simple insertion sort works well inside Numba).
- For the bubble step: after overwriting the old value with the new value at position `k`, bubble up (`while k > 0 and buf[k] < buf[k-1]: swap`) or bubble down (`while k < window-1 and buf[k] > buf[k+1]: swap`).
- To find the old value in the sorted buffer, use a linear scan (O(W)) — binary search is also fine but adds complexity for marginal gain at typical window sizes.
- Remember: `inline='always'` is required on the decorator.
- Update `__init__.py` with: `from packages.indicators.rolling_robust_z_score import rolling_robust_z_score`
- Update `/agents/general/indicators.md` adding a new `### rolling_robust_z_score(array, window=60)` section following the existing documentation pattern.
