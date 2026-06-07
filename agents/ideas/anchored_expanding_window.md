# Idea: Anchored Expanding Window

## Purpose

A reusable computation pattern for running an operation over an **expanding** slice of an
array, sampled at fixed `step` intervals, anchored to one end of the data.
Every study or function that references **anchored_expanding_window** follows this
specification exactly.

The window grows by `step` items on each iteration. After iteration `j` (0-indexed) the
window covers `(j + 1) * step` items, and one result is written to the output array. The
result is therefore a coarse, fixed-length (`count`) summary of an expanding region of the
array.

---

## What Must Be Defined First

Before any code is written, an anchored-expanding-window request **must** specify all four
of the following. If any is missing, the agent must ask for it.

| Item        | Description |
|-------------|-------------|
| **Anchor**  | `right_anchored` or `left_anchored` — which end of the array the window is anchored to (see below) |
| **`step`**  | Number of array items added to the window on each iteration |
| **`count`** | Number of iterations / output items. The full span consumed is `count * step` items |
| **Operation** | The exact calculation performed and stored at each step (e.g. VWAP, mean, sum, std, min/max, custom) |

> The total number of array items touched is always `count * step`. The array must contain
> at least `count * step` items in the anchored direction; otherwise the request is invalid.

---

## Parameters

| Parameter | Type            | Description |
|-----------|-----------------|-------------|
| `arr`     | `np.ndarray`    | Source data, ordered oldest → newest (same convention as candle arrays) |
| `step`    | int             | Items added to the window per iteration |
| `count`   | int             | Number of outputs (and iterations) |
| `output`  | `np.ndarray`    | Pre-allocated 1-D array of length `count` to receive one result per step |

---

## Anchors

### `left_anchored_expanding_window`

The window starts at the **first** item of `arr` and expands to the **right**. The loop
walks items `0 … count*step - 1`. Output index `0` is computed first and corresponds to the
smallest window (`arr[0 : step]`); the last output (`count - 1`) corresponds to the full
span (`arr[0 : count*step]`).

```
output[0]         ← operation over arr[0 :   1*step]
output[1]         ← operation over arr[0 :   2*step]
...
output[count-1]   ← operation over arr[0 : count*step]
```

### `right_anchored_expanding_window`

Everything is identical to left-anchored **except** the window is anchored to the **last**
item of `arr` and expands to the **left**, and the output fills **from the end**. The first
result computed is placed in `output[-1]` and uses `arr[-step:]`; the second is placed in
`output[-2]` and uses `arr[-2*step:]`; and so on.

```
output[-1]        ← operation over arr[ -1*step : ]
output[-2]        ← operation over arr[ -2*step : ]
...
output[-count]    ← operation over arr[ -count*step : ]   (== output[0])
```

> In both anchors, `output[j]` summarizes a window of `(j + 1) * step` items measured from
> the anchored end. Left-anchored counts windows from the start; right-anchored counts them
> from the end.

---

## Reference Logic (illustrative pseudo-code)

The snippet below mixes Python / C / numpy syntax for clarity only. It demonstrates the
expanding-window + accumulator idea for a VWAP over candles with `count = 5`, `step = 10`.
Accumulators carry forward across iterations, so each output reuses prior work instead of
re-summing the whole window.

```text
# left_anchored, operation = VWAP, count = 5, step = 10
i           = 0
volume_acc  = 0
quote_acc   = 0
output      = np.zeros(count)

current_output_count = 0
while (current_output_count < count) {
    current_step = 0
    while (current_step < step) {
        volume_acc += arr[i].volume
        quote_acc  += arr[i].quote
        i           += 1
        current_step += 1
    }
    output[current_output_count] = quote_acc / volume_acc
    current_output_count += 1
}
```

For `right_anchored`, walk `i` from the last item leftward and fill `output` from the end:
the first computed value goes to `output[-1]` (using `arr[-step:]`), the next to
`output[-2]` (using `arr[-2*step:]`), continuing to `output[-count]`.

> This logic is illustrative. The same result may be produced with cleaner or safer loop
> structures (e.g. an outer loop over `count` and an inner loop over `step`, or slicing).
> Prefer whichever is clearer and equally efficient.

---

## Accumulators and Multiple Outputs

Because the window only ever **grows** (items are added, never removed), running
accumulators are the natural, O(count * step) way to implement these windows — never
re-scan the whole window on each step.

Using **more accumulators** lets a single pass compute several operations at once. For
example, keeping `sum`, `sum_sq`, and `count_n` simultaneously yields mean **and**
population standard deviation in the same loop; keeping `volume_acc` and `quote_acc`
yields VWAP. When multiple operations are requested, accumulate all required quantities in
one pass and write each into its own `output`-shaped array.

---

## Implementation Guidance

- Default language is **Python + numpy**. Use `np.float64` arrays unless stated otherwise.
- When possible, JIT the core loop with **numba** `@nb.njit` for best performance, matching
  the conventions in `agents/general/indicators.md` (explicit `for` loops inside jitted
  functions; no numpy high-level calls inside `@nb.njit`).
- Pre-allocate `output` with length `count` (`np.zeros(count, dtype=np.float64)`).
- Validate that `arr` has at least `count * step` items in the anchored direction.
- Keep the anchor direction explicit in the function name or a parameter so the caller
  always knows which end is fixed and where `output[0]` maps.

### Suggested Signatures

```python
import numpy as np
import numba as nb

@nb.njit
def left_anchored_expanding_window(arr: np.ndarray, step: int, count: int) -> np.ndarray:
    output = np.zeros(count, dtype=np.float64)
    # accumulate over arr[0 : count*step], writing output[j] after (j+1)*step items
    ...
    return output

@nb.njit
def right_anchored_expanding_window(arr: np.ndarray, step: int, count: int) -> np.ndarray:
    output = np.zeros(count, dtype=np.float64)
    # accumulate from the last item leftward, writing output[-1], output[-2], …
    ...
    return output
```

> The exact body depends on the requested **operation**. For multi-column inputs (e.g.
> candle arrays needing both volume and quote), pass the required 1-D columns or the 2-D
> array plus column indices, following the column-index convention in
> `agents/ideas/idea_look_back_look_ahead.md`.

---

## Quick-Reference Cheat Sheet

```
define first : anchor (left/right) + step + count + operation
span used    : count * step items from the anchored end
output length: count
window j     : (j + 1) * step items, measured from the anchored end

left_anchored :
  output[0]       = op(arr[0 : 1*step])
  output[count-1] = op(arr[0 : count*step])

right_anchored:
  output[-1]      = op(arr[-1*step : ])
  output[-count]  = op(arr[-count*step : ])

implementation: growing window → running accumulators, O(count*step), one pass
                more accumulators → multiple operations in one run
                python + numpy, numba @nb.njit when possible
```

---

## Relation to Other Ideas

- **look_back_look_ahead** (`idea_look_back_look_ahead.md`): supplies the candle arrays and
  column-index conventions that anchored expanding windows commonly operate on.
- **normalize_based_on_last_price_clip** (`idea_normalize_based_on_last_price_clip.md`):
  may be applied to window inputs or outputs when normalized features are required.
