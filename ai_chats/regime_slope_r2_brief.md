# Brief: Does (slope, R²) of the recent trend predict the forward move?

**Deliverable:** one Jupyter notebook. Single asset, single horizon, no zigzag.
For each minute snapshot, fit a line to the recent `look_back` of normalized
vwap, get its **slope** and **R²**, then look `look_ahead` minutes ahead and ask
whether price moved in the slope's direction. Summarize as one heatmap table and
two sets of confusion matrices.

**Read first (project files, follow their conventions):**
- `idea_look_back_look_ahead.md` — windowing & snapshot model. Here we use the
  simple `look_back + look_ahead` window (not 1440+240).
- `idea_normalize_based_on_last_price_clip.md` — the `k`/clip normalization.

You already know how to retrieve the assets, normalize vwaps, and use numba.

---

## Configuration

```python
# --- pick exactly ONE asset (uncomment one; use the 7 tickers you retrieve) ---
ASSET = "asset_1"
# ASSET = "asset_2"
# ASSET = "asset_3"
# ASSET = "asset_4"
# ASSET = "asset_5"
# ASSET = "asset_6"
# ASSET = "asset_7"

look_back     = 180      # candles used for the slope/R² regression
look_ahead    = 240      # candles used for the forward vwap
target_return = 0.0020   # 20 bps; barrier threshold on the forward return
k             = 100      # normalization scale: ±0.01 (1%) -> ±1 after clip
clip          = 1.0      # hard clip bound for normalized prices: [-clip, +clip]
stride        = 1        # snapshot step in minutes (see "Overlap" note)
```

| Param | Default | Meaning |
|---|---|---|
| `ASSET` | one of 7 | only one uncommented |
| `look_back` | 180 | regression window length |
| `look_ahead` | 240 | forward-vwap window length |
| `target_return` | 0.0020 | 20 bps barrier on forward return |
| `k` | 100 | maps ±1% to ±1 |
| `clip` | 1.0 | clip normalized prices to [-1, 1] |
| `stride` | 1 | minutes between snapshots |

Per-snapshot data window = `look_back + look_ahead` candles. Extend load
boundaries per `idea_look_back_look_ahead.md` (exclusive mode) so every snapshot
has a full look-back behind and full look-ahead ahead.

---

## Per-snapshot computation

Let the snapshot anchor be the **last look-back candle**.
`current_price = vwap[last look-back candle]` (this is the normalization anchor).

### 1. Look-back → slope and R² (on normalized, clipped vwap)

```python
# look-back vwap series, length look_back, ending at the anchor candle
lb_vwap   = vwap[anchor - look_back + 1 : anchor + 1]
y         = np.clip(k * (lb_vwap / current_price - 1.0), -clip, clip)   # ∈ [-1, 1]

# time axis in 1440-minute units (fixed reference, comparable across horizons)
x         = np.arange(look_back) / 1440.0

slope_raw, r2 = ols(y on x)        # OLS slope and R² ; R² is naturally in [0, 1]
```

**Normalize the slope to [-1, 1].** Derivation (confirm this number):

```
y is clipped to [-1, 1], so the steepest possible line spans the full band
(−1 → +1) over the window. With x in 1440-min units:

  slope_max = (1 − (−1)) / (look_back / 1440) = 2 · 1440 / look_back
  look_back = 180  ->  slope_max = 2·1440/180 = 16

Normalize so slope_max maps to 1:
  slope = slope_raw / (2 · 1440 / look_back)        # = slope_raw · look_back / 2880
  look_back = 180  ->  divide by 16   ->  slope ∈ [-1, 1]
```

Note: the divisor is `2·1440/look_back = 16` at `look_back=180` — i.e. **twice**
`1440/look_back`, because the clip band has full width 2 (from −1 to +1). Clip
the result to `[-1, 1]` for numerical safety.

```python
slope = np.clip(slope_raw / (2 * 1440.0 / look_back), -1.0, 1.0)
```

### 2. Look-ahead → forward vwap and barrier label

The forward vwap is the **volume-weighted average over the next `look_ahead`
candles** (after the anchor):

```python
fwd = slice(anchor + 1, anchor + 1 + look_ahead)
forward_vwap_240 = q[fwd].sum() / v[fwd].sum()      # true vwap = Σ quote_vol / Σ base_vol
# (equivalently the v-weighted mean of per-candle vwap)

raw_fwd_return   = forward_vwap_240 / current_price - 1.0
forward_vwap_norm = np.clip(k * raw_fwd_return, -clip, clip)   # ∈ [-1, 1], for display

# barrier label from the RAW return vs target_return:
#   +1 top   if raw_fwd_return >=  target_return
#   -1 bottom if raw_fwd_return <= -target_return
#    0 time   otherwise
barrier = +1 if raw_fwd_return >= target_return else (-1 if raw_fwd_return <= -target_return else 0)
```

> Scale check: `target_return = 0.002` is a **raw-return** threshold. In normalized
> units it equals `k·target_return = 0.2`, well inside the clip band — so the
> barrier test is equivalent whether applied to `raw_fwd_return` vs `±0.002` or to
> `forward_vwap_norm` vs `±0.2`. Use the raw return to avoid scale confusion.

### 3. Sample row

Store one row per snapshot:

| timestamp | current_price | forward_vwap_240 | raw_fwd_return | forward_vwap_norm | slope | r2 | barrier |
|---|---|---|---|---|---|---|---|

Save the full table to parquet/csv.

---

## Output 1 — the 21 × 11 directional-accuracy table

A grid: **slope on rows (−1.0 → 1.0, step 0.1 → 21 bins)**, **R² on columns
(0.0 → 1.0, step 0.1 → 11 bins)**. Bin by rounding `slope` and `r2` to one
decimal (after the clip), so values land in `{-1.0,…,1.0}` and `{0.0,…,1.0}`.

Each cell holds the **percentage of samples in that (slope, R²) bin whose forward
direction matches the slope direction**:

```
match = sign(raw_fwd_return) == sign(slope)
cell  = 100 * mean(match)   over samples in the bin
```

- Render as a 21×11 heatmap (e.g. diverging colormap centered at 50%).
- **Also render two companion grids:** (a) the **sample count `N`** per cell (so
  sparse cells are visible — mask cells below ~50 samples), and (b) the **mean
  `raw_fwd_return × sign(slope)`** per cell (continuation magnitude, not just
  direction).
- **Zero-slope row (`slope = 0.0`):** direction is undefined; report this row as
  the fraction with `raw_fwd_return > 0` (or mark NaN) and note it.

---

## Output 2 — confusion matrices over an R² gate

Split by slope sign, with the user's "trend validated" truth:

- **Positive slope:** truth `T = (raw_fwd_return >= +target_return)`
- **Negative slope:** truth `T = (raw_fwd_return <= -target_return)`

For each slope sign, sweep an **R² threshold** `τ ∈ {0.0, 0.1, …, 0.9}` (10
values). At each `τ`, the decision is "trust the trend if it's clean enough":

```
D = (r2 >= τ)        # take the directional signal
# 2×2 confusion matrix over that slope-sign subset:
TP = #(D=1, T=1)     # took it, trend validated
FP = #(D=1, T=0)     # took it, did NOT reach the target (reversed or timed out)
FN = #(D=0, T=1)     # skipped, but it would have validated
TN = #(D=0, T=0)     # skipped, correctly
```

This yields **10 confusion matrices for positive slope** and **10 for negative
slope** (20 total). For each, also compute and tabulate:

- precision `= TP/(TP+FP)`  (hit-rate of taken signals)
- recall `= TP/(TP+FN)`
- accuracy `= (TP+TN)/N`
- take-rate `= (TP+FP)/N`

Plot **precision and recall vs τ** (one curve per slope sign) — this is the
practical read: *how much does requiring higher R² raise the directional
hit-rate, and at what cost in coverage?* `τ = 0.0` is the take-everything
baseline (precision = unconditional continuation rate).

> If you instead meant a per-R²-bin breakdown rather than a cumulative gate,
> additionally report, per R² decile and slope sign, the share of outcomes that
> are {continued / reversed / timed-out}. Cheap; include it too.

---

## How to read the result (interpretation guide)

- **Output 1 flat ~50% everywhere** → slope & R² carry little directional
  information at this horizon for this asset. **Brightens toward high R² and/or
  steep slope** → they do; the bright region is where the trend tends to
  continue. Watch for **non-monotonicity** — accuracy may *peak then fall* at the
  steepest/cleanest extremes (exhaustion), in which case the loyal region is
  interior, not at the corner.
- **Output 2:** if precision rises with `τ`, the R² gate is real — cleaner trends
  continue more often. If precision is flat in `τ`, R² adds nothing and only
  slope sign matters. Compare both directions; they need not be symmetric.

---

## Notes

- **Overlap:** at `stride=1`, consecutive snapshots share almost all of their
  look-back and forward windows, so the descriptive percentages are reliable but
  the *effective* sample count is far below the row count — don't attach naive
  significance to small differences. For honest error bars later, re-run with
  `stride ≥ look_ahead` (non-overlapping forward windows). For these tables,
  small `stride` is fine.
- **Single asset, single horizon by design.** `look_back`, `look_ahead`,
  `target_return`, `k`, and `ASSET` are the only knobs; everything else is
  derived.
- Use numba for the per-snapshot regression if the row count is large; the slope,
  R², and forward-vwap aggregations all vectorize cleanly over snapshots.
