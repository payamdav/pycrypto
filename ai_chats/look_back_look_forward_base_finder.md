# Task: Create `notebooks/tests/look_back_look_forward_base_finder.ipynb`

Before writing any code, read the following project files in full — they are the
authoritative specification for every convention, formula, and naming rule used in
this codebase:

- `@agents/datasets/assets.md` — supported assets and lowercase folder names
- `@agents/datasets/huggingface_candles.md` — dataset identity, column schema, URL pattern, and the `load_range()` helper
- `@agents/ideas/idea_look_back_look_ahead.md` — windowing pattern: parameters, `last_candle`, `current_time`, `price_l`, inclusive vs exclusive boundaries, loop / vectorized / chunked-vectorized modes
- `@agents/ideas/idea_normalize_based_on_last_price_clip.md` — price normalization formula, hard vs tanh clipping, k derivation, vectorized 2-D application, time normalization vector, inverse transform

---

## Cells

### Cell 0 — pip install

```python
%pip install -q huggingface_hub pyarrow pandas matplotlib requests numpy scipy
```

---

### Cell 1 — Parameters

```python
ASSET        = "BTCUSDT"
LOOK_BACK    = 1440
LOOK_AHEAD   = 240
DATETIME     = "2025-12-12 20:00:00"   # current_time (UTC)
WINDOW_MODE  = "exclusive"
NORM_MODE    = "clip"                  # "clip" | "tanh"
K            = 100

BINS_COUNT         = 200
BANDWIDTH          = 5
KERNEL             = "Triangular"      # "Triangular" | "Epanechnikov" | "Uniform"
KDE_IGNORE_BORDERS = True              # exclude ±1.0 clipped prices from KDE histogram
```

---

### Cell 2 — Load data and compute

- Define `load_range(asset, start, end, columns=None)` exactly as specified in
  `huggingface_candles.md`.
- Derive `current_time`, `last_candle_time`, `lb_start_time`, `lb_end_time`,
  `la_start_time`, `la_end_time` from `DATETIME` and the parameters per
  `idea_look_back_look_ahead.md`.
- Load only columns `["ts", "vwap"]` — no volume columns needed.
- Split into `df_lb` (exactly `LOOK_BACK` rows) and `df_la` (exactly `LOOK_AHEAD` rows).
- Assert both lengths are correct.
- Compute price normalization per `idea_normalize_based_on_last_price_clip.md`:
  - `price_l = df_lb["vwap"].values[-1]`
  - `raw_lb = (df_lb["vwap"].values - price_l) / price_l`; same for look-ahead
  - Apply `NORM_MODE` (`clip` → `np.clip(K * raw, -1, 1)`; `tanh` → `np.tanh(K * raw)`)
- Compute time normalization per the same idea file:
  - `t    = np.arange(LOOK_BACK) / LOOK_BACK`
  - `t_la = (LOOK_BACK + np.arange(LOOK_AHEAD)) / LOOK_BACK`
  - `t_la[0]` must equal exactly `1.0` — the first look-ahead candle opens at `current_time`
- Compute KDE using unit weights (no volume in this notebook — `weights=np.ones(len(kde_prices))`):
  - If `KDE_IGNORE_BORDERS=True`, exclude prices at exactly `±1.0` before histogramming
  - `counts, bin_edges = np.histogram(kde_prices, bins=BINS_COUNT, range=(-1,1), weights=kde_weights)`
  - Build a kernel from `KERNEL` and `BANDWIDTH` (Triangular / Epanechnikov / Uniform);
    normalize it so `kernel.sum() == 1` (mass conservation)
  - `kde = np.convolve(counts, kernel_arr, mode="same")`
  - `bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2`
- Print summary: `price_l`, `K`, `t[0]`, `t[-1]`, `t_la[0]`, `t_la[-1]`,
  `scaled_lb` range, `scaled_la` range, KDE mass.

---

### Cell 3 — Border datetimes

Print a formatted block showing:

- `current_time`
- `last_candle` open time
- look-back start / end
- look-ahead start / end

Use `pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")` for formatting.

---

### Cell 4 — Original VWAP charts

Side-by-side charts using `GridSpec(1, 2, width_ratios=[LOOK_BACK, LOOK_AHEAD], wspace=0.04)`.

- Both axes share the same y-limits: compute `vwap_all = concat(lb_vwap, la_vwap)`,
  add 5% margin, apply with `set_ylim` on both axes.
- Look-back in teal `#26a69a`, look-ahead in red `#ef5350`.
- x-tick labels: `HH:MM` only, `fontsize=6`; at most 6 ticks per axis.
- Look-ahead y-axis on the right side.
- `wspace=0.04` so the two panels sit close together giving a proportional sense of time.

---

### Cell 5 — KDE peak detection

Use `scipy.signal.find_peaks` and `peak_prominences`:

- Split KDE into a positive sub-array (`bin_centers >= 0`) and negative sub-array
  (`bin_centers < 0`).
- Call `find_peaks` on each with `distance=BANDWIDTH`.
- Rank by prominence descending; keep top 3.
- Print peak prices and prominences for both series.
- `PEAK_COLORS = ["red", "orange", "yellow"]` (rank 1 = red, rank 2 = orange, rank 3 = yellow).

---

### Cell 6 — KDE + Normalized VWAP chart

Layout: `GridSpec(1, 3, width_ratios=[LOOK_AHEAD, LOOK_BACK, LOOK_AHEAD], wspace=0.04)` — 1 row, 3 columns.

- `ax_kde  = gs[0, 0]` — horizontal bar chart of KDE (bars flow left→right, y-axis = scaled price)
- `ax_lb_n = gs[0, 1]`, `sharey=ax_kde` — look-back normalized VWAP line, x-axis = `t`
- `ax_la_n = gs[0, 2]`, `sharey=ax_kde` — look-ahead normalized VWAP, x-axis = `t_la`

KDE chart:
- `ax_kde.barh(bin_centers, kde, height=bin_width*0.95, color="#26a69a", alpha=0.85)`
- Dashed zero line at y=0; dotted boundary lines at y=±1.
- Title shows kernel name, bandwidth, and whether borders were excluded.

Normalized VWAP charts:
- Draw peak price `axhline` on all three axes using `PEAK_COLORS` (positive peaks from
  positive KDE series, negative peaks from negative KDE series).
- Dashed zero line and dotted ±1 lines on each axis.
- `set_ylim(-1.08, 1.08)` is guaranteed structurally by `sharey=ax_kde`.
- x-tick labels: `HH:MM`, fontsize=6, at most 6 ticks per axis (derive from `ts_lb` /
  `ts_la` DatetimeIndex).
- Look-ahead y-axis on the right.
- Figure title: `f"KDE + Normalized VWAP  |  price_l={price_l:.4f}   k={K}   current_time={DATETIME} UTC"`.

---

## Constraints

- Do **not** add volume, DVR, or any oscillator charts — this notebook intentionally omits them.
- All formulas must follow `idea_normalize_based_on_last_price_clip.md` exactly —
  especially `price_l = lb[-1]` as the anchor for both windows, and `t_la[0] = 1.0`.
- Use the `load_range()` pattern from `huggingface_candles.md` verbatim.
- No comments except where a non-obvious constraint or invariant needs explanation.
- After creating the notebook, commit and push to branch
  `claude/add-candle-downloader-script-nrORo` on `payamdav/pycrypto`.
