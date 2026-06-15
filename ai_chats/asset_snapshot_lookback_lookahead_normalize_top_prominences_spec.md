# Spec: Look-Back/Look-Ahead Normalize **+ KDE Top-Prominence Peak Features**

## Goal

Add a new variant of the look-back/look-ahead feature/label builder that, in
addition to the existing 60-element `fl` vector, computes **6 KDE top-prominence
peak features** (3 above price, 3 below price) using `packages/kde_tools/`, and
stores them at indices **60–65** of an enlarged **66-element** `fl` vector.

This mirrors the KDE peak-finding done interactively in
`notebooks/tests/look_back_look_ahead.ipynb` (cells 5 and 6), but produces the
peaks as numeric features per observation.

Deliverables:

1. `packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize_top_prominences.py`
   — main prepare function **plus all callers in the same file** (no separate
   caller file).
2. `notebooks/studies/lookback_lookahead_features_labels/lookback_lookahead_fl_create_store.ipynb`
   — generates the `fl` data for an asset and stores it to GCS as
   `fl_data_peak_{ASSET}`.
3. Update `agents/datasets/lookback_lookahead_fl.md` to document the 6 new
   features, the new shape `(n_obs, 66)`, and the new GCS key.

---

## 1. Reference material the implementer MUST read first

- `packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize.py` —
  existing njit `asset_snapshot_lookback_lookahead_normalize_prepare` that builds
  the 60-element vector. **Reuse it unchanged** (call it for the first 60
  values). Do not duplicate its logic.
- `packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize_caller.py`
  — the existing 3 callers (`..._single_by_index`,
  `..._single_by_date_string`, `..._all`). The new callers mirror these but
  target the new prepare function and write a 66-wide result.
- `packages/kde_tools/` — exported via `packages.kde_tools.__init__`:
  `make_kernel`, `weighted_histogram`, `convolve_same`, `compute_kde`,
  `top_kde_peaks`, `kde_peaks_above_below`. Use **`compute_kde`** and
  **`kde_peaks_above_below`**; do not re-implement KDE/peak logic.
- `notebooks/tests/look_back_look_ahead.ipynb` cells 1, 5, 6 — source of the KDE
  parameter defaults and the above/below top-3 semantics.
- `notebooks/studies/lookback_lookahead_features_labels/lookback_lookahead_data_process.ipynb`
  — template for the new notebook.
- `agents/datasets/lookback_lookahead_fl.md` — dataset doc to update.

---

## 2. Background: source array and existing 60-vector

Source candle array `arr` has shape `(N, 12)`, columns:

```
ts=0, c=1, v=2, q=3, vwap=4, vb=5, vs=6, v_rzs=7, v_ma=8,
rsi_p7=9, rsi_p14=10, rsi_p60=11
```

The existing `asset_snapshot_lookback_lookahead_normalize_prepare(arr_slice,
look_back, look_ahead, k_scaler)` takes a slice of shape
`(look_back + look_ahead, 12)` and returns a `float64` vector of length **60**.
Indices 0–59 are already specified in `lookback_lookahead_fl.md` and must remain
**byte-for-byte identical** in the new variant.

Per-observation normalization anchor:

```
last_candle_price = arr_slice[look_back - 1, 4]   # vwap of last look-back candle (== price_l)
```

---

## 3. The 6 new features (indices 60–65)

Reproduce the KDE construction and the above/below top-3-prominence peak
detection from the test notebook, then store the **peak price positions**
(normalized vwap, in `[-1, 1]`) as features.

| Index | Name | Meaning |
|-------|------|---------|
| 60 | `f_kde_above_peak_1` | price of the 1st-highest-prominence KDE peak **above** price |
| 61 | `f_kde_above_peak_2` | price of the 2nd-highest-prominence KDE peak above price |
| 62 | `f_kde_above_peak_3` | price of the 3rd-highest-prominence KDE peak above price |
| 63 | `f_kde_below_peak_1` | price of the 1st-highest-prominence KDE peak **below** price |
| 64 | `f_kde_below_peak_2` | price of the 2nd-highest-prominence KDE peak below price |
| 65 | `f_kde_below_peak_3` | price of the 3rd-highest-prominence KDE peak below price |

Ordering: peaks are ranked by **descending prominence** (peak_1 = most
prominent), exactly as `kde_tools.top_kde_peaks` returns them.

**Missing-peak defaults** (fewer than 3 peaks found on a side):

- Above positions with no peak → **`+1.0`**
- Below positions with no peak → **`-1.0`**

So `fl[60:63]` default to `1.0` and `fl[63:66]` default to `-1.0`, then
overwritten by however many peaks were actually found.

### 3.1 KDE input construction (per observation)

Using the look-back slice `arr_slice[:look_back]`:

```python
price_l   = arr_slice[look_back - 1, 4]                       # vwap anchor
vwap_lb   = arr_slice[:look_back, 4]
scaled_lb = np.clip(k_scaler * (vwap_lb - price_l) / price_l, -1.0, 1.0)

v_lb      = arr_slice[:look_back, 2]                          # volume column
v_lb_norm = v_lb / v_lb.mean()                                # daily-avg-normalized weights
```

> Note: weight scaling by a positive constant does not change peak positions or
> prominence ordering; `v_lb_norm` is used to match the notebook. (Raw `v_lb`
> would yield identical peaks — implementer may use either, but spec the
> normalized form for fidelity.)

### 3.2 KDE + peaks (use kde_tools)

```python
from packages.kde_tools import compute_kde, kde_peaks_above_below

kde_res = compute_kde(
    scaled_lb,
    v_lb_norm,
    bins=kde_bins,                 # default 200
    kernel_type=kde_kernel,        # default "Triangular"
    bandwidth=kde_bandwidth,       # default 5
    range_min=-1.0,
    range_max=1.0,
    ignore_borders=kde_ignore_borders,   # default True
)

peaks = kde_peaks_above_below(
    kde_res["kde"],
    kde_res["bin_centers"],
    distance=kde_peak_distance,    # default = kde_bandwidth (5)
    n=3,
    split_at=0.0,                  # current price normalizes to 0.0
)
# peaks: {"above_prices","above_proms","below_prices","below_proms"}
```

### 3.3 Fill indices 60–65

```python
fl[60] = fl[61] = fl[62] = 1.0     # above defaults
fl[63] = fl[64] = fl[65] = -1.0    # below defaults

for j, p in enumerate(peaks["above_prices"][:3]):
    fl[60 + j] = p
for j, p in enumerate(peaks["below_prices"][:3]):
    fl[63 + j] = p
```

---

## 4. File: `asset_snapshot_lookback_lookahead_normalize_top_prominences.py`

Place in `packages/asset_analyzer/`. **All functions (prepare + callers) live in
this one file.**

### 4.1 Why this file is NOT njit

`kde_tools.kde_peaks_above_below` uses **scipy** (`find_peaks`,
`peak_prominences`), which cannot run inside `@nb.njit` / `nb.prange`.
Therefore the prepare function and the `_all` caller are **plain Python**. The
heavy inner pieces are still jitted internally: the 60-vector via the existing
njit `asset_snapshot_lookback_lookahead_normalize_prepare`, and the KDE
histogram/convolution via `kde_tools`' jitted `weighted_histogram` /
`convolve_same`.

### 4.2 KDE parameters (new function arguments + defaults)

Pass these through every function so callers can override them. Defaults come
from `look_back_look_ahead.ipynb` cell 1:

| Parameter | Default | Source |
|-----------|---------|--------|
| `kde_bins` | `200` | `BINS_COUNT` |
| `kde_bandwidth` | `5` | `BANDWIDTH` |
| `kde_kernel` | `"Triangular"` | `KERNEL` |
| `kde_ignore_borders` | `True` | `KDE_IGNORE_BORDERS` |
| `kde_peak_distance` | `5` | notebook uses `distance=BANDWIDTH` |

`n` (peaks per side) is fixed at **3** (the feature layout hard-codes 3+3).
`split_at` is fixed at **0.0**.

### 4.3 Main prepare function

```python
def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare(
    arr,                 # slice shape (look_back + look_ahead, 12)
    look_back,
    look_ahead,
    k_scaler,
    kde_bins=200,
    kde_bandwidth=5,
    kde_kernel="Triangular",
    kde_ignore_borders=True,
    kde_peak_distance=5,
) -> np.ndarray:         # returns float64 vector length 66
```

Steps:

1. `base = asset_snapshot_lookback_lookahead_normalize_prepare(arr, look_back, look_ahead, k_scaler)` (length 60).
2. Allocate `fl = np.zeros(66, dtype=np.float64)`; copy `fl[:60] = base`.
3. Build `scaled_lb`, `v_lb_norm` (§3.1).
4. `compute_kde` + `kde_peaks_above_below` (§3.2).
5. Fill `fl[60:66]` (§3.3).
6. Return `fl`.

### 4.4 Callers (same file) — mirror the existing caller file

```python
def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_single_by_index(
    arr, look_back, look_ahead, k_scaler, index, **kde_kwargs): ...

def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_single_by_date_string(
    arr, look_back, look_ahead, k_scaler, date_string, **kde_kwargs): ...

def asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_all(
    arr, look_back, look_ahead, k_scaler, **kde_kwargs): ...
```

- `_single_by_index`: slice `arr[index - look_back + 1 : index + look_ahead + 1]`
  and call the prepare function. (Same slicing as the existing caller.)
- `_single_by_date_string`: identical binary-search-on-`ts` logic as the
  existing caller; on match call `_single_by_index`. Reuse the existing code's
  structure (`np.datetime64(date_string, 'ms')`, error if not found).
- `_all`: **plain Python loop** (NOT `nb.prange`), build
  `result = np.zeros((n_obs, 66), dtype=np.float64)` where
  `n_obs = arr.shape[0] - look_back - look_ahead + 1`, fill
  `result[i - (look_back - 1)] = prepare(slice...)` for
  `i in range(look_back - 1, arr.shape[0] - look_ahead)`.
  - **Performance note (include as a code comment):** scipy peak-finding runs
    per observation, so this is much slower than the njit/prange original
    (~1M observations for btcusdt). A simple loop is acceptable for this spec.
    The implementer MAY add optional parallelism (e.g. chunked
    `concurrent.futures`/`joblib`) **only if** it does not change results;
    keep the default path a straightforward loop and report wall-clock time.

### 4.5 Exports

Add the new prepare + 3 callers to `packages/asset_analyzer/__init__.py`.

### 4.6 Dependencies

`packages/asset_analyzer/requirements.txt` already lists `numpy`, `numba`,
`scipy`. Add nothing unless missing; `kde_tools` shares the same deps. Confirm
`scipy` is present (it is).

---

## 5. Notebook: `lookback_lookahead_fl_create_store.ipynb`

Location:
`notebooks/studies/lookback_lookahead_features_labels/lookback_lookahead_fl_create_store.ipynb`

Model it on `lookback_lookahead_data_process.ipynb`. Cells:

1. **`%pip install`** (rules.md): `%pip install -q numpy numba scipy google-cloud-storage google-auth`
2. **Repo clone + path setup** (rules.md branch pattern). This notebook is on
   branch `claude/dreamy-fermat-mnqjs9`, so the clone cell MUST use:
   ```python
   BRANCH_NAME = "claude/dreamy-fermat-mnqjs9"
   if not os.path.exists(REPO_NAME):
       !git clone -b {BRANCH_NAME} {REPO_URL}
   ```
   Also `sys.path.insert(0, os.path.join(REPO_PATH, "packages/tools/google_cloud_storage_tools"))`
   and `gcs_json_key_file()`.
3. **Parameters**: `ASSET = "btcusdt"` (with the other 6 assets commented out,
   like the template); `GCS_BUCKET = "payamdprojectbucket"`;
   `GCS_KEY = f"lookback_lookahead_{ASSET}"`.
4. **Load source candles** from `gs://payamdprojectbucket/lookback_lookahead_{ASSET}`
   via `read_file` + `np.load(io.BytesIO(...))` → `candles` shape `(N, 12)`.
5. **Params**: `LOOK_BACK = 1440`, `LOOK_AHEAD = 240`, `K = 100`. Optionally
   surface KDE params (`KDE_BINS=200`, `KDE_BANDWIDTH=5`, `KDE_KERNEL="Triangular"`,
   `KDE_IGNORE_BORDERS=True`, `KDE_PEAK_DISTANCE=5`) for clarity.
6. **Compute**:
   ```python
   from packages.asset_analyzer import asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_all
   fl = asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_all(
       candles, LOOK_BACK, LOOK_AHEAD, K)
   ```
   Print `fl.shape` (expect `(n_obs, 66)`) and elapsed time.
7. **Store** to GCS as `fl_data_peak_{ASSET}`:
   ```python
   GCS_KEY = f"fl_data_peak_{ASSET}"
   buf = io.BytesIO(); np.save(buf, fl); buf.seek(0)
   write_file("payamdprojectbucket", GCS_KEY, buf)
   ```
8. **(Optional) Sanity plot**: histogram of the 6 new columns (`fl[:, 60:66]`)
   so the distributions of above/below peak positions are visible. Keep it
   matplotlib (this is a study notebook, not a report).

> Heads-up for the implementer: the full `_all` run over btcusdt (~1M obs) with
> per-observation scipy peak-finding can be slow. The notebook is still correct;
> just note the runtime. Consider testing on a small slice of `candles` first.

---

## 6. Update `agents/datasets/lookback_lookahead_fl.md`

Make the doc describe both the original 60-wide dataset and the new 66-wide
"peak" variant. Required edits:

- **Identity table**: note the peak variant — key pattern `fl_data_peak_{asset}`,
  shape `(n_obs, 66)`. Keep the original `fl_data_{asset}` / `(n_obs, 60)` rows.
- **Vector Layout**: add rows for indices 60–65:

  | Index | Name | Description |
  |-------|------|-------------|
  | 60 | `f_kde_above_peak_1` | normalized vwap price of top-1-prominence KDE peak above price (default `+1.0` if none) |
  | 61 | `f_kde_above_peak_2` | top-2 above (default `+1.0`) |
  | 62 | `f_kde_above_peak_3` | top-3 above (default `+1.0`) |
  | 63 | `f_kde_below_peak_1` | normalized vwap price of top-1-prominence KDE peak below price (default `−1.0` if none) |
  | 64 | `f_kde_below_peak_2` | top-2 below (default `−1.0`) |
  | 65 | `f_kde_below_peak_3` | top-3 below (default `−1.0`) |

- **New "KDE Peak Features" section** explaining: volume-weighted KDE over
  normalized look-back vwap (`scaled_lb` clipped to `[-1,1]`, weights =
  `v / v.mean()`), borders ignored by default; peaks split at `0.0` into
  above/below; top-3 by prominence per side; defaults `+1`/`−1`. List the KDE
  parameter defaults (`bins=200`, `bandwidth=5`, kernel `Triangular`,
  `ignore_borders=True`, `peak_distance=5`). Reference `packages/kde_tools/`.
- **Index Constants** block: append
  ```python
  F_KDE_ABOVE_PEAK_1_I = 60
  F_KDE_ABOVE_PEAK_2_I = 61
  F_KDE_ABOVE_PEAK_3_I = 62
  F_KDE_BELOW_PEAK_1_I = 63
  F_KDE_BELOW_PEAK_2_I = 64
  F_KDE_BELOW_PEAK_3_I = 65
  ```
- **Source & Processing**: add that the peak variant is produced by
  `asset_snapshot_lookback_lookahead_normalize_top_prominences_prepare_all` in
  `packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize_top_prominences.py`,
  regenerated via `lookback_lookahead_fl_create_store.ipynb`, stored at
  `gs://payamdprojectbucket/fl_data_peak_{asset}`.
- **How to Load**: add a snippet loading `fl_data_peak_{asset}` → shape
  `(n_obs, 66)`.

---

## 7. Acceptance criteria

1. New file exists with prepare + 3 callers, all in one file; `__init__.py`
   exports them.
2. `prepare` returns a length-**66** float64 vector; `fl[:60]` exactly equals
   the original function's output for the same inputs.
3. Indices 60–62 hold above-peak prices (default `+1.0`), 63–65 hold below-peak
   prices (default `−1.0`), ordered by descending prominence, computed via
   `kde_tools.compute_kde` + `kde_peaks_above_below` with the specified defaults.
4. `_all` returns shape `(n_obs, 66)` via a plain Python loop (no `nb.prange`).
5. `_single_by_date_string` reproduces the existing binary-search behavior.
6. Notebook runs end-to-end and stores `fl_data_peak_{ASSET}`; clone cell pins
   the working branch.
7. `lookback_lookahead_fl.md` documents the 6 new features, `(n_obs, 66)` shape,
   index constants, and the `fl_data_peak_{asset}` key.
8. Branch `claude/dreamy-fermat-mnqjs9`, committed and pushed.

---

## 8. Out of scope

- Do not modify the existing 60-wide function/callers or `fl_data_{asset}`
  artifacts.
- Do not change `kde_tools`.
- No model training / reporting changes.
