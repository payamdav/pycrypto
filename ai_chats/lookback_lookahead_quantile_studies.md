# Spec: Quantile-Label Lookback/Lookahead Studies (LSTM & GRU)

## Goal

Create **4 new study folders** as siblings of
`scripts/studies/lookback_lookahead_nn/`. Each new folder is a near-copy of that
study with specific modifications. Additionally, add a one-time **local data
download** mechanism so the shared `fl_data` is pulled from GCS only once and all
studies load it from `~/data/`.

The 4 new study folders (names are authoritative — they are reused verbatim as
the GCS folder name, `OBSERVATION_SET_NAME`, web prefix, and config default):

1. `scripts/studies/lookback_lookahead_lstm_singlehead_quantile`
2. `scripts/studies/lookback_lookahead_lstm_singlehead_quantile_mse`
3. `scripts/studies/lookback_lookahead_gru_singlehead_quantile`
4. `scripts/studies/lookback_lookahead_gru_singlehead_quantile_mse`

Per-study differences:

| Study folder | RNN cell | Training loss |
|---|---|---|
| `..._lstm_singlehead_quantile`     | LSTM | Huber (delta=1.0) |
| `..._lstm_singlehead_quantile_mse` | LSTM | MSE |
| `..._gru_singlehead_quantile`      | GRU  | Huber (delta=1.0) |
| `..._gru_singlehead_quantile_mse`  | GRU  | MSE |

ALL four apply the **quantile label transform** (below). The only difference
between the `_mse` and non-`_mse` variants is the training loss function. The
only difference between `gru` and `lstm` is the RNN cell.

Each new study folder must contain exactly the same file set as the existing
study:

```
lookback_lookahead_nn.py            # single-process trainer (also the "base" module)
lookback_lookahead_nn_parallel.py   # CUDA-streams trainer
lookback_lookahead_nn_mp.py         # torch.multiprocessing trainer
upload_web_app.py
do_workflow.sh
pod_self_terminate.py
requirements.txt
web/                                # full copy of the web SPA folder
```

> **Keep the three trainer filenames identical** to the originals
> (`lookback_lookahead_nn.py`, `..._parallel.py`, `..._mp.py`). The parallel and
> mp scripts do `import lookback_lookahead_nn as base`, and because each script
> inserts its own directory at `sys.path[0]`, that import resolves to the local
> folder's copy. Do not rename these files.

---

## Part A — One-time local data cache (`~/data/`)

### A1. New download script (in the EXISTING base folder)

Create `scripts/studies/lookback_lookahead_nn/download_fl_data.py`. It copies
every asset's `fl_data_{asset}` blob from GCS bucket `payamdprojectbucket` to
`~/data/fl_data_{asset}.npy`, skipping any that already exist (unless `--force`).

```python
#!/usr/bin/env python3
"""
One-time copy of every asset's fl_data blob from GCS to the local ~/data cache.

All lookback/lookahead studies share the SAME fl_data. Downloading it from GCS
for every observation/run is slow, so this script pulls each asset's blob from
gs://payamdprojectbucket/fl_data_{asset} ONCE and writes it to
~/data/fl_data_{asset}.npy. The study trainers then load from ~/data (see each
study's load_fl_data), eliminating repeated GCS downloads.

Data spec: agents/datasets/lookback_lookahead_fl.md
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_GCS_TOOLS_DIR = os.path.join(
    _REPO_ROOT, "packages", "tools", "google_cloud_storage_tools")
if _GCS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _GCS_TOOLS_DIR)

from gcs_tools import gcs_json_key_file, read_file  # noqa: E402

DATA_BUCKET = "payamdprojectbucket"
LOCAL_DATA_DIR = os.path.expanduser("~/data")
ASSETS = ["btcusdt", "ethusdt", "trumpusdt", "vineusdt",
          "adausdt", "xrpusdt", "dogeusdt"]


def main():
    parser = argparse.ArgumentParser(
        description="Copy fl_data blobs from GCS to the local ~/data cache.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the local file exists.")
    args = parser.parse_args()

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    gcs_json_key_file()  # resolve credentials once before any GCS call

    for asset in ASSETS:
        local_path = os.path.join(LOCAL_DATA_DIR, f"fl_data_{asset}.npy")
        if os.path.exists(local_path) and not args.force:
            print(f"  skip {asset}: already at {local_path}", flush=True)
            continue
        try:
            print(f"  downloading fl_data_{asset} ...", flush=True)
            data_bytes = read_file(DATA_BUCKET, f"fl_data_{asset}")
            with open(local_path, "wb") as fh:
                fh.write(data_bytes)
            print(f"  saved -> {local_path} "
                  f"({len(data_bytes) / (1024 ** 2):.1f} MiB)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {asset}: {exc}", flush=True)

    print("\nDone. Local cache dir: " + LOCAL_DATA_DIR, flush=True)


if __name__ == "__main__":
    main()
```

### A2. `load_fl_data` now reads from `~/data` (with GCS fallback)

In **every** study's base module `lookback_lookahead_nn.py` (the existing base
study AND all 4 new studies), replace the `load_fl_data` function so it loads
from the local cache, falling back to a one-time GCS download that also
populates the cache. Add a `LOCAL_DATA_DIR` constant near the other constants.

```python
LOCAL_DATA_DIR = os.path.expanduser("~/data")
```

```python
def load_fl_data(asset: str) -> np.ndarray:
    """Load fl_data_{asset} as a float64 array of shape (n_obs, 60).

    Reads from the local ~/data cache when present (populated once by
    scripts/studies/lookback_lookahead_nn/download_fl_data.py). If the local
    file is missing, downloads the blob from GCS exactly once, writes it to the
    cache, and loads it from there.
    """
    local_path = os.path.join(LOCAL_DATA_DIR, f"fl_data_{asset}.npy")
    if os.path.exists(local_path):
        fl = np.load(local_path)
    else:
        data_bytes = read_file(DATA_BUCKET, f"fl_data_{asset}")
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(data_bytes)
        fl = np.load(local_path)
    return np.ascontiguousarray(fl, dtype=np.float64)
```

> The existing base study keeps importing `read_file` already, so the fallback
> compiles there. This change is non-breaking for the existing study (the GCS
> fallback runs if `~/data` is empty). The `io` import is no longer strictly
> needed by `load_fl_data` but leave existing imports as-is to avoid churn.

---

## Part B — The quantile label transform (core change for all 4 studies)

### Concept

Original labels `l_e_vwap[k]` are already in `[-1, 1]`. For TRAINING ONLY,
transform them to a (near) uniform distribution in `[-1, 1]`:

```
Y_train = QuantileTransformer(uniform).fit_transform(Y) * 2 - 1
```

Rules:

- **Fit the transformer on the TRAIN split labels only** (no leakage from
  val/test). One transformer is fit **per observation** (per asset × label
  combination), on that observation's train-split single label column.
- Transform **train and val** labels with that fitted transformer (the model
  trains on transformed labels and early-stops on transformed val loss).
- Keep the **test** labels in ORIGINAL space.
- The model output (tanh) is in transformed space. For evaluation/reporting,
  **inverse-transform the predictions back to original label space**, then run
  the entire Eval_/Report_ suite (metrics, bins, confusion, directional
  accuracy, etc.) in ORIGINAL space against the original test labels.

### Helpers to add to each new study's base module (`lookback_lookahead_nn.py`)

Add `from sklearn.preprocessing import QuantileTransformer` to the imports, and
these module-level helpers (place them near the data-prep helpers):

```python
QUANTILE_N = 1000  # number of quantile knots for the transformer

def fit_label_transform(y_train: np.ndarray) -> QuantileTransformer:
    """Fit a uniform QuantileTransformer on the TRAIN-split label column."""
    n = y_train.shape[0]
    qt = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(QUANTILE_N, n),
        subsample=min(n, 1_000_000),
        random_state=42,
    )
    qt.fit(y_train.reshape(-1, 1))
    return qt


def transform_labels(qt: QuantileTransformer, y: np.ndarray) -> np.ndarray:
    """Map labels to a (near) uniform distribution in [-1, 1] for training."""
    u = qt.transform(y.reshape(-1, 1)).ravel()   # uniform in [0, 1]
    return (u * 2.0 - 1.0).astype(np.float64)     # -> [-1, 1]


def inverse_transform_labels(qt: QuantileTransformer,
                             y_scaled: np.ndarray) -> np.ndarray:
    """Invert transform_labels: predictions in [-1, 1] -> original label space."""
    u = (np.clip(y_scaled, -1.0, 1.0) + 1.0) / 2.0   # back to [0, 1]
    orig = qt.inverse_transform(u.reshape(-1, 1)).ravel()
    return orig.astype(np.float64)
```

---

## Part C — Per-study base module (`lookback_lookahead_nn.py`)

Start from a copy of the existing
`scripts/studies/lookback_lookahead_nn/lookback_lookahead_nn.py`, then apply ALL
of the following changes. To keep the parallel/mp scripts identical across all 4
studies, push every per-study difference into THIS base module via the constants
and factory functions below.

### C1. Constants (per study)

Set `OBSERVATION_SET_NAME` to the study's folder name, and add the new
constants. Use this table:

| Study | `OBSERVATION_SET_NAME` | `MODEL_TYPE` | `LOSS_NAME` |
|---|---|---|---|
| lstm quantile     | `lookback_lookahead_lstm_singlehead_quantile`     | `LSTM_Base_2_64_SingleHead_Quantile`     | `Huber Loss` |
| lstm quantile mse | `lookback_lookahead_lstm_singlehead_quantile_mse` | `LSTM_Base_2_64_SingleHead_Quantile_MSE` | `MSE Loss` |
| gru quantile      | `lookback_lookahead_gru_singlehead_quantile`      | `GRU_Base_2_64_SingleHead_Quantile`      | `Huber Loss` |
| gru quantile mse  | `lookback_lookahead_gru_singlehead_quantile_mse`  | `GRU_Base_2_64_SingleHead_Quantile_MSE`  | `MSE Loss` |

Also add this constant to every new study's base module (same value in all 4):

```python
LABEL_TRANSFORM_NAME = "quantile_uniform_scaled[-1,1]"
```

`REPORTS_SUBDIR`, `REPORTS_PREFIX`, `DATA_BUCKET`, `REPORT_BUCKET` stay as-is
(prefix derives from `OBSERVATION_SET_NAME`).

### C2. Model class + factory

For **LSTM** studies keep `LSTMRegressor` as-is. For **GRU** studies replace it
with a `GRURegressor` (swap `nn.LSTM` → `nn.GRU`; note `nn.GRU` returns
`(output, h_n)` — there is no cell state):

```python
class GRURegressor(nn.Module):
    """Lean 2-layer GRU with a single tanh-bounded regression output."""

    def __init__(self, n_features: int, hidden: int = HIDDEN_UNITS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.gru1 = nn.GRU(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        seq, _ = self.gru1(x)
        seq = self.drop1(seq)
        seq, h_n = self.gru2(seq)       # GRU returns h_n (no cell state)
        last = self.drop2(h_n[-1])
        out = torch.tanh(self.head(last))
        return out.squeeze(-1)
```

In **all 4** studies add a factory so parallel/mp never reference the concrete
class name:

```python
def build_model(n_features: int):
    """Construct the study's RNN regressor."""
    return LSTMRegressor(n_features=n_features)   # GRU studies: GRURegressor(...)
```

### C3. Loss factory

Add a criterion factory so parallel/mp never hardcode the loss:

```python
def make_criterion():
    """The study's training loss."""
    return nn.HuberLoss(delta=1.0)   # _mse studies: return nn.MSELoss()
```

Replace the criterion construction inside `train_model` with
`criterion = make_criterion()`. In the `run_observation` NO_TRAIN branch replace
`nn.HuberLoss(delta=1.0)` with `make_criterion()`.

### C4. Convergence-plot y-axis label

In `fig_convergence`, change the hardcoded `yaxis_title="Huber Loss"` to
`yaxis_title=LOSS_NAME` so MSE studies label the axis correctly.

### C5. `run_observation` — integrate the quantile transform

Modify `run_observation` so:

- After the split, fit the transformer on the train labels and transform
  train/val labels; build the train/val loaders from the TRANSFORMED labels and
  the test loader from the ORIGINAL labels:

```python
    qt = fit_label_transform(y_tr)
    y_tr_t = transform_labels(qt, y_tr)
    y_va_t = transform_labels(qt, y_va)

    train_loader = make_loader(x_tr, y_tr_t, shuffle=True)
    val_loader = make_loader(x_va, y_va_t, shuffle=False)
    test_loader = make_loader(x_te, y_te, shuffle=False)   # ORIGINAL labels
```

- Use the model factory: `model = build_model(n_features)`.
- After predicting, inverse-transform the predictions back to original space
  BEFORE any Eval_ call (`y_true` stays original, from the test loader):

```python
    y_true, y_pred = predict(model, test_loader)
    y_pred = inverse_transform_labels(qt, y_pred)   # transformed -> original
```

- In `model_arch`, set `"model_type": MODEL_TYPE`.
- In `telemetry`, add `"label_transform": LABEL_TRANSFORM_NAME`.
- In the NO_TRAIN branch keep using `make_criterion()`; the rest of the dry-run
  path is unchanged (it still inverse-transforms predictions for plumbing).

Everything else in `run_observation` (eval modules, Plotly figs, metrics dict,
report upload, model-param upload, memory cleanup) stays the same.

---

## Part D — Per-study parallel trainer (`lookback_lookahead_nn_parallel.py`)

Start from a copy of the existing parallel script. Apply these edits so it works
with the factories + quantile transform. After these edits the file is
**identical across all 4 studies** (all per-study choices come from `base`).

1. **Observation.__init__** — fit the transformer on the train labels, transform
   train/val, keep test original, and store the transformer on the observation:

```python
        self.qt = base.fit_label_transform(y_all[tr_s:tr_e])
        self.x_tr = to_dev(x_all[tr_s:tr_e])
        self.y_tr = to_dev(base.transform_labels(self.qt, y_all[tr_s:tr_e]))
        self.x_va = to_dev(x_all[va_s:va_e])
        self.y_va = to_dev(base.transform_labels(self.qt, y_all[va_s:va_e]))
        self.x_te = to_dev(x_all[te_s:te_e])
        self.y_te = to_dev(y_all[te_s:te_e])   # ORIGINAL labels
```

   `free_device_tensors` is unchanged — it must NOT clear `self.qt` (the
   transformer is a small host object reused by `report_observation`).

2. **_init_training_state** — build the model via the factory:

```python
    model = base.build_model(obs.n_features).to(device)
```

3. **train_wave** — replace `criterion = nn.HuberLoss(delta=1.0)` with
   `criterion = base.make_criterion()`.

4. **report_observation** — inverse-transform predictions to original space
   before the Eval_ calls:

```python
    y_true = obs.y_te.cpu().numpy()                       # original
    y_pred = _predict_resident(obs.model, obs.x_te)       # transformed space
    y_pred = base.inverse_transform_labels(obs.qt, y_pred)  # -> original
```

5. **report_observation** — set `model_arch["model_type"] = base.MODEL_TYPE`
   (replace the hardcoded `"LSTM_Base_2_64_SingleHead"`), and add
   `"label_transform": base.LABEL_TRANSFORM_NAME` to the `telemetry` dict. Keep
   the existing `"trainer": "parallel_gpu_resident"` note.

No other logic changes. The unused `nn` import may remain.

---

## Part E — Per-study multiprocessing trainer (`lookback_lookahead_nn_mp.py`)

Start from a copy of the existing mp script. Apply these edits to
`run_observation_mp` so it mirrors the modified `base.run_observation`. After
these edits the file is **identical across all 4 studies**.

1. Build the model via the factory: `model = base.build_model(n_features)`.

2. After the split, fit/transform labels and build loaders accordingly:

```python
    qt = base.fit_label_transform(y_tr)
    y_tr_t = base.transform_labels(qt, y_tr)
    y_va_t = base.transform_labels(qt, y_va)

    train_loader = base.make_loader(x_tr, y_tr_t, shuffle=True)
    val_loader = base.make_loader(x_va, y_va_t, shuffle=False)
    test_loader = base.make_loader(x_te, y_te, shuffle=False)   # ORIGINAL
```

3. NO_TRAIN branch: replace `base.nn.HuberLoss(delta=1.0)` with
   `base.make_criterion()`.

4. After predicting, inverse-transform:

```python
    y_true, y_pred = base.predict(model, test_loader)
    y_pred = base.inverse_transform_labels(qt, y_pred)   # -> original
```

5. Set `model_arch["model_type"] = base.MODEL_TYPE` (replace the hardcoded
   string) and add `"label_transform": base.LABEL_TRANSFORM_NAME` to
   `telemetry`. Keep the `"trainer": "torch_multiprocessing"` note.

The `del (...)` cleanup line references `y_tr`, `y_va` etc. — those names still
exist; leave the cleanup as-is (you may also `del y_tr_t, y_va_t` but it is
optional).

---

## Part F — Uploader, web SPA, workflow, requirements, pod terminate

### F1. `upload_web_app.py`

Copy from the existing study and change only `OBSERVATION_SET_NAME` to the new
study's folder name. `WEB_PREFIX = f"{OBSERVATION_SET_NAME}/app"` then derives
automatically.

### F2. `web/` folder

Copy the entire `web/` folder verbatim. Then, in each new study's copy:

- `web/js/config.js`: set `const DEFAULT_STUDY = "<study_folder_name>";`
  (this fallback only matters for local file:// testing — the SPA self-locates
  from its served URL — but keep it correct per study).
- `web/index.html`: update the `#setNameLabel` subtitle text to a readable
  study name, e.g. `Lookback / Lookahead LSTM Single-Head (Quantile)` /
  `... LSTM Single-Head (Quantile, MSE)` / `... GRU Single-Head (Quantile)` /
  `... GRU Single-Head (Quantile, MSE)`.
- `web/README.md`: update the `OBSERVATION_SET_NAME` references and the example
  URL to the new study folder name. (Cosmetic but keep consistent.)

All other web files are copied unchanged.

### F3. `do_workflow.sh`

Copy from the existing study and change:

- `STUDY_SUBDIR="scripts/studies/<new_study_folder_name>"`.
- Add a data-download step **before** the trainer step (step 3.5), calling the
  base folder's download script so the shared `~/data` cache is populated once:

```bash
# 3.5 Populate the shared local fl_data cache (~/data) ONCE. All studies read
#     from ~/data instead of GCS. The download script lives in the base study
#     folder; it skips assets already cached.
echo "Downloading fl_data to ~/data (one-time) ..."
python3 "scripts/studies/lookback_lookahead_nn/download_fl_data.py"
```

- Keep the SAME active/commented trainer structure as the original: Option B
  (the `_parallel.py` CUDA-streams trainer) ACTIVE, Options A and C commented,
  all referencing `${STUDY_SUBDIR}`.
- The `upload_web_app.py`, sleep, and `pod_self_terminate.py` steps reference
  `${STUDY_SUBDIR}` (the new folder).

### F4. `requirements.txt`

Copy from the existing study and add `scikit-learn` (needed for
`QuantileTransformer`). Final contents:

```
numpy
torch
plotly
scikit-learn
google-cloud-storage
google-auth
requests
```

### F5. `pod_self_terminate.py`

Copy verbatim (no changes).

---

## Part G — Validation

After creating all files:

1. `python3 -m py_compile` every new `.py` file (the base download script and
   all four studies' `lookback_lookahead_nn.py`, `..._parallel.py`, `..._mp.py`,
   `upload_web_app.py`, `pod_self_terminate.py`) plus the modified existing base
   study's `lookback_lookahead_nn.py`. Fix any syntax errors. (`py_compile` does
   not import torch/sklearn, so it works without those installed.)
2. `bash -n do_workflow.sh` for each new study to syntax-check the shell script.
3. Do NOT run any training — there is no GPU here and the runs are heavy.
4. Confirm each new study folder contains the full file set listed at the top.

---

## Summary of what differs between the 4 studies

Everything is identical across the 4 studies EXCEPT the base module
(`lookback_lookahead_nn.py`) and the cosmetic web/uploader/workflow strings:

- **build_model**: `LSTMRegressor` vs `GRURegressor` (+ the class definition).
- **make_criterion**: `nn.HuberLoss(delta=1.0)` vs `nn.MSELoss()`.
- **OBSERVATION_SET_NAME / MODEL_TYPE / LOSS_NAME**: per the table in C1.
- **upload_web_app.py / config.js / index.html / README.md / do_workflow.sh**:
  the study folder name / readable label.

The `_parallel.py` and `_mp.py` files end up byte-identical across all 4
studies because every per-study choice is funneled through `base` constants and
factories.
