#!/usr/bin/env python3
"""
Lookback / Lookahead single-target LSTM regression study.

This script trains a lean 2-layer LSTM (per the base model defined in
ai_chats/nn_model_for_lookback_lookahead.md) for every combination of:

    7 assets  x  4 label varieties (l_e_vwap[0..3])  x  2 input varieties
        => 56 observations total.

Input varieties
    - "vwap"        : 1 feature  -> input shape (24, 1)  (f_e_vwap only)
    - "vwap_imb"    : 2 features -> input shape (24, 2)  (f_e_vwap + f_e_n_imbalances)

Label varieties (single target, bounded in [-1, 1])
    - l_e_vwap[0]  (1h horizon)
    - l_e_vwap[1]  (2h horizon)
    - l_e_vwap[2]  (3h horizon)
    - l_e_vwap[3]  (4h horizon)

For every observation the script:
    1. Loads the fl_data_{asset} numpy array from GCS (bucket payamdprojectbucket).
    2. Builds a chronological 70/15/15 train/val/test split with a purge gap.
    3. Trains the modified LSTM (single linear output unit + tanh).
    4. Computes the full Eval_/Report_ metrics suite from the evaluation blueprint.
    5. Packages everything into the master JSON schema (Plotly-only visualizations).
    6. Uploads the report (and the trained model params) to the study's
       reports/ subfolder:
         gs://payamdpycryptoreports/{Observation_Set_Name}/reports/
             {Observation_Set_Name}_{Observation_Name}.json
             {Observation_Set_Name}_{Observation_Name}.pt

Data spec      : agents/datasets/lookback_lookahead_fl.md
Base model     : ai_chats/nn_model_for_lookback_lookahead.md
Eval blueprint : agents/ideas/evaluation_metrics_pipeline_blueprint.md
Report spec    : agents/ideas/headless_observation_reporting.md
"""

import io
import os
import sys
import gc
import json
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go

from sklearn.preprocessing import QuantileTransformer

# --------------------------------------------------------------------------- #
#  Repository / package import bootstrap
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_GCS_TOOLS_DIR = os.path.join(
    _REPO_ROOT, "packages", "tools", "google_cloud_storage_tools"
)
if _GCS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _GCS_TOOLS_DIR)

from gcs_tools import gcs_json_key_file, read_file, write_file  # noqa: E402

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
DATA_BUCKET = "payamdprojectbucket"
REPORT_BUCKET = "payamdpycryptoreports"
LOCAL_DATA_DIR = os.path.expanduser("~/data")
OBSERVATION_SET_NAME = "lookback_lookahead_lstm_singlehead_quantile_mse"
MODEL_TYPE = "LSTM_Base_2_64_SingleHead_Quantile_MSE"
LOSS_NAME = "MSE Loss"
LABEL_TRANSFORM_NAME = "quantile_uniform_scaled[-1,1]"
# Reports and model params live under this subfolder of the study folder:
#   gs://{REPORT_BUCKET}/{OBSERVATION_SET_NAME}/reports/
REPORTS_SUBDIR = "reports"
REPORTS_PREFIX = f"{OBSERVATION_SET_NAME}/{REPORTS_SUBDIR}"

ASSETS = [
    "btcusdt",
    "ethusdt",
    "trumpusdt",
    "vineusdt",
    "adausdt",
    "xrpusdt",
    "dogeusdt",
]

# fl_data column layout (agents/datasets/lookback_lookahead_fl.md)
TS_I = 0
F_E_VWAP_SI = 1            # inclusive
F_E_VWAP_EI = 24           # inclusive
F_E_N_IMBALANCES_SI = 25   # inclusive
F_E_N_IMBALANCES_EI = 48   # inclusive
L_E_VWAP_SI = 52           # inclusive
L_E_VWAP_EI = 55           # inclusive

SEQ_LEN = 24               # timesteps
N_LABEL_VARIETIES = 4      # l_e_vwap[0..3]

# Input varieties: name -> list of (start, end) inclusive column ranges
INPUT_VARIETIES = {
    "vwap": [(F_E_VWAP_SI, F_E_VWAP_EI)],
    "vwap_imb": [(F_E_VWAP_SI, F_E_VWAP_EI),
                 (F_E_N_IMBALANCES_SI, F_E_N_IMBALANCES_EI)],
}
INPUT_FEATURE_NAMES = {
    "vwap": ["f_e_vwap"],
    "vwap_imb": ["f_e_vwap", "f_e_n_imbalances"],
}

# Split / purge configuration
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test
PURGE_GAP = 1440 * 7       # at least 1 week of minutes between splits

# Training hyperparameters (from the base model spec)
HIDDEN_UNITS = 64
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 2048
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# When True the training fit loop is skipped (CPU smoke-test / dry run). The
# untrained model's forward pass is still used to produce predictions so that
# the entire evaluation + reporting + GCS plumbing is exercised end-to-end.
NO_TRAIN = False

# Bucketing for the [-1, 1] label space (21 bins, step 0.1)
BIN_EDGES = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 1)   # 21 values
N_BINS = len(BIN_EDGES)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    """Lean 2-layer LSTM with a single tanh-bounded regression output.

    Mirrors the base architecture (64 units, two LSTM layers, dropout 0.3,
    tanh output) but the output head emits a single unit instead of 8.
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN_UNITS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        seq, _ = self.lstm1(x)          # return_sequences=True
        seq = self.drop1(seq)
        seq, (h_n, _) = self.lstm2(seq)  # final hidden state of layer 2
        last = self.drop2(h_n[-1])      # (batch, hidden)
        out = torch.tanh(self.head(last))  # (batch, 1) bounded in (-1, 1)
        return out.squeeze(-1)          # (batch,)


def build_model(n_features: int):
    """Construct the study's RNN regressor."""
    return LSTMRegressor(n_features=n_features)


def make_criterion():
    """The study's training loss."""
    return nn.MSELoss()


# --------------------------------------------------------------------------- #
#  Data preparation
# --------------------------------------------------------------------------- #
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




def build_features(fl: np.ndarray, input_variety: str) -> np.ndarray:
    """Return feature tensor of shape (n_obs, SEQ_LEN, n_features).

    Columns are sliced per the requested input variety and stacked along the
    feature axis. fl[:, 1:25] is f_e_vwap (oldest..newest), fl[:, 25:49] is
    f_e_n_imbalances — both already chronological and normalized to [-1, 1].
    """
    cols = []
    for si, ei in INPUT_VARIETIES[input_variety]:
        block = fl[:, si:ei + 1]              # (n_obs, 24)
        cols.append(block[:, :, None])        # (n_obs, 24, 1)
    return np.concatenate(cols, axis=2)       # (n_obs, 24, n_features)


def build_labels(fl: np.ndarray, label_idx: int) -> np.ndarray:
    """Return a single l_e_vwap label column (variant 0..3), shape (n_obs,)."""
    return fl[:, L_E_VWAP_SI + label_idx]


def chronological_split(n: int):
    """Indices for chronological train/val/test split with purge gaps.

    The fl_data rows are already ordered oldest->newest, so a positional split
    is chronological. A PURGE_GAP block is dropped at each split boundary to
    avoid lookahead leakage from overlapping look-back/look-ahead windows.
    """
    train_end = int(n * TRAIN_FRAC)
    val_start = train_end + PURGE_GAP
    val_end = val_start + int(n * VAL_FRAC)
    test_start = val_end + PURGE_GAP

    train_idx = (0, train_end)
    val_idx = (val_start, val_end)
    test_idx = (test_start, n)
    return train_idx, val_idx, test_idx


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader):
    """Train with AdamW + Huber loss + ReduceLROnPlateau + early stopping.

    Returns (history, epochs_completed, train_seconds).
    """
    model.to(DEVICE)
    criterion = make_criterion()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHED_FACTOR, patience=LR_SCHED_PATIENCE
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_completed = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- train ----
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / max(n_seen, 1)

        # ---- validate ----
        val_loss = _eval_loss(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        epochs_completed = epoch

        elapsed = time.time() - t0
        print(
            f"    epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_huber {train_loss:.6f} | val_huber {val_loss:.6f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"batches {len(train_loader)} | {elapsed:6.1f}s",
            flush=True,
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"    early stopping at epoch {epoch}", flush=True)
                break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, epochs_completed, train_seconds


def _eval_loss(model, loader, criterion) -> float:
    model.eval()
    running = 0.0
    n_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            running += criterion(pred, yb).item() * xb.size(0)
            n_seen += xb.size(0)
    return running / max(n_seen, 1)


def predict(model, loader):
    """Return (y_true, y_pred) numpy arrays over the loader."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = model(xb).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


# --------------------------------------------------------------------------- #
#  Evaluation helpers (Eval_ / Report_ blueprint modules)
# --------------------------------------------------------------------------- #
def _bucketize(values: np.ndarray) -> np.ndarray:
    """Round to nearest 0.1 then map to bin index 0..20 (-1.0 .. 1.0)."""
    clipped = np.clip(values, -1.0, 1.0)
    rounded = np.round(clipped * 10.0).astype(int) + 10   # 0..20
    return np.clip(rounded, 0, N_BINS - 1)


def eval_tabular_error_metrics(y_true, y_pred) -> dict:
    """MSE, MAE and Huber loss on the test set."""
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    delta = 1.0
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    huber = float(np.mean(0.5 * quad ** 2 + delta * lin))
    return {"mse": mse, "mae": mae, "huber": huber}


def eval_directional_accuracy(y_true, y_pred) -> float:
    """Percentage of samples where sign(pred) == sign(true)."""
    same = np.sign(y_pred) == np.sign(y_true)
    return float(np.mean(same) * 100.0)


def eval_error_distribution_curve(y_true, y_pred):
    """Per-bin MAE across the 21 true-label bins. Returns (centers, maes)."""
    tb = _bucketize(y_true)
    abs_err = np.abs(y_pred - y_true)
    maes = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = tb == b
        if mask.any():
            maes[b] = float(np.mean(abs_err[mask]))
    return BIN_EDGES, maes


def eval_bucketed_confusion(y_true, y_pred):
    """21x21 confusion counts. rows = predicted bin, cols = true bin."""
    tb = _bucketize(y_true)
    pb = _bucketize(y_pred)
    matrix = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    for p, t in zip(pb, tb):
        matrix[p, t] += 1
    return matrix


def bucketed_confusion_rates(matrix: np.ndarray) -> np.ndarray:
    """Row-normalized (per predicted bin) percentage matrix."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    return matrix / safe * 100.0


def eval_rolling_temporal_error(y_true, y_pred, ts, window=1440):
    """Rolling MAE over a chronological window. Returns (times, rolling_mae)."""
    abs_err = np.abs(y_pred - y_true)
    n = len(abs_err)
    centers, vals = [], []
    for start in range(0, n, window):
        end = min(start + window, n)
        if end - start == 0:
            continue
        centers.append(int(ts[start]))
        vals.append(float(np.mean(abs_err[start:end])))
    return np.array(centers), np.array(vals)


# --------------------------------------------------------------------------- #
#  Plotly figure builders (all visualizations must be Plotly)
# --------------------------------------------------------------------------- #
def fig_convergence(history) -> go.Figure:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             mode="lines+markers", name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             mode="lines+markers", name="Validation Loss"))
    fig.update_layout(title="Eval_Convergence_Plot",
                      xaxis_title="Epoch", yaxis_title=LOSS_NAME)
    return fig


def fig_certainty_distribution(y_pred) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_pred, xbins=dict(start=-1.0, end=1.0, size=0.05),
                               name="Predictions"))
    fig.update_layout(title="Eval_Certainty_Distribution",
                      xaxis_title="Predicted value [-1, 1]",
                      yaxis_title="Count",
                      xaxis=dict(range=[-1.0, 1.0]))
    return fig


def fig_error_distribution_curve(centers, maes) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=maes, name="MAE per true bin"))
    fig.update_layout(title="Eval_Error_Distribution_Curve",
                      xaxis_title="True label bin",
                      yaxis_title="Mean Absolute Error")
    return fig


def fig_prediction_heatmap(matrix) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"{e:.1f}" for e in BIN_EDGES],
        y=[f"{e:.1f}" for e in BIN_EDGES],
        colorscale="Viridis",
    ))
    fig.update_layout(title="Eval_Prediction_Heatmap",
                      xaxis_title="True Label Bin",
                      yaxis_title="Predicted Label Bin")
    return fig


def fig_rolling_temporal_error(times, vals) -> go.Figure:
    if len(times):
        x = [datetime.datetime.utcfromtimestamp(t / 1000.0) for t in times]
    else:
        x = []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=vals, mode="lines", name="Rolling MAE"))
    fig.update_layout(title="Eval_Rolling_Temporal_Error",
                      xaxis_title="Time (UTC)",
                      yaxis_title="Rolling MAE (1-day window)")
    return fig


def fig_label_histogram_before_transform(y_train: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_train,
        xbins=dict(start=-1.0, end=1.0, size=0.1),
        name="Before Transform",
    ))
    fig.update_layout(
        title="Label Histogram — Before Quantile Transform (Train Set)",
        xaxis_title="Label value [-1, 1]",
        yaxis_title="Count",
        xaxis=dict(range=[-1.1, 1.1]),
    )
    return fig


def fig_label_histogram_after_transform(y_train_t: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_train_t,
        xbins=dict(start=-1.0, end=1.0, size=0.1),
        name="After Transform",
    ))
    fig.update_layout(
        title="Label Histogram — After Quantile Transform (Train Set)",
        xaxis_title="Transformed label value [-1, 1]",
        yaxis_title="Count",
        xaxis=dict(range=[-1.1, 1.1]),
    )
    return fig


# --------------------------------------------------------------------------- #
#  Report packaging & GCS export (headless_observation_reporting.md)
# --------------------------------------------------------------------------- #
def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def generate_and_upload_report(observation_name, metadata, model_arch,
                               telemetry, metrics, plotly_figs):
    """Build the master JSON payload and upload to the report bucket."""
    master_report = {
        "metadata": {
            "observation_set_name": OBSERVATION_SET_NAME,
            "observation_name": observation_name,
            "execution_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            **metadata,
        },
        "model_architecture": model_arch,
        "training_telemetry": telemetry,
        "evaluation_metrics": metrics,
        "visualizations": {},
    }
    for fig_name, fig_obj in plotly_figs.items():
        master_report["visualizations"][fig_name] = fig_obj.to_json()

    json_payload = json.dumps(master_report, indent=2)
    file_path = (f"{REPORTS_PREFIX}/"
                 f"{OBSERVATION_SET_NAME}_{observation_name}.json")
    buffer = io.BytesIO(json_payload.encode("utf-8"))
    write_file(REPORT_BUCKET, file_path, buffer, content_type="application/json")
    print(f"  uploaded report -> gs://{REPORT_BUCKET}/{file_path}", flush=True)


def upload_model_params(observation_name, model):
    """Serialize the model state_dict and upload it next to the JSON report.

    The params land in the same bucket folder as the observation's report
    (the study's reports/ subfolder):
        gs://{REPORT_BUCKET}/{OBSERVATION_SET_NAME}/reports/
            {OBSERVATION_SET_NAME}_{observation_name}.pt
    """
    file_path = (f"{REPORTS_PREFIX}/"
                 f"{OBSERVATION_SET_NAME}_{observation_name}.pt")
    buffer = io.BytesIO()
    # move to CPU before serializing so the checkpoint is portable
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, buffer)
    buffer.seek(0)
    write_file(REPORT_BUCKET, file_path, buffer,
               content_type="application/octet-stream")
    print(f"  uploaded params -> gs://{REPORT_BUCKET}/{file_path}", flush=True)


# --------------------------------------------------------------------------- #
#  Single observation pipeline
# --------------------------------------------------------------------------- #
def run_observation(asset, label_idx, input_variety, fl):
    observation_name = f"{asset}_l_e_vwap{label_idx}_{input_variety}"
    print(f"\n=== Observation: {observation_name} (device={DEVICE}) ===",
          flush=True)

    x_all = build_features(fl, input_variety)        # (n, 24, n_feat)
    y_all = build_labels(fl, label_idx)              # (n,)
    ts_all = fl[:, TS_I]                             # (n,)
    n = x_all.shape[0]
    n_features = x_all.shape[2]

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = chronological_split(n)

    x_tr, y_tr = x_all[tr_s:tr_e], y_all[tr_s:tr_e]
    x_va, y_va = x_all[va_s:va_e], y_all[va_s:va_e]
    x_te, y_te = x_all[te_s:te_e], y_all[te_s:te_e]
    ts_te = ts_all[te_s:te_e]

    qt = fit_label_transform(y_tr)
    y_tr_t = transform_labels(qt, y_tr)
    y_va_t = transform_labels(qt, y_va)

    train_loader = make_loader(x_tr, y_tr_t, shuffle=True)
    val_loader = make_loader(x_va, y_va_t, shuffle=False)
    test_loader = make_loader(x_te, y_te, shuffle=False)   # ORIGINAL labels

    model = build_model(n_features)

    if NO_TRAIN:
        # Dry run: skip the fit loop entirely. The untrained model's forward
        # pass still produces predictions so the evaluation/reporting pipeline
        # is fully exercised. A single-point "history" keeps the convergence
        # plot well-formed.
        print("    [NO-TRAIN] skipping fit loop (dry run)", flush=True)
        model.to(DEVICE)
        train_criterion = make_criterion()
        train_loss = _eval_loss(model, train_loader, train_criterion)
        val_loss = _eval_loss(model, val_loader, train_criterion)
        history = {"train_loss": [train_loss], "val_loss": [val_loss]}
        epochs_done = 0
        train_secs = 0.0
    else:
        history, epochs_done, train_secs = train_model(
            model, train_loader, val_loader)

    # ---- predictions ----
    y_true, y_pred = predict(model, test_loader)
    y_pred = inverse_transform_labels(qt, y_pred)   # transformed -> original

    # ---- Eval_ modules ----
    tabular = eval_tabular_error_metrics(y_true, y_pred)
    dir_acc = eval_directional_accuracy(y_true, y_pred)
    edc_centers, edc_maes = eval_error_distribution_curve(y_true, y_pred)
    conf_matrix = eval_bucketed_confusion(y_true, y_pred)
    conf_rates = bucketed_confusion_rates(conf_matrix)
    roll_times, roll_vals = eval_rolling_temporal_error(y_true, y_pred, ts_te)

    # ---- Eval_Tabular_Error_Metrics terminal print ----
    print(f"  TEST  MSE={tabular['mse']:.6f}  MAE={tabular['mae']:.6f}  "
          f"Huber={tabular['huber']:.6f}  DirAcc={dir_acc:.2f}%", flush=True)

    # ---- Plotly visualizations ----
    plotly_figs = {
        "Eval_Convergence_Plot": fig_convergence(history),
        "Eval_Certainty_Distribution": fig_certainty_distribution(y_pred),
        "Eval_Error_Distribution_Curve": fig_error_distribution_curve(
            edc_centers, edc_maes),
        "Eval_Prediction_Heatmap": fig_prediction_heatmap(conf_matrix),
        "Eval_Rolling_Temporal_Error": fig_rolling_temporal_error(
            roll_times, roll_vals),
        "Label_Histogram_Before_Transform": fig_label_histogram_before_transform(y_tr),
        "Label_Histogram_After_Transform": fig_label_histogram_after_transform(y_tr_t),
    }

    # ---- metadata / telemetry / metrics payloads ----
    start_ms = int(ts_all[0])
    end_ms = int(ts_all[-1])
    metadata = {
        "asset_identifier": asset.upper(),
        "data_date_range": {
            "start": datetime.datetime.utcfromtimestamp(
                start_ms / 1000.0).strftime("%Y-%m-%d %H:%M"),
            "end": datetime.datetime.utcfromtimestamp(
                end_ms / 1000.0).strftime("%Y-%m-%d %H:%M"),
        },
    }
    model_arch = {
        "model_type": MODEL_TYPE,
        "input_features": INPUT_FEATURE_NAMES[input_variety],
        "sequence_length": SEQ_LEN,
        "target_labels": [f"l_e_vwap[{label_idx}]"],
    }
    telemetry = {
        "total_parameters": count_parameters(model),
        "epochs_completed": epochs_done,
        "batch_size": BATCH_SIZE,
        "hardware_utilized": (f"CUDA: {torch.cuda.get_device_name(0)}"
                              if torch.cuda.is_available() else "CPU"),
        "total_train_time_seconds": round(train_secs, 2),
        "rows_train": int(x_tr.shape[0]),
        "rows_val": int(x_va.shape[0]),
        "rows_test": int(x_te.shape[0]),
        "feature_count": n_features,
        "no_train_dry_run": bool(NO_TRAIN),
        "label_transform": LABEL_TRANSFORM_NAME,
    }
    if NO_TRAIN:
        # Mark the dry run prominently in metadata so the SPA / consumer never
        # mistakes an untrained observation for a real training result.
        metadata["dry_run"] = True
        metadata["dry_run_note"] = (
            "NO-TRAIN dry run: predictions come from an UNTRAINED model. "
            "Metrics and visualizations are plumbing smoke-tests only.")
    metrics = {
        "global_metrics": {
            "test_huber_loss": tabular["huber"],
            "test_mse": tabular["mse"],
        },
        "per_head_metrics": {
            f"l_e_vwap[{label_idx}]": {
                "mae": tabular["mae"],
                "mse": tabular["mse"],
                "directional_accuracy_pct": dir_acc,
            }
        },
        "bucketed_confusion_matrix": conf_matrix.tolist(),
        "bucketed_confusion_rates": np.round(conf_rates, 4).tolist(),
        "error_distribution_curve": {
            "bins": [float(b) for b in edc_centers],
            "mae": [None if np.isnan(v) else float(v) for v in edc_maes],
        },
    }

    generate_and_upload_report(observation_name, metadata, model_arch,
                               telemetry, metrics, plotly_figs)

    # ---- persist trained model parameters next to the report ----
    if NO_TRAIN:
        print("    [NO-TRAIN] skipping model-param upload (untrained weights)",
              flush=True)
    else:
        upload_model_params(observation_name, model)

    # ---- release per-observation memory before the next iteration ----
    del (model, x_all, y_all, x_tr, y_tr, x_va, y_va, x_te, y_te,
         train_loader, val_loader, test_loader, y_true, y_pred, plotly_figs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Main loop: 7 assets x 4 labels x 2 input varieties = 56 observations
# --------------------------------------------------------------------------- #
def main():
    global NO_TRAIN

    parser = argparse.ArgumentParser(
        description="Lookback/Lookahead single-target LSTM sweep.")
    # Accept both the exact '-notrain' spelling and the conventional
    # '--notrain'. When set, the fit loop is skipped (CPU smoke-test).
    parser.add_argument(
        "-notrain", "--notrain", dest="notrain", action="store_true",
        help=("Skip the training fit loop. Runs the whole pipeline (data load, "
              "model build, evaluation, reporting, GCS upload) using the "
              "untrained model's forward pass for a CPU dry run."))
    args = parser.parse_args()
    NO_TRAIN = bool(args.notrain)
    if NO_TRAIN:
        print("########## NO-TRAIN DRY RUN ENABLED ##########", flush=True)
        print("  Training is skipped. Predictions use an UNTRAINED model.",
              flush=True)

    gcs_json_key_file()  # resolve credentials once, before any GCS call

    total = len(ASSETS) * N_LABEL_VARIETIES * len(INPUT_VARIETIES)
    done = 0
    failures = []

    # Outer loop over assets: each asset's raw fl_data blob is downloaded /
    # loaded from GCS exactly ONCE per execution and reused across all 8 of
    # that asset's observations (4 labels x 2 input varieties). Memory is
    # released before moving on to the next asset.
    for asset in ASSETS:
        fl = None
        try:
            print(f"\n########## Loading fl_data_{asset} ##########", flush=True)
            fl = load_fl_data(asset)
            print(f"  loaded shape {fl.shape}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load {asset}: {exc}", flush=True)
            failures.append((asset, "load", str(exc)))
            continue

        for label_idx in range(N_LABEL_VARIETIES):
            for input_variety in INPUT_VARIETIES:
                done += 1
                tag = f"{asset}/l_e_vwap{label_idx}/{input_variety}"
                print(f"\n[{done}/{total}] {tag}", flush=True)
                try:
                    run_observation(asset, label_idx, input_variety, fl)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED observation {tag}: {exc}", flush=True)
                    failures.append((asset, tag, str(exc)))

        # release this asset's raw data before loading the next asset
        del fl
        gc.collect()

    print("\n========== SWEEP COMPLETE ==========", flush=True)
    print(f"  observations attempted: {total}", flush=True)
    print(f"  failures: {len(failures)}", flush=True)
    for f in failures:
        print(f"    - {f}", flush=True)


if __name__ == "__main__":
    main()
