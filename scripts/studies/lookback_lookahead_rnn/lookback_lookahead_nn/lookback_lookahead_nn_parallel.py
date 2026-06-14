#!/usr/bin/env python3
"""
Lookback / Lookahead single-target LSTM study — PARALLEL, GPU-resident trainer.

This is a performance-optimized sibling of ``lookback_lookahead_nn.py``. It
trains the SAME 56 observations (7 assets x 4 label varieties x 2 input
varieties) and produces IDENTICAL reports (same eval modules, same Plotly
figures, same master JSON schema, same GCS report/ location). The ONLY
differences are *how* the work is scheduled on the hardware — never *what*
math each model performs:

    * Phase 1 — Load ALL data once. Every asset's fl_data blob is pulled from
      GCS exactly one time into host RAM (timed, per-asset + total).
    * Phase 2 — Move data to GPU once. Train/val/test feature & label tensors
      for every observation are built and transferred host->device a single
      time, then kept resident and shared across all 56 models (timed).
    * Phase 3 — Train all 56 models concurrently. Training uses the
      GPU-resident tensors with index-sliced mini-batches (no DataLoader, no
      TensorDataset, no per-batch host->device copy). Models are run in waves
      of ``--concurrency N``; within a wave each model gets its own
      ``torch.cuda.Stream`` so the GPU overlaps their (individually tiny,
      launch-bound) kernels and idle capacity is filled.

Why the results are unchanged (vs. the sequential script):
    * Per-model PROCEDURE is byte-for-byte the same: same model class, same
      AdamW + Huber + ReduceLROnPlateau + early-stopping + MAX_EPOCHS, same
      BATCH_SIZE, a full per-epoch random permutation reshuffle (matching
      ``DataLoader(shuffle=True)``), and identical best-weight tracking.
    * Each model owns its own optimizer / scheduler / early-stop / best-state.
      A model's outcome does NOT depend on which wave-mates it shares — CUDA
      streams only let independent kernels overlap in time; they do not mix
      gradients or state across models.
    * The evaluation, report building and GCS upload all REUSE the functions
      from ``lookback_lookahead_nn`` so every report is identical in format and
      lands in the same ``.../reports/`` location.

The companion performance rationale lives in
``agents/general/nn_training_performance.md``.

Eval blueprint : agents/ideas/evaluation_metrics_pipeline_blueprint.md
Report spec    : agents/ideas/headless_observation_reporting.md
Data spec      : agents/datasets/lookback_lookahead_fl.md
"""

import os
import sys
import gc
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn

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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gcs_tools import gcs_json_key_file  # noqa: E402

# Import the existing study module wholesale so that the model class,
# constants, evaluation modules, Plotly builders, and report/upload helpers are
# REUSED verbatim — guaranteeing identical eval/report behavior.
import lookback_lookahead_nn as base  # noqa: E402


# --------------------------------------------------------------------------- #
#  Phase 1 — Load ALL data once (timed)
# --------------------------------------------------------------------------- #
def load_all_data():
    """Download/load every asset's fl_data from GCS exactly once.

    Returns (fl_by_asset, per_asset_seconds, total_seconds). Assets that fail
    to load are simply omitted from the dict (a warning is printed).
    """
    print("\n########## PHASE 1: load ALL fl_data (once) ##########", flush=True)
    fl_by_asset = {}
    per_asset_seconds = {}
    t_total0 = time.time()
    for asset in base.ASSETS:
        t0 = time.time()
        try:
            fl = base.load_fl_data(asset)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load {asset}: {exc}", flush=True)
            continue
        dt = time.time() - t0
        per_asset_seconds[asset] = dt
        fl_by_asset[asset] = fl
        print(f"  loaded fl_data_{asset} shape {fl.shape} in {dt:6.2f}s",
              flush=True)
    total = time.time() - t_total0
    print(f"  >> total data-load time: {total:6.2f}s "
          f"({len(fl_by_asset)} assets)", flush=True)
    return fl_by_asset, per_asset_seconds, total


# --------------------------------------------------------------------------- #
#  Phase 2 — Build observation specs + move tensors to GPU once (timed)
# --------------------------------------------------------------------------- #
class Observation:
    """Holds one model's GPU-resident tensors and per-model training state.

    Tensors (x_tr/y_tr/x_va/y_va/x_te/y_te) are uploaded to the device ONCE in
    phase 2 and reused for every epoch/batch. ts_te stays on host (only used by
    the rolling-temporal-error eval, which runs on numpy).
    """

    def __init__(self, asset, label_idx, input_variety, fl, device):
        self.asset = asset
        self.label_idx = label_idx
        self.input_variety = input_variety
        self.name = f"{asset}_l_e_vwap{label_idx}_{input_variety}"
        self.device = device

        x_all = base.build_features(fl, input_variety)   # (n, 24, n_feat)
        y_all = base.build_labels(fl, label_idx)         # (n,)
        ts_all = fl[:, base.TS_I]                        # (n,)
        n = x_all.shape[0]
        self.n_features = x_all.shape[2]

        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = base.chronological_split(n)

        # Build float32 tensors and move to device ONCE.
        def to_dev(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

        self.x_tr = to_dev(x_all[tr_s:tr_e])
        self.y_tr = to_dev(y_all[tr_s:tr_e])
        self.x_va = to_dev(x_all[va_s:va_e])
        self.y_va = to_dev(y_all[va_s:va_e])
        self.x_te = to_dev(x_all[te_s:te_e])
        self.y_te = to_dev(y_all[te_s:te_e])

        # Host-side metadata used by reporting / eval.
        self.ts_te = ts_all[te_s:te_e].copy()
        self.rows_train = int(self.x_tr.shape[0])
        self.rows_val = int(self.x_va.shape[0])
        self.rows_test = int(self.x_te.shape[0])
        self.data_start_ms = int(ts_all[0])
        self.data_end_ms = int(ts_all[-1])

        # Per-model objects (created in phase 3). Kept independent per model.
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = {"train_loss": [], "val_loss": []}
        self.best_val = float("inf")
        self.best_state = None
        self.epochs_no_improve = 0
        self.epochs_completed = 0
        self.done = False                # early-stopped or hit MAX_EPOCHS
        self.train_seconds = 0.0
        self._t0 = None

    def device_bytes(self):
        return sum(t.element_size() * t.nelement() for t in
                   (self.x_tr, self.y_tr, self.x_va, self.y_va,
                    self.x_te, self.y_te))

    def free_device_tensors(self):
        for attr in ("x_tr", "y_tr", "x_va", "y_va", "x_te", "y_te", "model"):
            setattr(self, attr, None)


def build_observations(fl_by_asset, device):
    """Phase 2: build every observation and move all tensors to device once."""
    print("\n########## PHASE 2: build tensors + host->device transfer "
          "(once) ##########", flush=True)
    observations = []
    t0 = time.time()
    for asset, fl in fl_by_asset.items():
        for label_idx in range(base.N_LABEL_VARIETIES):
            for input_variety in base.INPUT_VARIETIES:
                obs = Observation(asset, label_idx, input_variety, fl, device)
                observations.append(obs)
    # Ensure all async copies have landed before timing the transfer.
    if device.type == "cuda":
        torch.cuda.synchronize()
    transfer_seconds = time.time() - t0

    total_bytes = sum(o.device_bytes() for o in observations)
    print(f"  built {len(observations)} observations; "
          f"{total_bytes / (1024 ** 2):.1f} MiB resident on {device}",
          flush=True)
    print(f"  >> total host->device transfer time: {transfer_seconds:6.2f}s",
          flush=True)
    return observations, transfer_seconds


# --------------------------------------------------------------------------- #
#  Phase 3 — Concurrent training over GPU-resident tensors
# --------------------------------------------------------------------------- #
def _init_training_state(obs, device):
    """Create the per-model objects: identical setup to base.train_model."""
    model = base.LSTMRegressor(n_features=obs.n_features).to(device)
    obs.model = model
    obs.optimizer = torch.optim.AdamW(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY
    )
    obs.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        obs.optimizer, mode="min", factor=base.LR_SCHED_FACTOR,
        patience=base.LR_SCHED_PATIENCE
    )
    obs.history = {"train_loss": [], "val_loss": []}
    obs.best_val = float("inf")
    obs.best_state = None
    obs.epochs_no_improve = 0
    obs.epochs_completed = 0
    obs.done = False
    obs._t0 = time.time()


def _run_epoch(obs, criterion, device):
    """Run one training epoch + validation for a single model.

    Mirrors base.train_model's per-epoch procedure exactly but operates on the
    GPU-resident tensors with index-sliced batches and a full per-epoch random
    permutation (== DataLoader(shuffle=True, drop_last=False)). Running loss is
    accumulated in an ON-DEVICE tensor; .item() is called once per epoch
    (logging only, never affects gradients).
    """
    model = obs.model
    optimizer = obs.optimizer
    bs = base.BATCH_SIZE
    n = obs.x_tr.shape[0]

    # ---- train ----
    model.train()
    perm = torch.randperm(n, device=device)             # full reshuffle
    running = torch.zeros((), device=device)            # on-device accumulator
    for start in range(0, n, bs):
        idx = perm[start:start + bs]
        xb = obs.x_tr[idx]
        yb = obs.y_tr[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.detach() * xb.size(0)           # no sync per batch
    train_loss = (running / max(n, 1)).item()           # single sync / epoch

    # ---- validate ----
    val_loss = _eval_loss_resident(model, obs.x_va, obs.y_va, criterion)
    obs.scheduler.step(val_loss)

    obs.history["train_loss"].append(train_loss)
    obs.history["val_loss"].append(val_loss)
    obs.epochs_completed += 1

    if val_loss < obs.best_val - 1e-6:
        obs.best_val = val_loss
        obs.best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        obs.epochs_no_improve = 0
    else:
        obs.epochs_no_improve += 1

    # Early stopping / max-epochs termination.
    if obs.epochs_no_improve >= base.EARLY_STOP_PATIENCE:
        obs.done = True
    if obs.epochs_completed >= base.MAX_EPOCHS:
        obs.done = True


def _eval_loss_resident(model, x, y, criterion):
    """Validation loss over a GPU-resident tensor, batched, single sync."""
    model.eval()
    bs = base.BATCH_SIZE
    n = x.shape[0]
    running = torch.zeros((), device=x.device)
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = x[start:start + bs]
            yb = y[start:start + bs]
            running += criterion(model(xb), yb) * xb.size(0)
    return (running / max(n, 1)).item()


def _predict_resident(model, x):
    """Predictions over a GPU-resident feature tensor -> numpy on host."""
    model.eval()
    bs = base.BATCH_SIZE
    n = x.shape[0]
    out = []
    with torch.no_grad():
        for start in range(0, n, bs):
            out.append(model(x[start:start + bs]))
    return torch.cat(out).cpu().numpy()


def train_wave(wave, device, use_streams):
    """Train a wave of models concurrently using one CUDA stream per model.

    All models in the wave advance epoch-by-epoch in lockstep. Within each
    epoch, every model's kernels are launched on its OWN stream so the GPU can
    overlap the (tiny, launch-bound) work of independent models. Streams only
    overlap execution in time — each model's gradients/state are fully
    independent, so per-model results are identical to sequential training.

    On CPU (or when streams are disabled) this degrades cleanly to running each
    model's epoch one after another, which is still correct.
    """
    criterion = nn.HuberLoss(delta=1.0)
    for obs in wave:
        _init_training_state(obs, device)

    if base.NO_TRAIN:
        # Dry run: skip fitting. Compute a single-point history so the
        # convergence plot stays well-formed, matching the base script.
        for obs in wave:
            tl = _eval_loss_resident(obs.model, obs.x_tr, obs.y_tr, criterion)
            vl = _eval_loss_resident(obs.model, obs.x_va, obs.y_va, criterion)
            obs.history = {"train_loss": [tl], "val_loss": [vl]}
            obs.epochs_completed = 0
            obs.train_seconds = 0.0
        return

    streams = None
    if use_streams and device.type == "cuda":
        streams = [torch.cuda.Stream(device=device) for _ in wave]

    # Lockstep epoch loop: stop a model as soon as it early-stops / maxes out;
    # keep iterating until every model in the wave is done.
    epoch = 0
    while not all(o.done for o in wave):
        epoch += 1
        active = [(i, o) for i, o in enumerate(wave) if not o.done]
        if streams is not None:
            for i, obs in active:
                with torch.cuda.stream(streams[i]):
                    _run_epoch(obs, criterion, device)
            torch.cuda.synchronize()    # boundary: all wave-mates land here
        else:
            for _, obs in active:
                _run_epoch(obs, criterion, device)

        # Lightweight progress line for this wave/epoch.
        lines = ", ".join(
            f"{o.name.split('_', 1)[0]}…{o.label_idx}/{o.input_variety[:3]}"
            f":v{o.history['val_loss'][-1]:.4f}" for _, o in active[:4])
        more = "" if len(active) <= 4 else f" (+{len(active) - 4} more)"
        print(f"    wave epoch {epoch:02d} | active {len(active)} | "
              f"{lines}{more}", flush=True)

    # Restore best weights + record per-model wall time.
    for obs in wave:
        obs.train_seconds = time.time() - obs._t0
        if obs.best_state is not None:
            obs.model.load_state_dict(obs.best_state)


# --------------------------------------------------------------------------- #
#  Reporting for one trained observation (reuses base eval / report helpers)
# --------------------------------------------------------------------------- #
def report_observation(obs):
    """Run the full Eval_/Report_ suite and upload, reusing base functions."""
    observation_name = obs.name
    print(f"\n=== Reporting: {observation_name} ===", flush=True)

    y_true = obs.y_te.cpu().numpy()
    y_pred = _predict_resident(obs.model, obs.x_te)
    ts_te = obs.ts_te

    # ---- Eval_ modules (identical functions as the sequential script) ----
    tabular = base.eval_tabular_error_metrics(y_true, y_pred)
    dir_acc = base.eval_directional_accuracy(y_true, y_pred)
    edc_centers, edc_maes = base.eval_error_distribution_curve(y_true, y_pred)
    conf_matrix = base.eval_bucketed_confusion(y_true, y_pred)
    conf_rates = base.bucketed_confusion_rates(conf_matrix)
    roll_times, roll_vals = base.eval_rolling_temporal_error(
        y_true, y_pred, ts_te)

    print(f"  TEST  MSE={tabular['mse']:.6f}  MAE={tabular['mae']:.6f}  "
          f"Huber={tabular['huber']:.6f}  DirAcc={dir_acc:.2f}%", flush=True)

    plotly_figs = {
        "Eval_Convergence_Plot": base.fig_convergence(obs.history),
        "Eval_Certainty_Distribution": base.fig_certainty_distribution(y_pred),
        "Eval_Error_Distribution_Curve": base.fig_error_distribution_curve(
            edc_centers, edc_maes),
        "Eval_Prediction_Heatmap": base.fig_prediction_heatmap(conf_matrix),
        "Eval_Rolling_Temporal_Error": base.fig_rolling_temporal_error(
            roll_times, roll_vals),
    }

    metadata = {
        "asset_identifier": obs.asset.upper(),
        "data_date_range": {
            "start": datetime.datetime.utcfromtimestamp(
                obs.data_start_ms / 1000.0).strftime("%Y-%m-%d %H:%M"),
            "end": datetime.datetime.utcfromtimestamp(
                obs.data_end_ms / 1000.0).strftime("%Y-%m-%d %H:%M"),
        },
    }
    model_arch = {
        "model_type": "LSTM_Base_2_64_SingleHead",
        "input_features": base.INPUT_FEATURE_NAMES[obs.input_variety],
        "sequence_length": base.SEQ_LEN,
        "target_labels": [f"l_e_vwap[{obs.label_idx}]"],
    }
    telemetry = {
        "total_parameters": base.count_parameters(obs.model),
        "epochs_completed": obs.epochs_completed,
        "batch_size": base.BATCH_SIZE,
        "hardware_utilized": (f"CUDA: {torch.cuda.get_device_name(0)}"
                              if torch.cuda.is_available() else "CPU"),
        "total_train_time_seconds": round(obs.train_seconds, 2),
        "rows_train": obs.rows_train,
        "rows_val": obs.rows_val,
        "rows_test": obs.rows_test,
        "feature_count": obs.n_features,
        "no_train_dry_run": bool(base.NO_TRAIN),
        # Provenance: distinguishes reports from the parallel trainer. This is a
        # telemetry NOTE only — it does NOT change any metric definition.
        "trainer": "parallel_gpu_resident",
    }
    if base.NO_TRAIN:
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
            f"l_e_vwap[{obs.label_idx}]": {
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

    base.generate_and_upload_report(observation_name, metadata, model_arch,
                                    telemetry, metrics, plotly_figs)

    if base.NO_TRAIN:
        print("    [NO-TRAIN] skipping model-param upload (untrained weights)",
              flush=True)
    else:
        base.upload_model_params(observation_name, obs.model)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=("Lookback/Lookahead single-target LSTM sweep — parallel, "
                     "GPU-resident concurrent trainer."))
    parser.add_argument(
        "-notrain", "--notrain", dest="notrain", action="store_true",
        help=("Skip the training fit loop. Runs the whole pipeline (data load, "
              "model build, evaluation, reporting, GCS upload) using the "
              "untrained model's forward pass for a CPU dry run."))
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help=("Number of models trained concurrently per wave (one CUDA stream "
              "each). 1 == sequential. Default 8."))
    args = parser.parse_args()

    base.NO_TRAIN = bool(args.notrain)
    concurrency = max(1, int(args.concurrency))
    device = base.DEVICE

    # Fixed input shapes per model -> autotune cuDNN once for a free speedup.
    torch.backends.cudnn.benchmark = True

    if base.NO_TRAIN:
        print("########## NO-TRAIN DRY RUN ENABLED ##########", flush=True)
        print("  Training is skipped. Predictions use an UNTRAINED model.",
              flush=True)
    print(f"  device={device} | concurrency={concurrency} | "
          f"streams={'on' if device.type == 'cuda' else 'off (cpu)'}",
          flush=True)

    gcs_json_key_file()  # resolve credentials once, before any GCS call

    # ---- Phase 1: load all data once ----
    fl_by_asset, per_asset_load, load_seconds = load_all_data()
    if not fl_by_asset:
        print("No data loaded; aborting.", flush=True)
        return

    # ---- Phase 2: build + transfer to GPU once ----
    observations, transfer_seconds = build_observations(fl_by_asset, device)
    # The raw host blobs are no longer needed once tensors are on device.
    fl_by_asset.clear()
    gc.collect()

    # ---- Phase 3: concurrent training in waves ----
    print(f"\n########## PHASE 3: train {len(observations)} models "
          f"(waves of {concurrency}) ##########", flush=True)
    train_t0 = time.time()
    failures = []
    n_total = len(observations)
    for w_start in range(0, n_total, concurrency):
        wave = observations[w_start:w_start + concurrency]
        names = ", ".join(o.name for o in wave)
        print(f"\n--- wave {w_start // concurrency + 1} "
              f"[{w_start + 1}-{w_start + len(wave)}/{n_total}]: {names} ---",
              flush=True)
        try:
            train_wave(wave, device, use_streams=(concurrency > 1))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED training wave: {exc}", flush=True)
            for o in wave:
                failures.append((o.name, "train", str(exc)))
            continue

        # Report + upload each model in the wave, then free its tensors.
        for obs in wave:
            try:
                report_observation(obs)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED reporting {obs.name}: {exc}", flush=True)
                failures.append((obs.name, "report", str(exc)))
            finally:
                obs.free_device_tensors()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    train_seconds = time.time() - train_t0

    n_trained = max(n_total - sum(1 for f in failures if f[1] == "train"), 1)

    # ---- Timing report ----
    print("\n========== SWEEP COMPLETE (parallel GPU-resident) ==========",
          flush=True)
    print(f"  observations attempted : {n_total}", flush=True)
    print(f"  failures               : {len(failures)}", flush=True)
    for f in failures:
        print(f"    - {f}", flush=True)
    print("\n  ---- timing report ----", flush=True)
    print(f"  total data-load time      : {load_seconds:8.2f}s", flush=True)
    print(f"  total host->device transfer: {transfer_seconds:8.2f}s", flush=True)
    print(f"  total training time       : {train_seconds:8.2f}s", flush=True)
    print(f"  avg training time / model : "
          f"{train_seconds / n_trained:8.2f}s "
          f"(over {n_trained} trained models)", flush=True)


if __name__ == "__main__":
    main()
