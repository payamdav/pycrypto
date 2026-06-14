#!/usr/bin/env python3
"""
Lookback / Lookahead MLP study (MSE loss) — MULTIPROCESSING trainer.

This is a process-parallel sibling of ``lookback_lookahead_nn.py`` (the
sequential "base" trainer). It trains the SAME 14 observations
(7 assets x 2 label varieties) and produces IDENTICAL reports (same eval
modules, same Plotly figures, same master JSON schema, same GCS report/
location). The ONLY difference is *how* the work is scheduled — never *what*
math each model performs.

Concurrency model:
    * Phase 1 — Load ALL data once, IN THE PARENT PROCESS. Every asset's
      fl_data blob is pulled from GCS exactly one time into host RAM (timed,
      per-asset + total). It is NEVER re-downloaded inside a worker.
    * Phase 2 — Spawn a pool of ``--concurrency N`` worker processes. The
      preloaded host data dict is handed to each worker exactly once via the
      Pool ``initializer`` (pickled once per worker at startup).
    * Phase 3 — The 14 observation specs (tiny tuples: asset, label_idx) are
      mapped across the pool.

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
import torch.multiprocessing as mp

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
#  Phase 1 — Load ALL data once, in the PARENT process (timed)
# --------------------------------------------------------------------------- #
def load_all_data():
    """Download/load every asset's fl_data from GCS exactly once (parent only).

    Returns (fl_by_asset, per_asset_seconds, total_seconds).
    """
    print("\n########## PHASE 1: load ALL fl_data (once, in parent) ##########",
          flush=True)
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
#  Worker process: init (receives preloaded data once) + per-observation work
# --------------------------------------------------------------------------- #

_WORKER_DATA = None
_WORKER_NO_TRAIN = False


def _init_worker(fl_by_asset, no_train):
    """Pool initializer: runs ONCE per worker process at startup."""
    global _WORKER_DATA, _WORKER_NO_TRAIN
    _WORKER_DATA = fl_by_asset
    _WORKER_NO_TRAIN = bool(no_train)
    base.NO_TRAIN = bool(no_train)
    try:
        gcs_json_key_file()
    except Exception as exc:  # noqa: BLE001
        print(f"  [worker {os.getpid()}] WARNING: gcs_json_key_file failed: "
              f"{exc}", flush=True)


def _run_one(spec):
    """Train + evaluate + report ONE observation. Runs inside a worker."""
    asset, label_idx = spec
    name = f"{asset}_l_e_vwap{label_idx}"
    t0 = time.time()
    try:
        fl = _WORKER_DATA[asset]
        run_observation_mp(asset, label_idx, fl)
        return {"name": name, "ok": True, "seconds": time.time() - t0,
                "error": None}
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        print(f"  [worker {os.getpid()}] FAILED {name}: {exc}\n{tb}", flush=True)
        return {"name": name, "ok": False, "seconds": time.time() - t0,
                "error": str(exc)}


# --------------------------------------------------------------------------- #
#  Single observation pipeline (mirrors base.run_observation)
# --------------------------------------------------------------------------- #
def run_observation_mp(asset, label_idx, fl):
    """Reproduce base.run_observation EXACTLY, with a provenance telemetry note."""
    observation_name = f"{asset}_l_e_vwap{label_idx}"
    print(f"\n=== Observation: {observation_name} "
          f"(pid={os.getpid()}, device={base.DEVICE}) ===", flush=True)

    x_all = base.build_features(fl)             # (n, 8)
    y_all = base.build_labels(fl, label_idx)    # (n,)
    ts_all = fl[:, base.TS_I]                   # (n,)
    n = x_all.shape[0]

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = base.chronological_split(n)

    x_tr, y_tr = x_all[tr_s:tr_e], y_all[tr_s:tr_e]
    x_va, y_va = x_all[va_s:va_e], y_all[va_s:va_e]
    x_te, y_te = x_all[te_s:te_e], y_all[te_s:te_e]
    ts_te = ts_all[te_s:te_e]

    # No label transform for the MSE study — raw labels used directly
    train_loader = base.make_loader(x_tr, y_tr, shuffle=True)
    val_loader = base.make_loader(x_va, y_va, shuffle=False)
    test_loader = base.make_loader(x_te, y_te, shuffle=False)

    model = base.build_model()

    if base.NO_TRAIN:
        print("    [NO-TRAIN] skipping fit loop (dry run)", flush=True)
        model.to(base.DEVICE)
        train_criterion = base.make_criterion()
        train_loss = base._eval_loss(model, train_loader, train_criterion)
        val_loss = base._eval_loss(model, val_loader, train_criterion)
        history = {"train_loss": [train_loss], "val_loss": [val_loss]}
        epochs_done = 0
        train_secs = 0.0
    else:
        history, epochs_done, train_secs = base.train_model(
            model, train_loader, val_loader)

    # ---- predictions ----
    y_true, y_pred = base.predict(model, test_loader)
    # No inverse transform needed for MSE study

    # ---- Eval_ modules ----
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
        "Eval_Convergence_Plot": base.fig_convergence(history),
        "Eval_Certainty_Distribution": base.fig_certainty_distribution(y_pred),
        "Eval_Error_Distribution_Curve": base.fig_error_distribution_curve(
            edc_centers, edc_maes),
        "Eval_Prediction_Heatmap": base.fig_prediction_heatmap(conf_matrix),
        "Eval_Rolling_Temporal_Error": base.fig_rolling_temporal_error(
            roll_times, roll_vals),
    }

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
        "model_type": base.MODEL_TYPE,
        "input_features": base.FEATURE_NAMES,
        "sequence_length": 1,
        "target_labels": [f"l_e_vwap[{label_idx}]"],
    }
    telemetry = {
        "total_parameters": base.count_parameters(model),
        "epochs_completed": epochs_done,
        "batch_size": base.BATCH_SIZE,
        "hardware_utilized": (f"CUDA: {torch.cuda.get_device_name(0)}"
                              if torch.cuda.is_available() else "CPU"),
        "total_train_time_seconds": round(train_secs, 2),
        "rows_train": int(x_tr.shape[0]),
        "rows_val": int(x_va.shape[0]),
        "rows_test": int(x_te.shape[0]),
        "feature_count": len(base.FEATURE_COLS),
        "no_train_dry_run": bool(base.NO_TRAIN),
        "label_transform": base.LABEL_TRANSFORM_NAME,
        # Provenance: distinguishes reports from the multiprocessing trainer.
        "trainer": "torch_multiprocessing",
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

    base.generate_and_upload_report(observation_name, metadata, model_arch,
                                    telemetry, metrics, plotly_figs)

    if base.NO_TRAIN:
        print("    [NO-TRAIN] skipping model-param upload (untrained weights)",
              flush=True)
    else:
        base.upload_model_params(observation_name, model)

    # ---- release per-observation memory ----
    del (model, x_all, y_all, x_tr, y_tr, x_va, y_va, x_te, y_te,
         train_loader, val_loader, test_loader, y_true, y_pred, plotly_figs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Observation spec enumeration (14 tiny task tuples)
# --------------------------------------------------------------------------- #
def build_specs(fl_by_asset):
    """Enumerate (asset, label_idx) for every loaded asset."""
    specs = []
    for asset in fl_by_asset:
        for label_idx in base.LABEL_INDICES:
            specs.append((asset, label_idx))
    return specs


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=("Lookback/Lookahead MLP sweep (MSE) — "
                     "torch.multiprocessing process-pool trainer."))
    parser.add_argument(
        "-notrain", "--notrain", dest="notrain", action="store_true",
        help=("Skip the training fit loop. Runs the whole pipeline (data load, "
              "model build, evaluation, reporting, GCS upload) using the "
              "untrained model's forward pass for a CPU dry run."))
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help=("Number of worker processes in the pool. Default 4."))
    args = parser.parse_args()

    no_train = bool(args.notrain)
    base.NO_TRAIN = no_train
    concurrency = max(1, int(args.concurrency))
    device = base.DEVICE

    torch.backends.cudnn.benchmark = True

    if no_train:
        print("########## NO-TRAIN DRY RUN ENABLED ##########", flush=True)
        print("  Training is skipped. Predictions use an UNTRAINED model.",
              flush=True)
    print(f"  device={device} | concurrency={concurrency} "
          f"(torch.multiprocessing, spawn)", flush=True)

    gcs_json_key_file()

    # ---- Phase 1: load all data once, in the parent ----
    fl_by_asset, per_asset_load, load_seconds = load_all_data()
    if not fl_by_asset:
        print("No data loaded; aborting.", flush=True)
        return

    specs = build_specs(fl_by_asset)
    n_total = len(specs)

    print(f"\n########## PHASE 2/3: train {n_total} models in a pool of "
          f"{concurrency} processes ##########", flush=True)

    results = []
    train_t0 = time.time()

    if concurrency == 1:
        _init_worker(fl_by_asset, no_train)
        for spec in specs:
            results.append(_run_one(spec))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=concurrency,
                      initializer=_init_worker,
                      initargs=(fl_by_asset, no_train)) as pool:
            for res in pool.imap_unordered(_run_one, specs):
                tag = "OK " if res["ok"] else "ERR"
                print(f"  [{tag}] {res['name']} ({res['seconds']:.1f}s)",
                      flush=True)
                results.append(res)

    train_seconds = time.time() - train_t0

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    print("\n========== SWEEP COMPLETE (torch.multiprocessing) ==========",
          flush=True)
    print(f"  observations attempted : {n_total}", flush=True)
    print(f"  completed (ok)         : {len(successes)}", flush=True)
    print(f"  failures               : {len(failures)}", flush=True)
    for f in failures:
        print(f"    - {f['name']}: {f['error']}", flush=True)

    print("\n  ---- timing report ----", flush=True)
    print(f"  total data-load time   : {load_seconds:8.2f}s", flush=True)
    print(f"  total training (pool)  : {train_seconds:8.2f}s "
          f"(wall-clock, {concurrency} workers)", flush=True)
    print(f"  observations completed : {len(successes)}/{n_total}", flush=True)
    if successes:
        sum_worker = sum(r["seconds"] for r in successes)
        print(f"  sum of per-model times : {sum_worker:8.2f}s "
              f"(serial-equivalent compute)", flush=True)
        print(f"  avg per-model time     : "
              f"{sum_worker / len(successes):8.2f}s", flush=True)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
