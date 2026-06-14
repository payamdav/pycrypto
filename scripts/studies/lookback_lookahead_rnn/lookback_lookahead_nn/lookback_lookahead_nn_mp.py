#!/usr/bin/env python3
"""
Lookback / Lookahead single-target LSTM study — MULTIPROCESSING trainer.

This is a process-parallel sibling of ``lookback_lookahead_nn.py`` (the
sequential "base" trainer) and ``lookback_lookahead_nn_parallel.py`` (the
single-process CUDA-streams trainer). It trains the SAME 56 observations
(7 assets x 4 label varieties x 2 input varieties) and produces IDENTICAL
reports (same eval modules, same Plotly figures, same master JSON schema, same
GCS report/ location). The ONLY difference is *how* the work is scheduled —
never *what* math each model performs.

Concurrency model (why this file exists): instead of CUDA streams inside one
process, this variant uses ``torch.multiprocessing`` with the ``spawn`` start
method. The user finds a process pool more familiar to reason about and tune.

    * Phase 1 — Load ALL data once, IN THE PARENT PROCESS. Every asset's
      fl_data blob is pulled from GCS exactly one time into host RAM (timed,
      per-asset + total). It is NEVER re-downloaded inside a worker.
    * Phase 2 — Spawn a pool of ``--concurrency N`` worker processes. The
      preloaded host data dict is handed to each worker exactly once via the
      Pool ``initializer`` (pickled once per worker at startup — still a single
      GCS download in the parent). Workers never touch GCS for data.
    * Phase 3 — The 56 observation specs (tiny tuples: asset, label_idx,
      input_variety) are mapped across the pool. Each worker looks up its
      asset's preloaded array from the dict it was given at init, then runs the
      base trainer's procedure verbatim.

Why the results are unchanged (vs. the sequential script):
    * Per-model PROCEDURE is byte-for-byte the same: each worker reuses
      ``base.build_features``, ``base.build_labels``, ``base.chronological_split``,
      ``base.LSTMRegressor``, ``base.make_loader`` and ``base.train_model``
      (same AdamW + Huber + ReduceLROnPlateau + early-stopping + MAX_EPOCHS,
      same BATCH_SIZE, same per-epoch full reshuffle via DataLoader(shuffle=True)).
    * Each observation is independent; running them in separate processes only
      changes *when* their kernels execute, not their math. Workers share the
      single GPU (cuda:0) via OS time-slicing. (Optional throughput boost on
      RunPod: enable NVIDIA MPS so concurrent processes share the GPU more
      efficiently — `nvidia-cuda-mps-control -d` before launch.)
    * Evaluation, report building and GCS upload all REUSE the functions from
      ``lookback_lookahead_nn`` so every report is identical in format and lands
      in the same ``.../reports/`` location. A telemetry note
      ``"trainer": "torch_multiprocessing"`` distinguishes provenance only.

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
import torch.multiprocessing as mp

# --------------------------------------------------------------------------- #
#  Repository / package import bootstrap
#
#  This file lives in scripts/studies/lookback_lookahead_nn/, so the repo root
#  is THREE levels up — matching the base module's bootstrap exactly.
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

    Returns (fl_by_asset, per_asset_seconds, total_seconds). Assets that fail
    to load are simply omitted from the dict (a warning is printed). This is the
    ONLY place data is fetched from GCS — workers receive the preloaded dict and
    never re-download anything.
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

# Module-global populated once per worker by the Pool initializer. Holds the
# preloaded {asset: fl_ndarray} dict so every task in this worker reuses the
# same in-memory data without any GCS access.
_WORKER_DATA = None
_WORKER_NO_TRAIN = False


def _init_worker(fl_by_asset, no_train):
    """Pool initializer: runs ONCE per worker process at startup.

    Receives the preloaded host data dict (pickled from the parent a single
    time) and the dry-run flag, stashing both in module globals for reuse by
    every task this worker handles. Also resolves GCS credentials once per
    worker (needed for report/param UPLOADS — never for data downloads).
    """
    global _WORKER_DATA, _WORKER_NO_TRAIN
    _WORKER_DATA = fl_by_asset
    _WORKER_NO_TRAIN = bool(no_train)
    # The base module's NO_TRAIN flag drives run_observation's dry-run path.
    base.NO_TRAIN = bool(no_train)
    # Each worker needs its own resolved credential file for the report/param
    # upload calls (write_file). gcs_json_key_file() is idempotent.
    try:
        gcs_json_key_file()
    except Exception as exc:  # noqa: BLE001
        print(f"  [worker {os.getpid()}] WARNING: gcs_json_key_file failed: "
              f"{exc}", flush=True)


def _run_one(spec):
    """Train + evaluate + report ONE observation. Runs inside a worker.

    `spec` is a tiny tuple (asset, label_idx, input_variety). The worker looks
    up the preloaded array from the dict it was given at init and reuses the
    base trainer's full procedure via run_observation_mp. Returns a small,
    picklable result dict so the parent can summarize successes/failures and
    aggregate timings. Exceptions are caught and returned (never raised) so one
    failed observation does not kill the pool.
    """
    asset, label_idx, input_variety = spec
    name = f"{asset}_l_e_vwap{label_idx}_{input_variety}"
    t0 = time.time()
    try:
        fl = _WORKER_DATA[asset]
        run_observation_mp(asset, label_idx, input_variety, fl)
        return {"name": name, "ok": True, "seconds": time.time() - t0,
                "error": None}
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        print(f"  [worker {os.getpid()}] FAILED {name}: {exc}\n{tb}", flush=True)
        return {"name": name, "ok": False, "seconds": time.time() - t0,
                "error": str(exc)}


# --------------------------------------------------------------------------- #
#  Single observation pipeline (mirrors base.run_observation; adds telemetry
#  provenance note and reuses base.train_model so results are identical)
# --------------------------------------------------------------------------- #
def run_observation_mp(asset, label_idx, input_variety, fl):
    """Reproduce base.run_observation EXACTLY, with a provenance telemetry note.

    This mirrors lookback_lookahead_nn.run_observation step-for-step, reusing
    the base feature/label/split builders, model class, loaders, train_model,
    eval modules, Plotly builders and report/upload helpers. The only addition
    is the ``"trainer": "torch_multiprocessing"`` telemetry note — metric
    definitions are untouched, so reports are byte-compatible with the other
    trainers' output.
    """
    observation_name = f"{asset}_l_e_vwap{label_idx}_{input_variety}"
    print(f"\n=== Observation: {observation_name} "
          f"(pid={os.getpid()}, device={base.DEVICE}) ===", flush=True)

    x_all = base.build_features(fl, input_variety)   # (n, 24, n_feat)
    y_all = base.build_labels(fl, label_idx)         # (n,)
    ts_all = fl[:, base.TS_I]                        # (n,)
    n = x_all.shape[0]
    n_features = x_all.shape[2]

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = base.chronological_split(n)

    x_tr, y_tr = x_all[tr_s:tr_e], y_all[tr_s:tr_e]
    x_va, y_va = x_all[va_s:va_e], y_all[va_s:va_e]
    x_te, y_te = x_all[te_s:te_e], y_all[te_s:te_e]
    ts_te = ts_all[te_s:te_e]

    train_loader = base.make_loader(x_tr, y_tr, shuffle=True)
    val_loader = base.make_loader(x_va, y_va, shuffle=False)
    test_loader = base.make_loader(x_te, y_te, shuffle=False)

    model = base.LSTMRegressor(n_features=n_features)

    if base.NO_TRAIN:
        # Dry run: skip the fit loop entirely. The untrained model's forward
        # pass still produces predictions so the evaluation/reporting pipeline
        # is fully exercised. A single-point "history" keeps the convergence
        # plot well-formed. Identical to the base script's NO-TRAIN branch.
        print("    [NO-TRAIN] skipping fit loop (dry run)", flush=True)
        model.to(base.DEVICE)
        train_criterion = base.nn.HuberLoss(delta=1.0)
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
        "model_type": "LSTM_Base_2_64_SingleHead",
        "input_features": base.INPUT_FEATURE_NAMES[input_variety],
        "sequence_length": base.SEQ_LEN,
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
        "feature_count": n_features,
        "no_train_dry_run": bool(base.NO_TRAIN),
        # Provenance: distinguishes reports from the multiprocessing trainer.
        # This is a telemetry NOTE only — it does NOT change any metric.
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
#  Observation spec enumeration (the 56 tiny task tuples)
# --------------------------------------------------------------------------- #
def build_specs(fl_by_asset):
    """Enumerate (asset, label_idx, input_variety) for every loaded asset."""
    specs = []
    for asset in fl_by_asset:
        for label_idx in range(base.N_LABEL_VARIETIES):
            for input_variety in base.INPUT_VARIETIES:
                specs.append((asset, label_idx, input_variety))
    return specs


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=("Lookback/Lookahead single-target LSTM sweep — "
                     "torch.multiprocessing process-pool trainer."))
    parser.add_argument(
        "-notrain", "--notrain", dest="notrain", action="store_true",
        help=("Skip the training fit loop. Runs the whole pipeline (data load, "
              "model build, evaluation, reporting, GCS upload) using the "
              "untrained model's forward pass for a CPU dry run."))
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help=("Number of worker processes in the pool (each trains a subset of "
              "the 56 observations, sharing the single GPU via OS time-slicing "
              "or MPS). 1 == effectively sequential. Default 4 — tune per pod."))
    args = parser.parse_args()

    no_train = bool(args.notrain)
    base.NO_TRAIN = no_train
    concurrency = max(1, int(args.concurrency))
    device = base.DEVICE

    # Fixed input shapes per model -> autotune cuDNN once for a free speedup.
    torch.backends.cudnn.benchmark = True

    if no_train:
        print("########## NO-TRAIN DRY RUN ENABLED ##########", flush=True)
        print("  Training is skipped. Predictions use an UNTRAINED model.",
              flush=True)
    print(f"  device={device} | concurrency={concurrency} "
          f"(torch.multiprocessing, spawn)", flush=True)

    # Resolve credentials once in the PARENT before any GCS call (data load).
    gcs_json_key_file()

    # ---- Phase 1: load all data once, in the parent ----
    fl_by_asset, per_asset_load, load_seconds = load_all_data()
    if not fl_by_asset:
        print("No data loaded; aborting.", flush=True)
        return

    specs = build_specs(fl_by_asset)
    n_total = len(specs)

    # ---- Phase 2 & 3: spawn the pool (preloaded data handed in once via the
    #      initializer) and map the 56 specs across it. ----
    print(f"\n########## PHASE 2/3: train {n_total} models in a pool of "
          f"{concurrency} processes ##########", flush=True)

    results = []
    train_t0 = time.time()

    if concurrency == 1:
        # Degenerate / CPU-smoke-test path: run in-process so the whole thing
        # works even where spawning CUDA children is undesirable. Still uses the
        # identical per-observation procedure.
        _init_worker(fl_by_asset, no_train)
        for spec in specs:
            results.append(_run_one(spec))
    else:
        # spawn is REQUIRED for CUDA in child processes. Guarded by the
        # __main__ entry point below (set_start_method there).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=concurrency,
                      initializer=_init_worker,
                      initargs=(fl_by_asset, no_train)) as pool:
            # imap_unordered lets results stream back as workers finish so a
            # slow observation never blocks reporting of the others.
            for res in pool.imap_unordered(_run_one, specs):
                tag = "OK " if res["ok"] else "ERR"
                print(f"  [{tag}] {res['name']} ({res['seconds']:.1f}s)",
                      flush=True)
                results.append(res)

    train_seconds = time.time() - train_t0

    # ---- Timing & success/failure summary ----
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
    # spawn is mandatory for CUDA in child processes; set it before any pool is
    # created. force=True keeps this robust if a default was already set.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already fixed (e.g. re-import in a spawned child) — fine.
        pass
    main()
