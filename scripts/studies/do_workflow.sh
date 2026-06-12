#!/usr/bin/env bash
#
# do_workflow.sh — run the Lookback/Lookahead LSTM study workflow.
#
# Invoke from the REPO ROOT, e.g.:
#     ./scripts/studies/do_workflow.sh
#     ./scripts/studies/do_workflow.sh --concurrency 12
#     ./scripts/studies/do_workflow.sh --notrain          # CPU smoke test
#
# All extra args ("$@") are forwarded to the trainer, so flags like
# --concurrency / --notrain pass straight through.
#
# The workflow has two stages:
#   1. Train all 56 observations and upload their JSON reports + .pt params to
#      gs://payamdpycryptoreports/Lookback_Lookahead_LSTM_Sweep_v1/reports/
#   2. (optional) Deploy the Observation Report Viewer SPA next to the reports.

set -euo pipefail

# Resolve the repo root from this script's own location so it also works if
# called by an absolute/relative path, then run everything from the root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Stage 1 — train the sweep.
# ---------------------------------------------------------------------------

# Original sequential trainer (kept for reference):
# python scripts/studies/lookback_lookahead_nn.py "$@"

# Parallel GPU-resident trainer (active): loads all data once, keeps tensors
# resident on the GPU, and trains the 56 models concurrently via CUDA streams.
python scripts/studies/lookback_lookahead_nn_parallel.py "$@"

# ---------------------------------------------------------------------------
# Stage 2 — deploy the report-viewer SPA (active).
# Uploads scripts/studies/web/ to the study's app/ prefix in the report bucket.
# Comment this out if you only want to (re)train without redeploying the viewer.
# ---------------------------------------------------------------------------
python scripts/studies/upload_web_app.py
