#!/usr/bin/env bash
#
# run_quantile_studies_workflow.sh — end-to-end runner for ALL FOUR quantile
# single-head lookback/lookahead studies on a fresh (e.g. RunPod) environment.
#
# Unlike each study's own do_workflow.sh (which runs a single study), this
# top-level script trains all four quantile studies back-to-back in one pod
# session, sharing the one-time fl_data download and a single pip install, then
# self-terminates the pod once at the very end.
#
# The four studies (spec: ai_chats/lookback_lookahead_quantile_studies.md):
#   1. lookback_lookahead_lstm_singlehead_quantile      (LSTM, Huber)
#   2. lookback_lookahead_lstm_singlehead_quantile_mse  (LSTM, MSE)
#   3. lookback_lookahead_gru_singlehead_quantile       (GRU,  Huber)
#   4. lookback_lookahead_gru_singlehead_quantile_mse   (GRU,  MSE)
#
# Steps (all run from the repo root):
#   1. Upgrade pip and install every study's Python requirements.
#   2. Download the shared fl_data cache to ~/data ONCE.
#   3. For each of the four studies:
#        a. Train with the torch.multiprocessing trainer
#           (lookback_lookahead_nn_mp.py) using --concurrency N.
#        b. Run upload_web_app.py to (re)deploy that study's SPA.
#   4. Self-terminate the RunPod pod.
#
# Usage:
#   bash run_quantile_studies_workflow.sh [--concurrency N] [extra trainer args]
#
#   --concurrency N : number of worker processes the mp trainer spawns to share
#                     the single GPU (default 4). Forwarded to every study's
#                     lookback_lookahead_nn_mp.py. Any additional args (e.g.
#                     -notrain) are forwarded verbatim to each trainer.
#
set -euo pipefail

# --- Resolve the repository root (this script lives at the repo root). --------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
echo "Working directory (repo root): $(pwd)"

# --- Parse --concurrency; forward everything else to the trainer. -------------
CONCURRENCY=4
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --concurrency=*)
            CONCURRENCY="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
echo "Concurrency (mp workers): ${CONCURRENCY}"

# The four study sub-directories, in execution order.
STUDIES=(
    "scripts/studies/lookback_lookahead_lstm_singlehead_quantile"
    "scripts/studies/lookback_lookahead_lstm_singlehead_quantile_mse"
    "scripts/studies/lookback_lookahead_gru_singlehead_quantile"
    "scripts/studies/lookback_lookahead_gru_singlehead_quantile_mse"
)

# --- 1. Upgrade pip and install requirements. ---------------------------------
# All four studies' requirements.txt are identical, but install each so the
# script stays correct even if they diverge later (pip is a near no-op once the
# packages are present).
echo "Upgrading pip ..."
python3 -m pip install --upgrade pip
for study in "${STUDIES[@]}"; do
    echo "Installing requirements from ${study}/requirements.txt ..."
    python3 -m pip install -r "${study}/requirements.txt"
done

# --- 2. Populate the shared local fl_data cache (~/data) ONCE. ----------------
# All four studies read fl_data from ~/data instead of GCS. The download script
# lives in the base study folder and skips assets already cached.
echo "Downloading fl_data to ~/data (one-time) ..."
python3 "scripts/studies/lookback_lookahead_nn/download_fl_data.py"

# --- 3. Train + upload each study using the multiprocessing trainer. ----------
for study in "${STUDIES[@]}"; do
    echo "########## STUDY: ${study} ##########"

    echo "Training (lookback_lookahead_nn_mp.py, --concurrency ${CONCURRENCY}) ..."
    python3 "${study}/lookback_lookahead_nn_mp.py" \
        --concurrency "${CONCURRENCY}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

    echo "Uploading web app (upload_web_app.py) ..."
    python3 "${study}/upload_web_app.py"
done

# --- 4. Self-terminate the pod. -----------------------------------------------
echo "All four studies complete. Waiting 5 seconds before self-termination ..."
sleep 5

echo "Running pod_self_terminate.py ..."
python3 "scripts/studies/lookback_lookahead_lstm_singlehead_quantile/pod_self_terminate.py"
