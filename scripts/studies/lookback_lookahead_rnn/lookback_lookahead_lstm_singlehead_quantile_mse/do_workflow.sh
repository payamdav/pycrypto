#!/usr/bin/env bash
#
# do_workflow.sh — end-to-end study runner for a fresh (e.g. RunPod) environment.
#
# All scripts are executed with the repository ROOT as the current working
# directory. This matters because gcs_tools.gcs_json_key_file() materializes the
# GCP service-account JSON key into os.getcwd(); running every script from the
# repo root means the key file is written once and reused by each script that
# needs it (lookback_lookahead_nn.py and upload_web_app.py).
#
# Steps (all run from the repo root):
#   1. Clone the pycrypto repository (skips cloning if already present).
#   2. cd into the repository root.
#   3. Install the study's Python requirements.
#   4. Run the lookback_lookahead_nn.py study.
#   5. Run upload_web_app.py (deploy the Observation Report Viewer SPA).
#   6. Wait 5 seconds.
#   7. Run pod_self_terminate.py (terminates the current RunPod pod).
#
set -euo pipefail

REPO_URL="https://github.com/payamdav/pycrypto"
REPO_NAME="pycrypto"
STUDY_SUBDIR="scripts/studies/lookback_lookahead_rnn/lookback_lookahead_lstm_singlehead_quantile_mse"

# 1. Clone the repository if it is not already present in the current directory.
if [ ! -d "${REPO_NAME}" ]; then
    echo "Cloning ${REPO_URL} ..."
    git clone "${REPO_URL}" "${REPO_NAME}"
else
    echo "Repository '${REPO_NAME}' already exists — skipping clone."
fi

# 2. cd into the repository ROOT. Every script below runs from here so the GCP
#    key file resolved by gcs_json_key_file() lands in (and is read from) the
#    repo root.
cd "${REPO_NAME}"
echo "Working directory (repo root): $(pwd)"

# 3. Install requirements.
echo "Installing requirements ..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "${STUDY_SUBDIR}/requirements.txt"

# 3.5 Populate the shared local fl_data cache (~/data) ONCE. All studies read
#     from ~/data instead of GCS. The download script lives in the base study
#     folder; it skips assets already cached.
echo "Downloading fl_data to ~/data (one-time) ..."
python3 "scripts/studies/lookback_lookahead_rnn/lookback_lookahead_nn/download_fl_data.py"

# 4. Run the study. THREE trainer options are available — exactly ONE is active
#    below; uncomment whichever you want and comment out the others. All three
#    train the SAME 56 observations and produce IDENTICAL reports; they differ
#    only in HOW the work is scheduled on the hardware. All forward extra args
#    (e.g. --concurrency N, -notrain) through "$@".
#
# Option A — sequential trainer: one observation at a time, simplest to reason
# about. No concurrency.
# echo "Running lookback_lookahead_nn.py ..."
# python3 "${STUDY_SUBDIR}/lookback_lookahead_nn.py" "$@"
#
# Option B — CUDA streams trainer (ACTIVE): single process, loads all asset data
# once, keeps it resident on the GPU, and trains the 56 independent models
# concurrently via CUDA streams (one stream per model within a wave).
echo "Running lookback_lookahead_nn_parallel.py ..."
python3 "${STUDY_SUBDIR}/lookback_lookahead_nn_parallel.py" "$@"
#
# Option C — torch.multiprocessing trainer: loads all asset data once in the
# parent, then spawns a pool of --concurrency N worker processes that share the
# single GPU (OS time-slicing / MPS). Per-model procedure is identical.
# echo "Running lookback_lookahead_nn_mp.py ..."
# python3 "${STUDY_SUBDIR}/lookback_lookahead_nn_mp.py" "$@"

# 5. Upload the Observation Report Viewer SPA.
echo "Running upload_web_app.py ..."
python3 "${STUDY_SUBDIR}/upload_web_app.py"

# 6. Wait 5 seconds.
echo "Waiting 5 seconds before self-termination ..."
sleep 5

# 7. Self-terminate the pod.
echo "Running pod_self_terminate.py ..."
python3 "${STUDY_SUBDIR}/pod_self_terminate.py"
