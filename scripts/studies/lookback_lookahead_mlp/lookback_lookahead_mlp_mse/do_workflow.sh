#!/usr/bin/env bash
#
# do_workflow.sh — end-to-end study runner for a fresh (e.g. RunPod) environment.
#
# All scripts are executed with the repository ROOT as the current working
# directory.
#
# Steps (all run from the repo root):
#   1. Clone the pycrypto repository (skips cloning if already present).
#   2. cd into the repository root.
#   3. Install the study's Python requirements.
#   4. Download fl_data to ~/data (one-time, shared cache).
#   5. Run lookback_lookahead_nn_mp.py (multiprocessing trainer).
#   6. Run upload_web_app.py (deploy the Observation Report Viewer SPA).
#   7. Wait 5 seconds.
#   8. Run pod_self_terminate.py (terminates the current RunPod pod).
#
set -euo pipefail

REPO_URL="https://github.com/payamdav/pycrypto"
REPO_NAME="pycrypto"
STUDY_SUBDIR="scripts/studies/lookback_lookahead_mlp/lookback_lookahead_mlp_mse"

# 1. Clone the repository if it is not already present.
if [ ! -d "${REPO_NAME}" ]; then
    echo "Cloning ${REPO_URL} ..."
    git clone "${REPO_URL}" "${REPO_NAME}"
else
    echo "Repository '${REPO_NAME}' already exists — skipping clone."
fi

# 2. cd into the repository ROOT.
cd "${REPO_NAME}"
echo "Working directory (repo root): $(pwd)"

# 3. Install requirements.
echo "Installing requirements ..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "${STUDY_SUBDIR}/requirements.txt"

# 4. Populate the shared local fl_data cache (~/data) ONCE.
echo "Downloading fl_data to ~/data (one-time) ..."
python3 "scripts/studies/lookback_lookahead_rnn/lookback_lookahead_nn/download_fl_data.py"

# 5. Run the multiprocessing trainer.
echo "Running lookback_lookahead_nn_mp.py ..."
python3 "${STUDY_SUBDIR}/lookback_lookahead_nn_mp.py" "$@"

# 6. Upload the Observation Report Viewer SPA.
echo "Running upload_web_app.py ..."
python3 "${STUDY_SUBDIR}/upload_web_app.py"

# 7. Wait 5 seconds.
echo "Waiting 5 seconds before self-termination ..."
sleep 5

# 8. Self-terminate the pod.
echo "Running pod_self_terminate.py ..."
python3 "${STUDY_SUBDIR}/pod_self_terminate.py"
