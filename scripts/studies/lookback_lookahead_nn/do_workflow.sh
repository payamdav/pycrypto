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
STUDY_SUBDIR="scripts/studies/lookback_lookahead_nn"

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

# 4. Run the study.
echo "Running lookback_lookahead_nn.py ..."
python3 "${STUDY_SUBDIR}/lookback_lookahead_nn.py"

# 5. Upload the Observation Report Viewer SPA.
echo "Running upload_web_app.py ..."
python3 "${STUDY_SUBDIR}/upload_web_app.py"

# 6. Wait 5 seconds.
echo "Waiting 5 seconds before self-termination ..."
sleep 5

# 7. Self-terminate the pod.
echo "Running pod_self_terminate.py ..."
python3 "${STUDY_SUBDIR}/pod_self_terminate.py"
