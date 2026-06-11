#!/usr/bin/env bash
#
# do_workflow.sh — end-to-end study runner for a fresh (e.g. RunPod) environment.
#
# Steps:
#   1. Clone the pycrypto repository (skips cloning if already present).
#   2. cd into scripts/studies/lookback_lookahead_nn/ inside the clone.
#   3. Install the study's Python requirements.
#   4. Run the lookback_lookahead_nn.py study.
#   5. Wait 5 seconds.
#   6. Run pod_self_terminate.py (terminates the current RunPod pod).
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

# 2. cd into the study folder.
cd "${REPO_NAME}/${STUDY_SUBDIR}"
echo "Working directory: $(pwd)"

# 3. Install requirements.
echo "Installing requirements ..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 4. Run the study.
echo "Running lookback_lookahead_nn.py ..."
python3 lookback_lookahead_nn.py

# 5. Wait 5 seconds.
echo "Waiting 5 seconds before self-termination ..."
sleep 5

# 6. Self-terminate the pod.
echo "Running pod_self_terminate.py ..."
python3 pod_self_terminate.py
