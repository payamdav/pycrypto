#!/usr/bin/env python3
"""
Test / utility script: self-terminate the current RunPod pod.

This is a thin wrapper around ``packages/tools/runpod_tools`` used to exercise
the RunPod helpers from a real pod. It:

    1. Reports whether the process is running on RunPod (prints the pod's
       RUNPOD_POD_ID / RUNPOD_PUBLIC_IP / RUNPOD_GPU_COUNT on a single line).
    2. If on RunPod, calls ``pod_self_terminate()`` to stop and destroy the pod.
       The RunPod API key is resolved from the ``RUN_POD_API_KEY`` secret.
    3. If not on RunPod, exits without taking any destructive action.

RunPod helpers : packages/tools/runpod_tools
"""

import os
import sys

# --------------------------------------------------------------------------- #
#  Repository / package import bootstrap
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_RUNPOD_TOOLS_DIR = os.path.join(_REPO_ROOT, "packages", "tools", "runpod_tools")
if _RUNPOD_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _RUNPOD_TOOLS_DIR)

from runpod_tools import (  # noqa: E402
    is_running_environment_runpod,
    pod_self_terminate,
)


def main() -> int:
    pod_id = is_running_environment_runpod()
    if pod_id is None:
        print("Not running on RunPod — nothing to terminate.", flush=True)
        return 0

    print(f"Running on RunPod pod '{pod_id}'. Requesting self-termination...",
          flush=True)
    pod_self_terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
