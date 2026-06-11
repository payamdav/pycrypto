# RunPod Tools Package

## Identity

| Key             | Value                                  |
|-----------------|----------------------------------------|
| Package path    | `packages/tools/runpod_tools/`         |
| Purpose         | RunPod.io utility functions: runtime environment detection, secret retrieval, and pod self-termination. |

| Exported function                | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `is_running_environment_runpod`  | Detect whether the process runs on a RunPod pod; returns `RUNPOD_POD_ID` or `None` |
| `get_secret`                     | Return the value of a RunPod secret by key                                  |
| `pod_self_terminate`             | Stop and terminate the current RunPod pod via the RunPod GraphQL API        |

---

## How RunPod Exposes State

- **Pod identity / networking** — RunPod sets environment variables on every pod it starts, including `RUNPOD_POD_ID`, `RUNPOD_PUBLIC_IP`, and `RUNPOD_GPU_COUNT`. The presence of `RUNPOD_POD_ID` is the canonical signal that the process is running on RunPod.
- **Secrets** — Every secret configured for a pod is injected into the environment as a variable named `RUNPOD_SECRET_{secret_key}`. For example, a secret named `GCP_KEY` is available as `RUNPOD_SECRET_GCP_KEY`.

---

## Setup

The package directory name (`runpod_tools`) is import-safe, but it lives under
`packages/tools/`. Use `sys.path` manipulation to import it:

```python
import sys
sys.path.insert(0, "/path/to/pycrypto/packages/tools/runpod_tools")
import runpod_tools
```

Or install dependencies first:

```
pip install -r packages/tools/runpod_tools/requirements.txt
```

The only dependency is `requests` (used by `pod_self_terminate`).

---

## `is_running_environment_runpod`

```python
def is_running_environment_runpod() -> str | None:
```

**Returns:** `RUNPOD_POD_ID` (a `str`) when running on RunPod, otherwise `None`.

**Side effect:** Prints the values of `RUNPOD_POD_ID`, `RUNPOD_PUBLIC_IP`, and
`RUNPOD_GPU_COUNT` on a **single line** so the pod's runtime identity is visible
in logs, e.g.:

```
RUNPOD_POD_ID=abc123 RUNPOD_PUBLIC_IP=1.2.3.4 RUNPOD_GPU_COUNT=2
```

---

## `get_secret`

```python
def get_secret(secret_key: str) -> str:
```

Returns the value of the RunPod secret `secret_key` by reading the
`RUNPOD_SECRET_{secret_key}` environment variable.

**Raises:** `KeyError` if the secret is not present in the environment.

```python
api_key = get_secret("RUN_POD_API_KEY")
gcp_key = get_secret("GCP_KEY")
```

---

## `pod_self_terminate`

```python
def pod_self_terminate() -> None:
```

Stops and **terminates** (destroys, releasing compute) the current RunPod pod.

- Resolves the pod id from `RUNPOD_POD_ID`.
- Resolves the RunPod API key from the `RUN_POD_API_KEY` secret (via `get_secret`).
- Issues a `podTerminate` mutation against the RunPod GraphQL API
  (`https://api.runpod.io/graphql`).

**Raises:**
- `RuntimeError` if not running on RunPod (`RUNPOD_POD_ID` unset) or if the
  termination request fails.
- `KeyError` if the `RUN_POD_API_KEY` secret is missing.

> Destructive: a successful call destroys the pod and releases its compute. The
> calling process is killed once the pod shuts down.

---

## Usage Example

```python
import sys
sys.path.insert(0, "/path/to/pycrypto/packages/tools/runpod_tools")

from runpod_tools import (
    is_running_environment_runpod,
    get_secret,
    pod_self_terminate,
)

pod_id = is_running_environment_runpod()
if pod_id is not None:
    print(f"On RunPod pod {pod_id}")
    token = get_secret("SOME_API_TOKEN")
    # ... do work ...
    pod_self_terminate()   # shut down and destroy this pod
else:
    print("Not running on RunPod.")
```

---

## Notes for Agents

- **Detection key is `RUNPOD_POD_ID`.** Treat its presence as the definitive
  signal of a RunPod environment.
- **Secrets are environment variables** named `RUNPOD_SECRET_{key}`. `get_secret`
  abstracts this prefix away — pass only the bare key (e.g. `"GCP_KEY"`).
- **`pod_self_terminate` is destructive.** Only call it at the very end of a
  workflow when the pod's work is complete (see
  `scripts/studies/lookback_lookahead_nn/do_workflow.sh`).
- The GCS tools package consumes the same RunPod conventions: when run on a pod,
  `gcs_json_key_file()` reads the `GCP_KEY` secret from `RUNPOD_SECRET_GCP_KEY`
  to materialize the service-account JSON. See `agents/packages/gcs_tools.md`.
