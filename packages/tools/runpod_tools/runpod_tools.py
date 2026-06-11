import os
import json

import requests


# RunPod injects every configured secret into the pod environment as an
# environment variable named "RUNPOD_SECRET_{secret_key}". RunPod also sets a
# number of identity / networking variables (RUNPOD_POD_ID, RUNPOD_PUBLIC_IP,
# RUNPOD_GPU_COUNT, ...) on every pod it starts.
_SECRET_ENV_PREFIX = "RUNPOD_SECRET_"
_RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


def is_running_environment_runpod():
    """Detect whether the current process is running on a RunPod pod.

    Returns the value of the ``RUNPOD_POD_ID`` environment variable when running
    on RunPod, otherwise ``None``.

    As a side effect, this function prints the values of ``RUNPOD_POD_ID``,
    ``RUNPOD_PUBLIC_IP`` and ``RUNPOD_GPU_COUNT`` on a single line so the runtime
    identity of the pod is visible in the logs.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP")
    gpu_count = os.environ.get("RUNPOD_GPU_COUNT")

    # Single-line dump of the pod identity / networking variables.
    print(
        f"RUNPOD_POD_ID={pod_id} RUNPOD_PUBLIC_IP={public_ip} "
        f"RUNPOD_GPU_COUNT={gpu_count}",
        flush=True,
    )

    if not pod_id:
        return None
    return pod_id


def get_secret(secret_key: str) -> str:
    """Return the value of a RunPod secret.

    RunPod exposes every secret to the pod as an environment variable named
    ``RUNPOD_SECRET_{secret_key}``. This reads that variable and returns its
    value.

    Raises:
        KeyError: if the secret is not present in the environment.
    """
    env_name = f"{_SECRET_ENV_PREFIX}{secret_key}"
    value = os.environ.get(env_name)
    if value is None:
        raise KeyError(
            f"RunPod secret '{secret_key}' not found "
            f"(expected environment variable '{env_name}'). "
            "Make sure the secret is configured for this pod."
        )
    return value


def pod_self_terminate() -> None:
    """Stop and terminate the current RunPod pod.

    Resolves the current pod id from ``RUNPOD_POD_ID`` and the RunPod API key
    from the ``RUN_POD_API_KEY`` secret, then issues a ``podTerminate`` mutation
    against the RunPod GraphQL API. After a successful call the pod is destroyed
    (compute released, not merely stopped).

    Raises:
        RuntimeError: if not running on RunPod, or if the termination request
            fails.
        KeyError: if the ``RUN_POD_API_KEY`` secret is missing.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        raise RuntimeError(
            "pod_self_terminate() can only run on a RunPod pod "
            "(RUNPOD_POD_ID is not set)."
        )

    api_key = get_secret("RUN_POD_API_KEY")

    query = "mutation ($input: PodTerminateInput!) { podTerminate(input: $input) }"
    payload = {"query": query, "variables": {"input": {"podId": pod_id}}}

    resp = requests.post(
        _RUNPOD_GRAPHQL_URL,
        params={"api_key": api_key},
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"RunPod podTerminate request failed: HTTP {resp.status_code} "
            f"- {resp.text}"
        )

    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(
            f"RunPod podTerminate returned errors: {json.dumps(body['errors'])}"
        )

    print(f"Pod '{pod_id}' termination requested successfully.", flush=True)
