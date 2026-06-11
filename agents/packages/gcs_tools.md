# GCS Tools Package

## Identity

| Key             | Value                                        |
|-----------------|----------------------------------------------|
| Package path    | `packages/tools/google_cloud_storage_tools/` |
| Purpose         | GCS utility functions with automatic credential resolution across Google Colab, Kaggle, RunPod, and other Python environments. |

| Exported function   | Description                                           |
|---------------------|-------------------------------------------------------|
| `gcs_json_key_file` | Resolve and return the path to a GCP service account JSON key file — **must be called first before any GCS operation** |
| `list_files`        | List all object names in a GCS bucket                 |
| `read_file`         | Download a GCS object and return its contents as bytes |
| `save_file`         | Download a GCS object to a local file                 |
| `write_file`        | Upload a file-like object to a GCS bucket             |

> **Mandatory initialization:** `gcs_json_key_file()` **must** be called once before any call to `list_files`, `read_file`, `save_file`, or `write_file`. It resolves and writes the credential file to disk; without it the GCS client cannot authenticate.

---

## Setup

Use `sys.path` manipulation or dot notation via the package path:

```python
import sys
sys.path.insert(0, "/path/to/pycrypto/packages/tools/google_cloud_storage_tools")
import gcs_tools
```

Or install dependencies first:

```
pip install -r packages/tools/google_cloud_storage_tools/requirements.txt
```

See `packages/tools/google_cloud_storage_tools/requirements.txt` for the full dependency list (`google-cloud-storage`, `google-auth`).

---

## `gcs_json_key_file`

```python
def gcs_json_key_file(key_file: str = "gcp_service_account_key.json", secret_key: str = "GCP_KEY") -> str:
```

### Parameters

| Parameter  | Default                        | Description                                                        |
|------------|--------------------------------|--------------------------------------------------------------------|
| `key_file` | `"gcp_service_account_key.json"` | Filename used when writing the key to disk (Colab / Kaggle only) |
| `secret_key` | `"GCP_KEY"`                  | Secret name to retrieve from Colab userdata or Kaggle secrets      |

**Returns:** `str` — absolute path to the resolved JSON key file.

### Environment Detection

| Environment | Detection method                                                   | Behavior                                                                          |
|-------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Colab       | `import google.colab` succeeds                                     | Reads JSON string from `google.colab.userdata.get(secret_key)`, writes to disk, returns path |
| Kaggle      | `KAGGLE_KERNEL_RUN_TYPE` env var is non-empty, or `import kaggle_secrets` succeeds | Reads JSON string from `kaggle_secrets.UserSecretsClient().get_secret(secret_key)`, writes to disk, returns path |
| RunPod      | `RUNPOD_POD_ID` env var is set                                     | Reads JSON string from the `RUNPOD_SECRET_{secret_key}` env var, falling back to the bare `{secret_key}` env var if the prefixed one is unset; writes to disk, returns path |
| Other       | None of the above matched                                          | Looks for `key_file` in `os.getcwd()`; returns its absolute path if found, raises `FileNotFoundError` otherwise |

> **RunPod:** the JSON key is materialized from the `GCP_KEY` RunPod secret (exposed as the `RUNPOD_SECRET_GCP_KEY` environment variable, or as a bare `GCP_KEY` variable on pod templates that omit the prefix). See `agents/packages/runpod_tools.md` for RunPod conventions.

> **Secret value format (all environments):** the secret may hold the service-account key either as **raw JSON** or, recommended, as **base64-encoded JSON**. `gcs_json_key_file()` auto-detects which and writes valid JSON to disk. Base64 is strongly preferred for secret/env-var storage because raw multi-line JSON is frequently truncated or mangled by secret stores (newlines/quotes dropped), whereas base64 is single-line and uses only `A-Z a-z 0-9 + / =`. Encode with `base64 -w0 your-key.json` (or `python3 -c "import base64;print(base64.b64encode(open('your-key.json','rb').read()).decode())"`) and store the resulting single line as the secret value.

---

## `list_files`

> **Requires** `gcs_json_key_file()` to be called before use.

```python
def list_files(bucket_name: str) -> list[str]:
```

Lists all objects in the specified GCS bucket. Returns a `list[str]` of blob names (object keys).

---

## `read_file`

> **Requires** `gcs_json_key_file()` to be called before use.

```python
def read_file(bucket_name: str, key: str, content_type: str = "application/octet-stream") -> bytes:
```

Downloads a GCS object and returns its raw contents as `bytes`.

> `content_type` is accepted for API consistency but is not used in the read path — GCS infers the type from the stored object.

---

## `save_file`

> **Requires** `gcs_json_key_file()` to be called before use.

```python
def save_file(bucket_name: str, key: str, path: str = None) -> str:
```

Downloads a GCS object to a local file. Returns the absolute path (`str`) of the saved file.

- If `path` is `None`, the file is saved to `os.path.join(os.getcwd(), os.path.basename(key))`.
- If `path` is provided, the file is saved to that location.

---

## `write_file`

> **Requires** `gcs_json_key_file()` to be called before use.

```python
def write_file(bucket_name: str, key: str, file_obj, content_type: str = "application/octet-stream") -> None:
```

Uploads a file-like object to the specified GCS bucket under the given key. Returns `None`.

> `content_type` is passed directly to `upload_from_file` to set the object's MIME type in GCS.

---

## Usage Examples

```python
import sys
import io

# The package directory name contains hyphens — use sys.path to import it
sys.path.insert(0, "/path/to/pycrypto/packages/tools/google_cloud_storage_tools")

from gcs_tools import gcs_json_key_file, list_files, read_file, save_file, write_file

# STEP 1 — always call gcs_json_key_file() first to resolve credentials
key_path = gcs_json_key_file()   # resolves from Colab / Kaggle secrets or local file

BUCKET = "my-gcs-bucket"

# List all objects in a bucket
files = list_files(BUCKET)
print(files)  # ["data/candles.parquet", "models/v1.pt", ...]

# Download an object and decode its contents
content_bytes = read_file(BUCKET, "data/notes.txt")
text = content_bytes.decode("utf-8")

# Save an object to disk
local_path = save_file(BUCKET, "data/candles.parquet")
# saved to os.getcwd()/candles.parquet by default

# Save to a specific path
local_path = save_file(BUCKET, "data/candles.parquet", path="/tmp/candles.parquet")

# Upload a file-like object
buffer = io.BytesIO(b"hello world")
write_file(BUCKET, "uploads/hello.txt", buffer, content_type="text/plain")
```

---

## Notes for Agents

- **Always call `gcs_json_key_file()` first.** Every script or notebook that uses `list_files`, `read_file`, `save_file`, or `write_file` must call `gcs_json_key_file()` before any of those functions. Skipping this step will cause authentication failures because the GCS client cannot locate the service account key.
- The package directory name (`google_cloud_storage_tools`) contains hyphens and cannot be imported using dot notation. Use `sys.path` manipulation or install the dependencies from `requirements.txt` and import `gcs_tools` directly.
- Credentials are resolved automatically by `gcs_json_key_file` based on the detected runtime environment (Colab, Kaggle, RunPod, or local file). Callers never need to handle the key file or credential object directly.
- Each function call creates a fresh `google.cloud.storage.Client` instance — there is no client caching or connection pooling across calls.
