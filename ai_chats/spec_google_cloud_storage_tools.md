# Specification: `google-cloud-storage-tools` Package

## Task Summary

Create a new reusable package at `packages/tools/google-cloud-storage-tools/` that provides GCS
utility functions with automatic credential resolution across different Python runtime environments
(Google Colab, Kaggle, and other environments such as local Jupyter or terminal). The package
exposes five public functions: `gcs_json_key_file`, `list_files`, `read_file`, `save_file`, and
`write_file`.

---

## Background and Context

Scripts and notebooks in this repository run in several different environments. Each environment
stores GCP service account credentials differently:

- **Google Colab** stores secrets in its user-data vault, accessible via `google.colab.userdata`.
- **Kaggle** stores secrets via its `kaggle_secrets.UserSecretsClient` API.
- **Local / other environments** expect the key JSON file to already exist on disk at a known path.

The package abstracts this environment detection so callers never need to write environment-specific
credential-loading code. All five functions live in a single implementation file `gcs_tools.py`
and are re-exported from `__init__.py`.

---

## Repository Conventions (from `/agents`)

- Reusable packages belong in `packages/<package_name>/` per `agents/general/paths_and_files.md`.
- The target sub-path `packages/tools/` does not yet exist; the agent must create it.
- Every package directory must contain a `requirements.txt` listing its pip dependencies, following
  the pattern established in `packages/candle_loader/requirements.txt` (one dependency per line,
  no version pins unless necessary).
- `__init__.py` re-exports all public symbols, following the pattern in
  `packages/indicators/__init__.py`.
- No `setup.py` or `pyproject.toml` is required unless explicitly stated.

---

## Package Structure

```
packages/tools/google-cloud-storage-tools/
  __init__.py       ← re-exports all 5 functions from gcs_tools
  gcs_tools.py      ← all implementation
  requirements.txt  ← pip dependencies
```

There must be no other files in the package directory.

---

## Dependencies

### `requirements.txt` contents

```
google-cloud-storage
google-auth
```

### Runtime-environment-specific imports (conditional, never listed in requirements.txt)

| Module | Environment | Import guard |
|---|---|---|
| `google.colab.userdata` | Colab only | Wrapped in `try/except ImportError` |
| `kaggle_secrets` | Kaggle only | Wrapped in `try/except ImportError` |

### Standard library (no listing required)

`os`, `io`

---

## Functional Requirements

### Function 1: `gcs_json_key_file`

#### Signature

```python
def gcs_json_key_file(key_file: str = "gcp_service_account_key.json", secret_key: str = "GCP_KEY") -> str:
```

#### Purpose

Detects the current Python runtime environment, retrieves the GCP service account JSON key
content from the appropriate secrets store or verifies it on disk, writes it to a file if
necessary, and returns the absolute path to the key file.

#### Environment Detection Order

Detection must be attempted in exactly this order. The first matching branch wins.

**Branch 1 — Google Colab**

Detection: attempt `import google.colab`. If the import succeeds without error, the runtime is Colab.

Behavior:
1. Import `google.colab.userdata`.
2. Call `google.colab.userdata.get(secret_key)` to retrieve the JSON string.
3. Resolve the output path as `os.path.join(os.getcwd(), key_file)`.
4. Write the retrieved string to that path using UTF-8 encoding (open with `"w"`, `encoding="utf-8"`).
5. Return the absolute path (use `os.path.abspath(...)` on the resolved path).

**Branch 2 — Kaggle**

Detection: check whether the environment variable `KAGGLE_KERNEL_RUN_TYPE` is set (non-empty).
If the variable is absent or empty, additionally attempt `import kaggle_secrets`; if that also
fails, this branch does not match.

Behavior:
1. Import `kaggle_secrets`.
2. Call `kaggle_secrets.UserSecretsClient().get_secret(secret_key)` to retrieve the JSON string.
3. Resolve the output path as `os.path.join(os.getcwd(), key_file)`.
4. Write the retrieved string to that path using UTF-8 encoding.
5. Return the absolute path.

**Branch 3 — Other environments (local Jupyter, terminal, CI, etc.)**

Behavior:
1. Resolve the candidate path as `os.path.join(os.getcwd(), key_file)`.
2. If the file exists at that path, return its absolute path.
3. If the file does not exist, raise `FileNotFoundError` with a message similar to:
   ```
   GCP key file not found: '<resolved_path>'.
   Place your GCP service account JSON key file at that path, or provide the correct filename
   via the key_file parameter.
   ```

#### Return Value

`str` — the resolved absolute path to the key file (always an absolute path string).

#### Notes

- "Project root" is defined as `os.getcwd()` at call time. Do not attempt to walk up the
  directory tree from `__file__`; use `os.getcwd()` directly.
- File writing uses Python's built-in `open()` in text mode (`"w"`) with `encoding="utf-8"`.
- If writing fails (e.g., `PermissionError`), let the exception propagate naturally — do not
  catch it.
- Do not print or log anything inside this function.

---

### Function 2: `list_files`

#### Signature

```python
def list_files(bucket_name: str) -> list[str]:
```

#### Purpose

Returns a list of all object keys (blob names) in the specified GCS bucket.

#### Behavior

1. Call `gcs_json_key_file()` (with its defaults) to obtain the key path.
2. Create credentials: `google.oauth2.service_account.Credentials.from_service_account_file(key_path)`.
3. Create a GCS client: `google.cloud.storage.Client(credentials=credentials)`.
4. Get the bucket: `client.bucket(bucket_name)`.
5. List all blobs: `client.list_blobs(bucket_name)`.
6. Return `[blob.name for blob in blobs]` as a plain Python `list[str]`.

#### Return Value

`list[str]` — unordered list of all blob names in the bucket.

---

### Function 3: `read_file`

#### Signature

```python
def read_file(bucket_name: str, key: str, content_type: str = "application/octet-stream") -> bytes:
```

#### Purpose

Downloads and returns the raw byte content of a GCS object.

#### Behavior

1. Call `gcs_json_key_file()` to obtain the key path.
2. Create credentials and a GCS client (same pattern as `list_files`).
3. Get the blob: `client.bucket(bucket_name).blob(key)`.
4. Download and return raw bytes: `blob.download_as_bytes()`.

#### The `content_type` parameter

The `content_type` parameter is accepted for API consistency and potential future use but is
**not actively passed** to `download_as_bytes()` (the GCS client read path does not require it).
Document this in a brief inline comment in the implementation.

#### Return Value

`bytes` — the raw content of the object.

---

### Function 4: `save_file`

#### Signature

```python
def save_file(bucket_name: str, key: str, path: str = None) -> str:
```

#### Purpose

Downloads a GCS object and saves it to a local file.

#### Behavior

1. Call `gcs_json_key_file()` to obtain the key path.
2. Create credentials and a GCS client.
3. Get the blob: `client.bucket(bucket_name).blob(key)`.
4. Resolve the local destination path:
   - If `path` is `None`: use `os.path.join(os.getcwd(), os.path.basename(key))`.
   - If `path` is provided: use `path` exactly as given.
5. Download to disk: `blob.download_to_filename(destination_path)`.
6. Return `os.path.abspath(destination_path)`.

#### Return Value

`str` — the absolute local path where the file was saved.

#### Notes

- `os.path.basename(key)` handles keys that include path separators (e.g., `"folder/file.parquet"`
  → `"file.parquet"`).
- If the destination directory does not exist, let the OS / GCS client exception propagate naturally.

---

### Function 5: `write_file`

#### Signature

```python
def write_file(bucket_name: str, key: str, file_obj, content_type: str = "application/octet-stream") -> None:
```

#### Purpose

Uploads the contents of a file-like object to a GCS bucket under the specified key.

#### Behavior

1. Call `gcs_json_key_file()` to obtain the key path.
2. Create credentials and a GCS client.
3. Get the blob: `client.bucket(bucket_name).blob(key)`.
4. Upload: `blob.upload_from_file(file_obj, content_type=content_type)`.

#### Parameter Notes

- `file_obj` is any readable file-like object (e.g., `io.BytesIO`, an open binary file handle).
  No type annotation is required beyond `file_obj` (use bare parameter name without annotation,
  or `file_obj: object` if a type annotation is desired).
- `content_type` is passed directly to `upload_from_file`. Default is `"application/octet-stream"`.

#### Return Value

`None`.

---

## `__init__.py` Requirements

Re-export all five public functions so callers can import directly from the package:

```python
from packages.tools.google-cloud-storage-tools.gcs_tools import (
    gcs_json_key_file,
    list_files,
    read_file,
    save_file,
    write_file,
)
```

Note: because the package directory name contains hyphens, which are not valid Python identifiers,
the import inside `__init__.py` must use a relative import:

```python
from .gcs_tools import (
    gcs_json_key_file,
    list_files,
    read_file,
    save_file,
    write_file,
)
```

---

## Error Handling Summary

| Scenario | Behavior |
|---|---|
| Key file missing in non-Colab/non-Kaggle environment | `FileNotFoundError` with descriptive message from `gcs_json_key_file` |
| File write permission error in Colab/Kaggle branch | Original `PermissionError` propagates naturally |
| Colab secret not found | `google.colab.userdata` raises its own exception — propagate naturally |
| Kaggle secret not found | `kaggle_secrets` raises its own exception — propagate naturally |
| GCS bucket or key not found | `google.cloud.exceptions.NotFound` from GCS client — propagate naturally |
| Any other GCS error | Propagate naturally; do not wrap in custom exceptions |

No try/except blocks are used anywhere except for the environment-detection import guards in
`gcs_json_key_file`.

---

## Non-Goals / Out of Scope

- No CLI interface.
- No async variants of any function.
- No caching of credentials or the GCS client between calls. Each function call creates a fresh
  client. (Optimization is out of scope for this specification.)
- No streaming upload/download modes beyond what the standard `google-cloud-storage` client provides.
- No support for GCS bucket creation, deletion, or ACL management.
- No `setup.py` or `pyproject.toml` — the package is used via direct path imports as per
  repository conventions.
- No unit tests — testing is out of scope for this specification.

---

## Assumptions

1. The caller's working directory (`os.getcwd()`) is the project root whenever these functions
   are invoked. This is consistent with how notebooks and scripts in this repository are run.
2. The `google-cloud-storage` package is installed in the target environment by the caller (via
   `requirements.txt` guidance or their own environment setup).
3. The `packages/tools/` sub-directory does not yet exist and must be created as part of the
   implementation.
4. No `__init__.py` is required in `packages/tools/` itself (it is not a Python package — only
   the leaf directory `google-cloud-storage-tools/` is a package).
5. The service account key JSON is a valid JSON string returned by the secrets stores; no
   validation of its contents is required.
6. `google.oauth2` is available as part of the `google-auth` package listed in `requirements.txt`.

---

## Agent Reference File

Per `agents/general/paths_and_files.md`, every reusable package must have a corresponding
description file under `/agents/` so that other agents can discover and use it.

### Location

```
agents/packages/gcs_tools.md
```

The `agents/packages/` sub-directory does not yet exist and must be created as part of the
implementation. It follows the same convention as `agents/datasets/` and `agents/ideas/`.

### Required Content

The file must document the following sections, in order:

1. **Identity** — package path, purpose (one sentence), and a table listing Python import path
   and the five exported function names.

2. **Setup** — how to install / make available (path-based import, `requirements.txt` pointer).

3. **`gcs_json_key_file`** — parameter table (`key_file`, `secret_key`, defaults), return value,
   and the three environment branches (Colab, Kaggle, other) summarized in a table.

4. **`list_files`** — signature, return type, one-line description.

5. **`read_file`** — signature (including `content_type`), return type, note that `content_type`
   is not used in the read path.

6. **`save_file`** — signature (including `path` default), return type, default save location.

7. **`write_file`** — signature (including `content_type`), return type, note that `content_type`
   is passed to `upload_from_file`.

8. **Usage Examples** — a minimal code snippet showing:
   - Importing the package.
   - Calling `list_files`.
   - Calling `read_file` and decoding the result.
   - Calling `save_file`.
   - Calling `write_file` with an `io.BytesIO` object.

9. **Notes for Agents** — bullet points covering:
   - The package directory name contains hyphens; use sys.path manipulation or relative imports.
   - Credentials are resolved automatically; callers never need to handle the key file directly.
   - Each function call creates a fresh GCS client (no caching).

The tone and format must match the existing agent reference files in `agents/datasets/`
(e.g., `agents/datasets/huggingface_candles.md`): markdown tables for schemas, fenced code
blocks for examples, terse descriptive prose.

---

## Acceptance Criteria

1. The directory `packages/tools/google-cloud-storage-tools/` exists and contains exactly three
   files: `__init__.py`, `gcs_tools.py`, `requirements.txt`.
2. `requirements.txt` lists `google-cloud-storage` and `google-auth`, one per line.
3. `__init__.py` uses relative imports and re-exports all five functions.
4. `gcs_tools.py` imports `os` and `io` from the standard library at the top level.
5. `gcs_tools.py` imports `google.cloud.storage` and `google.oauth2.service_account` at the top
   level (not conditionally).
6. `gcs_tools.py` imports `google.colab.userdata` and `kaggle_secrets` only inside
   `gcs_json_key_file`, each guarded by a `try/except ImportError`.
7. `gcs_json_key_file` detects environments in order: Colab first, Kaggle second, other third.
8. In Colab and Kaggle branches, the key file is written to `os.path.join(os.getcwd(), key_file)`.
9. In the fallback branch, `FileNotFoundError` is raised with a message that includes the
   resolved path and guidance on how to fix the issue.
10. All four GCS functions call `gcs_json_key_file()` (no arguments) to resolve credentials.
11. All four GCS functions construct a fresh `google.cloud.storage.Client` per call.
12. `list_files` returns a `list[str]` of blob names.
13. `read_file` returns raw `bytes` and accepts `content_type` without using it in the read call.
14. `save_file` defaults to saving in `os.getcwd()` using `os.path.basename(key)` and returns the
    absolute path.
15. `write_file` passes `content_type` to `blob.upload_from_file` and returns `None`.
16. No exceptions are caught inside any GCS function body; all GCS errors propagate naturally.
17. The file `agents/packages/gcs_tools.md` exists and documents all nine sections listed in the
    **Agent Reference File** section of this spec.
18. `agents/packages/gcs_tools.md` includes working usage examples with import statements, calls
    to all five functions, and notes for agents.

---

## Open Questions

None. The specification is fully determined by the request.

---

## Notes for the Downstream Coding Agent

- The package directory name `google-cloud-storage-tools` contains hyphens. Python cannot use
  this as a module name with dot-notation imports. Use relative imports (`from .gcs_tools import
  ...`) inside `__init__.py` to avoid this issue entirely.
- When creating `packages/tools/`, do NOT add an `__init__.py` inside `packages/tools/` — it is
  not a Python package, only a filesystem grouping folder.
- The existing `packages/__init__.py` is present at the repository root of the packages directory.
  Do not modify it.
- The credential-creation pattern must be:
  ```python
  key_path = gcs_json_key_file()
  credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_path)
  client = google.cloud.storage.Client(credentials=credentials)
  ```
  This pattern must be repeated (or extracted into a private helper `_make_client()`) in each
  of the four GCS functions. Extracting it into a private helper `_make_client() -> google.cloud.storage.Client`
  is recommended to avoid code duplication, but is not required by the spec — either approach
  is acceptable.
- Do not add a `project` argument to `google.cloud.storage.Client(...)`. The project is inferred
  from the service account key file automatically.
- Do not add type stubs, docstrings, or logging unless they arise naturally — keep the
  implementation minimal.
- After creating the package, also create `agents/packages/gcs_tools.md` following the content
  requirements in the **Agent Reference File** section of this spec. The `agents/packages/`
  sub-directory must be created; do not add an `__init__.py` to it. Match the tone and
  markdown style of `agents/datasets/huggingface_candles.md`.
