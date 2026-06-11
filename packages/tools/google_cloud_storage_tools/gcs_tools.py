import os
import io
import json
import base64
import google.cloud.storage
import google.oauth2.service_account


def _normalize_key_json(raw: str) -> str:
    """Return a valid service-account JSON string from a raw secret value.

    Secret stores (RunPod env vars / secrets, Colab/Kaggle userdata) frequently
    mangle a raw multi-line JSON key — newlines, quotes or braces get dropped,
    leaving a truncated, unparseable value. Storing the key **base64-encoded**
    avoids this entirely: base64 is single-line and contains only
    ``A-Z a-z 0-9 + / =``.

    This accepts the value either as raw JSON or as base64-encoded JSON and
    returns valid JSON text. If the value cannot be interpreted as either, it is
    returned unchanged so the caller surfaces the original parse error.
    """
    s = raw.strip()
    # Already valid JSON?
    try:
        json.loads(s)
        return s
    except ValueError:
        pass
    # base64-encoded JSON?
    try:
        decoded = base64.b64decode(s, validate=True).decode("utf-8")
        json.loads(decoded)
        return decoded
    except Exception:  # noqa: BLE001 — fall through to returning the raw value
        pass
    return raw


def _make_client() -> google.cloud.storage.Client:
    key_path = gcs_json_key_file()
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_path)
    return google.cloud.storage.Client(credentials=credentials)


def gcs_json_key_file(key_file: str = "gcp_service_account_key.json", secret_key: str = "GCP_KEY") -> str:
    """Resolve the path to a GCP service account JSON key file.

    Detection order: Google Colab → Kaggle → RunPod → other (local file).
    """
    # Branch 1: Google Colab
    try:
        import google.colab
        try:
            import google.colab.userdata
        except ImportError:
            pass
        json_str = google.colab.userdata.get(secret_key)
        path = os.path.join(os.getcwd(), key_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_normalize_key_json(json_str))
        return os.path.abspath(path)
    except ImportError:
        pass

    # Branch 2: Kaggle — detected via env var or kaggle_secrets import
    in_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    if not in_kaggle:
        try:
            import kaggle_secrets
            in_kaggle = True
        except ImportError:
            pass

    if in_kaggle:
        try:
            import kaggle_secrets
        except ImportError:
            pass
        json_str = kaggle_secrets.UserSecretsClient().get_secret(secret_key)
        path = os.path.join(os.getcwd(), key_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_normalize_key_json(json_str))
        return os.path.abspath(path)

    # Branch 3: RunPod — secrets are injected as env vars. RunPod's documented
    # convention prefixes them with "RUNPOD_SECRET_", but some pod templates
    # expose them under their bare name (e.g. plain "GCP_KEY"). Try the
    # prefixed name first, then fall back to the bare key name.
    if os.environ.get("RUNPOD_POD_ID"):
        json_str = (os.environ.get(f"RUNPOD_SECRET_{secret_key}")
                    or os.environ.get(secret_key))
        if json_str:
            path = os.path.join(os.getcwd(), key_file)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_normalize_key_json(json_str))
            return os.path.abspath(path)

    # Branch 4: Other (local file)
    path = os.path.join(os.getcwd(), key_file)
    if os.path.exists(path):
        return os.path.abspath(path)
    raise FileNotFoundError(
        f"GCP key file not found: '{os.path.abspath(path)}'. "
        "Place your GCP service account JSON key file at that path, "
        "or provide the correct filename via the key_file parameter."
    )


def list_files(bucket_name: str) -> list[str]:
    """Return a list of all blob names in the given GCS bucket."""
    client = _make_client()
    return [blob.name for blob in client.list_blobs(bucket_name)]


def read_file(bucket_name: str, key: str, content_type: str = "application/octet-stream") -> bytes:
    """Download a GCS object and return its contents as bytes.

    content_type is accepted for API consistency but is not passed to
    download_as_bytes() — GCS infers the type from the stored object.
    """
    client = _make_client()
    blob = client.bucket(bucket_name).blob(key)
    return blob.download_as_bytes()


def save_file(bucket_name: str, key: str, path: str = None) -> str:
    """Download a GCS object to a local file and return its absolute path.

    If path is None, the file is saved to os.getcwd()/basename(key).
    """
    client = _make_client()
    blob = client.bucket(bucket_name).blob(key)
    if path is None:
        destination = os.path.join(os.getcwd(), os.path.basename(key))
    else:
        destination = path
    blob.download_to_filename(destination)
    return os.path.abspath(destination)


def write_file(bucket_name: str, key: str, file_obj, content_type: str = "application/octet-stream") -> None:
    """Upload a file-like object to GCS.

    content_type is passed to upload_from_file to set the object's MIME type.
    """
    client = _make_client()
    blob = client.bucket(bucket_name).blob(key)
    blob.upload_from_file(file_obj, content_type=content_type)
