# HuggingFace Candles Dataset

## Identity

| Key        | Value                        |
|------------|------------------------------|
| Repo ID    | `payamdavaee/candles`        |
| Repo type  | `dataset`                    |
| Visibility | public — no token required   |
| Candle TF  | 1-minute                     |
| Coverage   | 2024-01 → present (monthly files) |

Assets → see [assets.md](assets.md)

---

## Folder & File Structure

```
payamdavaee/candles/
  {asset}/                          ← lowercase asset name
    {asset}-1m-{year}-{month}.parquet
```

**Naming example:** `btcusdt/btcusdt-1m-2026-03.parquet`

**Pattern:** `{asset}-1m-{YYYY}-{MM}.parquet`  (month zero-padded, e.g. `03`)

---

## Column Schema

Every parquet file has 11 columns. One row = one 1-minute candle.

| Column | Type           | Description                              |
|--------|----------------|------------------------------------------|
| `ts`   | timestamp(ms)  | Candle open time (UTC, milliseconds)     |
| `o`    | float64        | Open price                               |
| `h`    | float64        | High price                               |
| `l`    | float64        | Low price                                |
| `c`    | float64        | Close price                              |
| `v`    | float64        | Base asset volume                        |
| `q`    | float64        | Quote asset volume                       |
| `n`    | int32          | Number of trades                         |
| `vwap` | float64        | Volume-weighted average price (`q / v`)  |
| `vb`   | float64        | Taker buy base volume (aggressive buys)  |
| `vs`   | float64        | Taker sell base volume (`v - vb`)        |

> **Aggressive buy volume** = `vb`.  
> **Aggressive sell volume** = `vs` = `v - vb`.

---

## Direct HTTPS URL

```
https://huggingface.co/datasets/payamdavaee/candles/resolve/main/{asset}/{asset}-1m-{YYYY}-{MM}.parquet
```

**Example:**
```
https://huggingface.co/datasets/payamdavaee/candles/resolve/main/btcusdt/btcusdt-1m-2026-03.parquet
```

---

## Access Methods

### 1. `huggingface_hub` — list files

```python
from huggingface_hub import HfApi

api = HfApi()
files = sorted(api.list_repo_files("payamdavaee/candles", repo_type="dataset"))

# Filter for one asset
btc_files = [f for f in files if f.startswith("btcusdt/") and f.endswith(".parquet")]
```

### 2. `requests` + `pyarrow` — read specific file (recommended, works everywhere)

```python
import io, requests
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/payamdavaee/candles/resolve/main/btcusdt/btcusdt-1m-2026-03.parquet"
resp = requests.get(url, timeout=60)
table = pq.read_table(io.BytesIO(resp.content))          # all columns
table = pq.read_table(io.BytesIO(resp.content),
                      columns=["ts", "vwap", "vb"])      # selected columns only
df = table.to_pandas()
```

### 3. `pandas` — read multiple months into one DataFrame

```python
import io, requests
import pandas as pd
import pyarrow.parquet as pq

REPO = "https://huggingface.co/datasets/payamdavaee/candles/resolve/main"
asset = "btcusdt"
months = [("2026", "03"), ("2026", "04")]

frames = []
for year, month in months:
    url = f"{REPO}/{asset}/{asset}-1m-{year}-{month}.parquet"
    resp = requests.get(url, timeout=60)
    frames.append(pq.read_table(io.BytesIO(resp.content)).to_pandas())

df = pd.concat(frames).sort_values("ts").reset_index(drop=True)
```

### 4. DuckDB — native `hf://` access (recommended)

```python
import duckdb

con = duckdb.connect()
asset = "btcusdt"
hf_parquet_path = f"hf://datasets/payamdavaee/candles/{asset}/*.parquet"

result = con.execute(f"""
    SELECT ts, vwap, vb
    FROM read_parquet('{hf_parquet_path}')
    WHERE ts >= epoch_ms('2026-03-01'::TIMESTAMP)
      AND ts <= epoch_ms('2026-03-31 23:59:00'::TIMESTAMP)
    ORDER BY ts ASC
""").fetchdf()
```

### 5. `huggingface_hub` — download file to disk

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="payamdavaee/candles",
    filename="btcusdt/btcusdt-1m-2026-03.parquet",
    repo_type="dataset",
)
# `path` is the local file path
```

---

## Common Patterns

### Get all months for an asset

```python
from huggingface_hub import HfApi
import re

api = HfApi()
files = api.list_repo_files("payamdavaee/candles", repo_type="dataset")

asset = "btcusdt"
pattern = re.compile(rf"^{asset}/{asset}-1m-(\d{{4}})-(\d{{2}})\.parquet$")
months = sorted(
    (m.group(1), m.group(2))
    for f in files
    if (m := pattern.match(f))
)
# months = [("2024","01"), ("2024","02"), ...]
```

### Load a date range (vwap + aggressive buy volume)

```python
import io, requests, re
import pyarrow.parquet as pq, pyarrow as pa
import pandas as pd
from huggingface_hub import HfApi
from datetime import datetime, timezone

REPO = "https://huggingface.co/datasets/payamdavaee/candles/resolve/main"

def load_range(asset: str, start: str, end: str, columns=None) -> pd.DataFrame:
    """
    asset  : lowercase, e.g. "btcusdt"
    start  : "YYYY-MM-DD" inclusive
    end    : "YYYY-MM-DD" inclusive
    columns: list of column names or None for all
    """
    api = HfApi()
    files = api.list_repo_files("payamdavaee/candles", repo_type="dataset")
    pattern = re.compile(rf"^{asset}/{asset}-1m-(\d{{4}})-(\d{{2}})\.parquet$")

    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime.fromisoformat(end  ).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000

    frames = []
    for f in sorted(files):
        m = pattern.match(f)
        if not m:
            continue
        url  = f"{REPO}/{f}"
        resp = requests.get(url, timeout=60)
        tbl  = pq.read_table(io.BytesIO(resp.content), columns=columns or None)
        df   = tbl.to_pandas()
        # ts is stored as timestamp(ms); convert to int for filtering if needed
        ts   = df["ts"].astype("int64")
        df   = df[(ts >= start_ts) & (ts < end_ts)]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("ts").reset_index(drop=True)

# Usage
df = load_range("btcusdt", "2026-03-01", "2026-03-31", columns=["ts", "vwap", "vb"])
```

---

## Notes for Agents

- `ts` is millisecond epoch. Convert: `pd.to_datetime(df["ts"], unit="ms", utc=True)`
- `vb` = aggressive buy volume (taker buy). Use this for buy pressure analysis.
- `vwap` is pre-computed as `q / v`; use it directly, do not recompute.
- Files are complete calendar months. To query a date range, load the relevant monthly files and filter rows by `ts`.
- If a month file is missing for an asset, that month's data does not exist (asset not yet listed on Binance at that time).
- All parquet files are snappy-compressed and column-prunable — always pass `columns=[...]` to `read_table` to reduce download size.
