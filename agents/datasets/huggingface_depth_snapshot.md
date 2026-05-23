# HuggingFace Depth Snapshot Dataset

## Identity

| Key        | Value                              |
|------------|------------------------------------|
| Repo ID    | `payamdavaee/depth_snapshot`       |
| Repo type  | `dataset`                          |
| Visibility | public — no token required         |
| Frequency  | ~1 snapshot per minute             |
| Coverage   | daily files per asset              |

Assets → see [assets.md](assets.md)

---

## Folder & File Structure

```
payamdavaee/depth_snapshot/
  {asset}/                                        ← lowercase asset name
    {asset}_depth_snapshot_{YYYY}_{MM}_{DD}.parquet
```

**Naming example:** `btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet`

**Pattern:** `{asset}_depth_snapshot_{YYYY}_{MM}_{DD}.parquet`  (month and day zero-padded, e.g. `05`, `20`)

---

## Column Schema

Every parquet file has 15 columns. One row = one depth snapshot (taken approximately every minute).

| Column   | Type                                      | Description                                                         |
|----------|-------------------------------------------|---------------------------------------------------------------------|
| `ts`     | timestamp(ms)                             | Snapshot time (UTC, milliseconds)                                   |
| `bids`   | list\<struct{price: float64, volume: float64}\> | Bid levels, sorted descending by price (index 0 = best bid)   |
| `asks`   | list\<struct{price: float64, volume: float64}\> | Ask levels, sorted ascending by price (index 0 = best ask)    |
| `bcount` | int32                                     | Number of bid levels                                                |
| `acount` | int32                                     | Number of ask levels                                                |
| `bmin`   | float64                                   | Lowest bid price (worst bid)                                        |
| `bmax`   | float64                                   | Highest bid price (best bid)                                        |
| `amin`   | float64                                   | Lowest ask price (best ask)                                         |
| `amax`   | float64                                   | Highest ask price (worst ask)                                       |
| `arange` | float64                                   | Ask price range (`amax - amin`)                                     |
| `brange` | float64                                   | Bid price range (`bmax - bmin`)                                     |
| `spread` | float64                                   | Spread (`amin - bmax`, best ask − best bid)                         |
| `mid`    | float64                                   | Mid price (`(amin + bmax) / 2`)                                     |
| `av`     | float64                                   | Total ask volume (sum of all ask level volumes)                     |
| `bv`     | float64                                   | Total bid volume (sum of all bid level volumes)                     |

> **`bids` and `asks`** are nested lists of structs. Each struct has `price` (float64) and `volume` (float64).  
> **Best bid** = `bids[0].price` = `bmax`.  
> **Best ask** = `asks[0].price` = `amin`.

---

## Direct HTTPS URL

```
https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main/{asset}/{asset}_depth_snapshot_{YYYY}_{MM}_{DD}.parquet
```

**Example:**
```
https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main/btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet
```

---

## Access Methods

### 1. `huggingface_hub` — list files

```python
from huggingface_hub import HfApi

api = HfApi()
files = sorted(api.list_repo_files("payamdavaee/depth_snapshot", repo_type="dataset"))

# Filter for one asset
btc_files = [f for f in files if f.startswith("btcusdt/") and f.endswith(".parquet")]
```

### 2. `requests` + `pyarrow` — read specific file (recommended, works everywhere)

```python
import io, requests
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main/btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet"
resp = requests.get(url, timeout=120)
table = pq.read_table(io.BytesIO(resp.content))          # all columns
table = pq.read_table(io.BytesIO(resp.content),
                      columns=["ts", "mid", "spread"])   # selected columns only
df = table.to_pandas()
```

### 3. `pandas` — read multiple days into one DataFrame

```python
import io, requests
import pandas as pd
import pyarrow.parquet as pq

REPO = "https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main"
asset = "btcusdt"
days = [("2026", "05", "18"), ("2026", "05", "19"), ("2026", "05", "20")]

frames = []
for year, month, day in days:
    url = f"{REPO}/{asset}/{asset}_depth_snapshot_{year}_{month}_{day}.parquet"
    resp = requests.get(url, timeout=120)
    frames.append(pq.read_table(io.BytesIO(resp.content)).to_pandas())

df = pd.concat(frames).sort_values("ts").reset_index(drop=True)
```

### 4. `huggingface_hub` — download file to disk

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="payamdavaee/depth_snapshot",
    filename="btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet",
    repo_type="dataset",
)
# `path` is the local file path
```

---

## Common Patterns

### Get all days for an asset

```python
from huggingface_hub import HfApi
import re

api = HfApi()
files = api.list_repo_files("payamdavaee/depth_snapshot", repo_type="dataset")

asset = "btcusdt"
pattern = re.compile(rf"^{asset}/{asset}_depth_snapshot_(\d{{4}})_(\d{{2}})_(\d{{2}})\.parquet$")
days = sorted(
    (m.group(1), m.group(2), m.group(3))
    for f in files
    if (m := pattern.match(f))
)
# days = [("2026","05","18"), ("2026","05","19"), ...]
```

### Load a date range (scalar columns only, excluding nested bids/asks)

```python
import io, requests, re
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfApi
from datetime import date, timedelta

REPO = "https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main"

def load_range(asset: str, start: str, end: str, columns=None) -> pd.DataFrame:
    """
    asset  : lowercase, e.g. "btcusdt"
    start  : "YYYY-MM-DD" inclusive
    end    : "YYYY-MM-DD" inclusive
    columns: list of column names or None for all
    """
    api = HfApi()
    files = api.list_repo_files("payamdavaee/depth_snapshot", repo_type="dataset")
    pattern = re.compile(rf"^{asset}/{asset}_depth_snapshot_(\d{{4}})_(\d{{2}})_(\d{{2}})\.parquet$")

    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    frames = []
    for f in sorted(files):
        m = pattern.match(f)
        if not m:
            continue
        file_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if file_date < start_date or file_date > end_date:
            continue
        url  = f"{REPO}/{f}"
        resp = requests.get(url, timeout=120)
        tbl  = pq.read_table(io.BytesIO(resp.content), columns=columns or None)
        frames.append(tbl.to_pandas())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("ts").reset_index(drop=True)

# Usage — load scalar summary columns for a week
df = load_range("btcusdt", "2026-05-14", "2026-05-20",
                columns=["ts", "mid", "spread", "av", "bv"])
```

### Access nested bid/ask levels

```python
import io, requests
import pyarrow.parquet as pq
import pandas as pd

url = "https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/main/btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet"
resp = requests.get(url, timeout=120)
df = pq.read_table(io.BytesIO(resp.content), columns=["ts", "bids", "asks"]).to_pandas()

# Each row's 'bids' and 'asks' are lists of dicts: [{"price": ..., "volume": ...}, ...]
# Example: get best bid/ask from first snapshot
first_row = df.iloc[0]
best_bid = first_row["bids"][0]    # {"price": 107500.0, "volume": 1.23}
best_ask = first_row["asks"][0]    # {"price": 107501.0, "volume": 0.87}
```

---

## Notes for Agents

- `ts` is millisecond epoch. Convert: `pd.to_datetime(df["ts"], unit="ms", utc=True)`
- Each file contains one full calendar day of snapshots (UTC boundaries).
- Snapshots are taken approximately every minute (~1440 rows per day).
- `bids` are sorted descending by price; `asks` are sorted ascending by price.
- `spread` = best ask (`amin`) − best bid (`bmax`). A tighter spread indicates higher liquidity.
- `mid` = midpoint price, useful as a reference price when candle data is not needed.
- `av` and `bv` represent total visible liquidity on each side of the order book.
- If a daily file is missing for an asset, that day's data was not captured.
- Files can be large due to nested bid/ask lists. Always pass `columns=[...]` to `read_table` to reduce download size — use scalar summary columns (`mid`, `spread`, `av`, `bv`, etc.) when full order book depth is not needed.
- The nested `bids`/`asks` columns contain the full order book snapshot. Use them when you need price-level granularity (e.g., liquidity heatmaps, depth profiles).
