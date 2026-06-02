import time
from datetime import datetime, timezone

import duckdb
import numpy as np

_VALID_COLUMNS = {"o", "h", "l", "c", "v", "q", "n", "vwap", "vb", "vs"}


def load_candles(asset: str, date_from: str, date_to: str, columns: list[str]) -> np.ndarray:
    if not columns:
        raise ValueError("columns must not be empty")
    invalid = [c for c in columns if c not in _VALID_COLUMNS]
    if invalid:
        if "ts" in invalid:
            raise ValueError(
                "ts must not be listed in columns — it is always included automatically as column 0"
            )
        raise ValueError(f"Invalid column name(s): {invalid}. Valid names: {sorted(_VALID_COLUMNS)}")

    asset = asset.lower()
    cols_sql = ", ".join(columns)
    hf_parquet_path = f"hf://datasets/payamdavaee/candles/{asset}/*.parquet"

    conditions = []
    if date_from:
        conditions.append(f"ts >= '{date_from}'::TIMESTAMP")
    if date_to:
        conditions.append(f"ts <= '{date_to}'::TIMESTAMP")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT ts, {cols_sql}
        FROM read_parquet('{hf_parquet_path}')
        {where_clause}
        ORDER BY ts ASC;
    """

    t0 = time.perf_counter()
    con = duckdb.connect()
    result = con.execute(query).fetchnumpy()
    elapsed = time.perf_counter() - t0

    n_rows = len(result["ts"])
    n_cols = 1 + len(columns)
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    ts_raw = result["ts"]
    # DuckDB may return ts as numpy datetime64[ms]; convert to float64 epoch ms
    if ts_raw.dtype.kind == "M":
        out[:, 0] = ts_raw.astype("datetime64[ms]").astype(np.float64)
    else:
        out[:, 0] = ts_raw.astype(np.float64)

    for i, col in enumerate(columns, start=1):
        out[:, i] = result[col].astype(np.float64)

    shape = out.shape
    col_labels = ["ts"] + list(columns)
    col_index = ",".join(f"{name}:{i}" for i, name in enumerate(col_labels))

    if n_rows > 0:
        first_dt = datetime.fromtimestamp(out[0, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        last_dt = datetime.fromtimestamp(out[-1, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"{shape} {first_dt} {last_dt} {elapsed:.2f}s [{col_index}]")
    else:
        print(f"{shape} - - {elapsed:.2f}s [{col_index}]")

    return out
