# Specification: `candle_loader` Package

## Task Summary

Create a new reusable package `packages/candle_loader/` that provides a `load_candles` function. This function fetches 1-minute candle data from the HuggingFace candles datastore using DuckDB's native `hf://` protocol, filters by date range, selects user-specified columns, and returns a 2D NumPy array. A single compact info line is printed on each call.

---

## Background and Context

- Candle data is stored as monthly parquet files in the HuggingFace dataset `payamdavaee/candles`.
- File structure: `{asset}/{asset}-1m-{YYYY}-{MM}.parquet` (asset always lowercase).
- Schema: `ts, o, h, l, c, v, q, n, vwap, vb, vs` — see `agents/datasets/huggingface_candles.md`.
- DuckDB supports `hf://datasets/...` natively with glob/wildcards — no download step or httpfs extension needed.
- The existing `packages/numpy_candles/` package loads candles from Binance CSV. This new package is a separate, independent package using DuckDB + HuggingFace parquet.

---

## Repository Conventions (from `/agents`)

- Reusable packages go in `packages/<package_name>/` (`agents/general/paths_and_files.md`).
- Python packages must have a `requirements.txt` in the same directory (`agents/general/rules.md`).
- Column schema is defined in `agents/datasets/huggingface_candles.md`.
- Valid assets are listed in `agents/datasets/assets.md`.

---

## Functional Requirements

### Function Signature

```python
def load_candles(asset: str, date_from: str, date_to: str, columns: list[str]) -> np.ndarray:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `asset` | `str` | Asset symbol (e.g. `"btcusdt"` or `"BTCUSDT"`). Case-insensitive — the function must normalize to lowercase internally. |
| `date_from` | `str` | Inclusive start datetime as `"YYYY-MM-DD HH:MM:SS"` or `""`. Empty string means from the very first available candle. |
| `date_to` | `str` | Inclusive end datetime as `"YYYY-MM-DD HH:MM:SS"` or `""`. Empty string means until the very last available candle. |
| `columns` | `list[str]` | List of column names to include in the output. Valid names: `o, h, l, c, v, q, n, vwap, vb, vs`. Must not be empty. |

### Return Value

- A 2D `numpy.ndarray` with `dtype=np.float64`.
- **Column 0 is always `ts`** (candle open time in milliseconds, UTC) — implicitly included regardless of the `columns` parameter.
- Columns 1..N correspond to the columns specified in the `columns` parameter, in the order given.
- Example: `load_candles("btcusdt", "2025-12-12 20:00:00", "", ["c", "vwap"])` → returns array with shape `(R, 3)` where col 0 = ts, col 1 = c, col 2 = vwap.

### Data Fetching via DuckDB

Use DuckDB's native `hf://` path with glob to query all parquet files for the asset:

```python
con = duckdb.connect()
hf_parquet_path = f"hf://datasets/payamdavaee/candles/{asset}/*.parquet"
query = f"""
    SELECT ts, {cols_sql}
    FROM read_parquet('{hf_parquet_path}')
    WHERE ts >= epoch_ms('{date_from}'::TIMESTAMP)
      AND ts <= epoch_ms('{date_to}'::TIMESTAMP)
    ORDER BY ts ASC;
"""
```

- When `date_from == ""`: omit the `ts >= ...` condition (no lower bound).
- When `date_to == ""`: omit the `ts <= ...` condition (no upper bound).
- When both are empty: load all available data for the asset.
- Convert the result to a NumPy array (use `fetchnumpy()` and stack into a 2D float64 array).

### Inclusive Boundaries

Both `date_from` and `date_to` are **inclusive** — the exact candle matching the boundary timestamp is included.

### Case Handling

- `asset` must be lowercased before use (user may pass `"BTCUSDT"` or `"btcusdt"`).

### Validation / Error Handling

- If `columns` is an empty list → raise a `ValueError` with a descriptive message.
- If any column name in `columns` is not one of the valid column names (`o, h, l, c, v, q, n, vwap, vb, vs`) → raise a `ValueError`.
- Note: `ts` should NOT be passed in the `columns` list since it is always implicitly included. If a user passes `ts` in the columns list, treat it as an invalid column name (raise error), OR silently ignore it — **preferred behavior: raise a ValueError explaining that ts is always included automatically**.
- No error should be raised for date ranges that partially or fully fall outside available data. The function simply returns whatever data exists within the range. If no data matches, return an empty array with shape `(0, len(columns) + 1)`.

### Info Print Line

After loading, print exactly **one line** to stdout with this compact format:

```
(43200, 3) 2025-12-12T20:00:00 2026-01-12T19:59:00 3.42s [ts:0,c:1,vwap:2]
```

Format breakdown:
- `(rows, cols)` — shape of the output array.
- First datetime — human-readable UTC time of the first row's `ts`.
- Last datetime — human-readable UTC time of the last row's `ts`.
- `X.XXs` — elapsed time for the load operation in seconds, 2 decimal places.
- `[col_name:index,...]` — all column names with their 0-based index.

If the result is empty (0 rows), still print the shape and elapsed time; skip the datetime fields or use a placeholder like `- -`.

---

## File Structure

```
packages/candle_loader/
├── __init__.py          # exports load_candles
└── requirements.txt     # duckdb, numpy
```

### `requirements.txt`

```
duckdb
numpy
```

---

## Additional Task: Update `agents/datasets/huggingface_candles.md`

The current DuckDB example (Section 4) uses a download-first approach with `requests` + `pyarrow`. **Replace or add** a DuckDB section that documents the native `hf://` access method:

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

Update the section title to reflect this is the recommended/native approach. Remove the old download-based DuckDB example.

---

## Non-Goals / Out of Scope

- No caching layer.
- No multi-asset loading in a single call.
- No DataFrame return type — only NumPy array.
- No gap-filling or interpolation — data is guaranteed gap-free by the datastore.
- No progress bar or verbose logging beyond the single info print line.

---

## Assumptions

- DuckDB natively supports `hf://datasets/...` paths without additional extensions or authentication (the dataset is public).
- Data within the datastore is guaranteed to have no gaps (no missing candles between existing monthly files).
- The `ts` column in parquet files is stored as `timestamp(ms)` type.
- The coding agent has access to install and test with `duckdb` and `numpy`.

---

## Acceptance Criteria

1. `packages/candle_loader/__init__.py` exists and exports `load_candles`.
2. `packages/candle_loader/requirements.txt` exists with `duckdb` and `numpy`.
3. `load_candles("BTCUSDT", "2025-12-12 20:00:00", "", ["c", "vwap"])` returns a 2D float64 NumPy array with `ts` as column 0, `c` as column 1, `vwap` as column 2.
4. `load_candles("btcusdt", "", "", ["o"])` loads all available data.
5. `load_candles("btcusdt", "2025-01-01 00:00:00", "2025-01-01 00:00:00", ["c"])` returns exactly 1 row (the candle at that exact timestamp) if it exists.
6. `load_candles("btcusdt", "", "", [])` raises `ValueError`.
7. `load_candles("btcusdt", "", "", ["invalid_col"])` raises `ValueError`.
8. A single compact info line is printed to stdout on every successful call.
9. `agents/datasets/huggingface_candles.md` DuckDB section is updated to show the `hf://` native approach.

---

## Notes for the Downstream Coding Agent

- Use `duckdb.connect()` (in-memory connection) — no persistent database file.
- Use `fetchnumpy()` which returns a dict of column-name → numpy-array. Stack them horizontally into the final 2D array.
- For the `ts` column: DuckDB may return it as a datetime/timestamp type. Convert to float64 milliseconds (epoch ms) to match the existing convention in `packages/numpy_candles/`.
- Time the operation using `time.time()` or `time.perf_counter()` — wrap the DuckDB execute + fetch.
- The `epoch_ms()` function in DuckDB converts a timestamp to milliseconds since epoch. The pattern `epoch_ms('2025-12-12 20:00:00'::TIMESTAMP)` casts the string to a TIMESTAMP then converts to epoch ms for comparison with the `ts` column.
- Keep the implementation simple and readable — no classes needed, just a single function in `__init__.py`.
- Ensure the SQL query is built safely. Since column names come from a validated whitelist, string interpolation is acceptable here (no SQL injection risk from user input after validation).
