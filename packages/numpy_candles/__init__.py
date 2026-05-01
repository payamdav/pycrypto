import io
import urllib.request
from types import SimpleNamespace

import numpy as np

# Column indices in the output array
NC = SimpleNamespace(
    ts=0,   # open_time (ms)
    o=1,    # open
    h=2,    # high
    l=3,    # low
    c=4,    # close
    v=5,    # volume
    q=6,    # quote_volume
    n=7,    # count (number of trades)
    vwap=8, # quote_volume / volume
    vb=9,   # taker_buy_volume
    vs=10,  # volume - taker_buy_volume
)

# CSV column indices to load (skip close_time=6, taker_buy_quote_volume=10, ignore=11)
_CSV_COLS = (0, 1, 2, 3, 4, 5, 7, 8, 9)


def load_numpy_candles_from_binance_file(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as response:
            raw = response.read()
        data = io.BytesIO(raw)
    else:
        data = path_or_url

    loaded = np.genfromtxt(
        data,
        delimiter=",",
        skip_header=1,
        usecols=_CSV_COLS,
        dtype=np.float64,
    )

    n_rows = loaded.shape[0]
    out = np.empty((n_rows, 11), dtype=np.float64)

    # ts, o, h, l, c, v, q, n
    out[:, NC.ts] = loaded[:, 0]
    out[:, NC.o]  = loaded[:, 1]
    out[:, NC.h]  = loaded[:, 2]
    out[:, NC.l]  = loaded[:, 3]
    out[:, NC.c]  = loaded[:, 4]
    out[:, NC.v]  = loaded[:, 5]
    out[:, NC.q]  = loaded[:, 6]
    out[:, NC.n]  = loaded[:, 7]

    # vwap = quote_volume / volume
    out[:, NC.vwap] = loaded[:, 6] / loaded[:, 5]

    # vb = taker_buy_volume
    out[:, NC.vb] = loaded[:, 8]

    # vs = volume - taker_buy_volume
    out[:, NC.vs] = loaded[:, 5] - loaded[:, 8]

    return out
