# Idea: Normalize Based on Last Price with Clipping

## Purpose

A price normalization scheme used inside **look_back_look_ahead** windows.
Every study that references **normalize_based_on_last_price_clip** follows this specification exactly.

---

## Core Formula

```
scaled = clip( k * (price - price_l) / price_l , clip_min, clip_max )
```

| Symbol      | Definition |
|-------------|------------|
| `price`     | The value being normalized (typically `vwap`; stated per study) |
| `price_l`   | **Base price** — `vwap` (or stated price column) of the **last candle** in the look-back window |
| `k`         | Scale factor (see below) |
| `clip_min`  | Lower clip bound — default `-1` |
| `clip_max`  | Upper clip bound — default `+1` |

> `price_l` is **always** anchored to the rightmost candle of the look-back window.
> It does **not** change between look-back and look-ahead normalization.
> It **does** change from observation to observation as the window slides.

---

## Clipping Modes

Default mode is **hard** unless stated otherwise.

### Hard (default)

```python
scaled = np.clip(k * (price - price_l) / price_l, clip_min, clip_max)
```

Values outside `[clip_min, clip_max]` are clamped to the boundary.

### Tanh

```python
scaled = np.tanh(k * (price - price_l) / price_l)
```

Output is naturally bounded to `(-1, +1)`; `clip_min`/`clip_max` are not applied
unless explicitly stated alongside tanh.

---

## Clipping Range

Default: `[-1, 1]`. Override by stating e.g. `clip to -2 and 2`.

---

## Determining k

### Stated explicitly

> "k = 5" → use `k = 5` directly.

### Derived from a target ratio

> "keep ratio of R" — means raw ratio `R` maps to `1` after scaling:

```
k * R = 1  →  k = 1 / R
```

**Examples:**

| Statement                    | R     | k = 1/R |
|------------------------------|-------|---------|
| keep ratio of 0.20           | 0.20  | 5.0     |
| keep ratio of 0.05           | 0.05  | 20.0    |
| keep ratio of 0.10           | 0.10  | 10.0    |

Interpretation: any candle whose price deviates from `price_l` by exactly `R`
(i.e. `(price - price_l) / price_l == R`) will map to `1.0`; larger deviations clip to `1.0`.

### Derived from a percentile of historical data

> "k such that the P90 of raw ratios maps to 1" — agent must:
> 1. Compute raw ratios `(price - price_l) / price_l` across a representative history.
> 2. Take the stated percentile (e.g. P90) of `abs(raw_ratio)` as `R`.
> 3. Set `k = 1 / R`.

---

## Application Windows

### Look-Back Window (primary / default target)

```python
# lb_vwap : 1-D array of vwap values, shape (look_back,)
# price_l is the LAST element of the look-back window
price_l = lb_vwap[-1]
raw     = (lb_vwap - price_l) / price_l          # shape (look_back,)
scaled  = np.clip(k * raw, clip_min, clip_max)   # hard clip
```

Index `[-1]` = `last_candle.vwap` = `current_price` equivalent for `vwap`.

### Look-Ahead Window

`price_l` is still the **same** `last_candle.vwap` from the look-back window — not reset.

```python
# la_vwap : 1-D array of vwap values, shape (look_ahead,)
price_l = lb_vwap[-1]                             # same anchor as look-back
raw     = (la_vwap - price_l) / price_l
scaled  = np.clip(k * raw, clip_min, clip_max)
```

---

## Vectorized (2-D) Application

When using **vectorized** or **chunked-vectorized** mode from `idea_look_back_look_ahead`:

```python
# lb_2d : shape (n_obs, look_back)  — vwap look-back matrix
# la_2d : shape (n_obs, look_ahead) — vwap look-ahead matrix

price_l = lb_2d[:, -1:]              # shape (n_obs, 1) — rightmost column, kept as column vector

# Look-back normalization
raw_lb     = (lb_2d - price_l) / price_l             # shape (n_obs, look_back)
scaled_lb  = np.clip(k * raw_lb, clip_min, clip_max)

# Look-ahead normalization (same price_l anchor)
raw_la     = (la_2d - price_l) / price_l             # shape (n_obs, look_ahead)
scaled_la  = np.clip(k * raw_la, clip_min, clip_max)
```

> `price_l` must be shaped `(n_obs, 1)` (column vector) to broadcast correctly across columns.

---

## Time Normalization

When **time normalization** is mentioned, the position of each candle within the
look-back window is mapped to a fixed value in `[0, 1)`.

### Definition

```
t[i] = i / look_back       for i = 0, 1, 2, …, look_back - 1
```

| Position        | Normalized time          |
|-----------------|--------------------------|
| First candle    | `0 / look_back = 0.0`    |
| Second candle   | `1 / look_back`          |
| Last candle     | `(look_back-1) / look_back`  e.g. `1439/1440 ≈ 0.9993` |
| `current_time`  | `1.0`  (not a candle — the observation moment after last candle closes) |

> `t = 1.0` is never part of the stored array; it represents `current_time` as defined in
> `idea_look_back_look_ahead.md`.

### Key Properties

- **Fixed vector** — identical for every observation window; compute once and reuse.
- **Independent of actual timestamps** — does not use `ts` values, only window position.
- **Not applied to look-ahead** unless explicitly stated. If applied to look-ahead, continue
  the same scale: `t_la[j] = (look_back + j) / look_back` for `j = 0, …, look_ahead - 1`,
  so the look-ahead candles fall in `[1.0, 1.0 + (look_ahead-1)/look_back]`.
  The first look-ahead candle opens exactly at `current_time` (t = 1.0).

### Code

```python
# Compute once — reuse for all windows
t = np.arange(look_back) / look_back          # shape (look_back,)
# t[0] = 0.0,  t[-1] = (look_back-1)/look_back,  "t=1.0" = current_time

# Look-ahead extension (only if stated)
t_la = (look_back + np.arange(look_ahead)) / look_back  # shape (look_ahead,)
# t_la[0] = look_back / look_back = 1.0  (first LA candle opens at current_time)
# t_la[-1] = (look_back + look_ahead - 1) / look_back
```

### Vectorized (2-D)

The time vector is the same for every observation row — broadcast directly:

```python
# t shape: (look_back,)  →  broadcasts against (n_obs, look_back) without copying
# No per-observation computation needed.
t = np.arange(look_back) / look_back   # computed once, used as-is
```

---

## Inverse Transform (de-normalization)

To recover the original price from a scaled value:

```python
# hard clip inverse (valid only for |scaled| < 1, i.e. not clipped)
price = price_l * (1 + scaled / k)

# or for an entire window
price = price_l * (1 + scaled_window / k)
```

> Clipped values (`±1`) do not have a unique inverse — the original price was `≥ price_l*(1+1/k)`
> or `≤ price_l*(1-1/k)`.

---

## Quick-Reference Cheat Sheet

```
price_l  = lb_window[-1].vwap          # always last candle of look-back, slides each observation
raw      = (price - price_l) / price_l
scaled   = clip( k * raw , -1, 1 )    # hard (default)
         = tanh( k * raw )             # tanh mode

k from ratio R  → k = 1 / R           # e.g. R=0.2 → k=5
k stated        → use directly

look-back target  : price_l = lb[-1]
look-ahead target : price_l = lb[-1]  (same anchor, not re-set)

vectorized:
  price_l = lb_2d[:, -1:]             # shape (n_obs, 1)  ← must be column vector
  scaled_lb = clip(k * (lb_2d - price_l) / price_l, -1, 1)
  scaled_la = clip(k * (la_2d - price_l) / price_l, -1, 1)
```

---

## Relation to Other Ideas

- **look_back_look_ahead** (`idea_look_back_look_ahead.md`): defines `lb_window`, `la_window`, `last_candle`, `price_l`.
- This normalization is applied **after** windows are extracted.
- Column used for `price` defaults to `vwap`; override by stating a different column (e.g. `c`, `h`, `l`).
