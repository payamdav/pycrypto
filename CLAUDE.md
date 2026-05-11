# Coding Agent Instructions

Before writing any code, read all files listed below. They contain the authoritative instructions, naming conventions, dataset schemas, and idea specifications that every implementation in this repository must follow.

## Required Reading

### Datasets

- @agents/datasets/assets.md — canonical list of supported crypto assets and their lowercase folder names
- @agents/datasets/huggingface_candles.md — HuggingFace candles dataset identity, column schema, URL pattern, access methods, and the `load_range()` helper

### Ideas

- @agents/ideas/idea_look_back_look_ahead.md — look-back / look-ahead windowing pattern: parameters, conventions (`last_candle`, `current_time`, `price_l`), inclusive vs exclusive date boundaries, loop / vectorized / chunked-vectorized modes, historic indicator boundary extension
- @agents/ideas/idea_normalize_based_on_last_price_clip.md — price and time normalization inside look-back / look-ahead windows: core formula, hard vs tanh clipping, k derivation, vectorized 2-D application, time normalization vector, inverse transform
