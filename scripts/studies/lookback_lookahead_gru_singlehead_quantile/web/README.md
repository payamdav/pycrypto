# Observation Report Viewer (SPA)

A static Single Page Application that renders the master-JSON observation
reports produced by `lookback_lookahead_nn.py` and stored in the public GCS
bucket `gs://payamdpycryptoreports/`.

## Per-study bucket layout

The bucket hosts MULTIPLE studies. Each study is self-contained under its own
parent folder whose name equals the study's `OBSERVATION_SET_NAME`. Each study
folder holds both its reports AND its own copy of this viewer:

```
gs://payamdpycryptoreports/{OBSERVATION_SET_NAME}/
    reports/   <- *.json reports AND *.pt model params
    app/       <- this viewer SPA (index.html, css/, js/, ...)
```

For the current study `OBSERVATION_SET_NAME = "lookback_lookahead_gru_singlehead_quantile"`,
so the viewer is served at:

```
https://storage.googleapis.com/payamdpycryptoreports/lookback_lookahead_gru_singlehead_quantile/app/index.html
```

## Self-locating viewer

The viewer discovers its own bucket and study folder from `window.location`
(see `js/config.js` → `deriveContext()`). When served from
`storage.googleapis.com` it splits the pathname into
`[bucket, ...studyPath..., "app", "index.html"]`, then:

- `BUCKET_NAME` = the first path segment.
- `STUDY_PREFIX` = everything between the bucket and the `app` directory
  (supports multi-segment study paths).
- `REPORT_PREFIXES` = `["{STUDY_PREFIX}/reports/"]`, so the listing returns only
  this study's reports and naturally ignores the sibling `app/` files.

Manual overrides (for local / `file://` testing) take precedence over the
derived values, in this order:

1. Hardcoded `OVERRIDE_BUCKET` / `OVERRIDE_STUDY` constants in `config.js`.
2. `?bucket=...` / `?study=...` query params.
3. Derived-from-path values.
4. Hardcoded fallbacks (`payamdpycryptoreports` /
   `lookback_lookahead_gru_singlehead_quantile`).

## Structure

```
web/
  index.html        entry point (loads CSS + JS, Plotly via CDN)
  css/
    styles.css      all styling (dark, responsive, card-based layout)
  js/
    config.js       bucket name + endpoints + manifest fallback path
    gcs.js          report discovery (XML listing + manifest fallback) + fetch
    render.js       builds DOM from a parsed report + renders Plotly figures
    app.js          SPA controller: dropdown, prev/next, upload, URL paste
  assets/           static assets (currently empty)
```

## How reports are discovered

1. **Bucket XML listing** (primary): fetch
   `https://storage.googleapis.com/{bucket}?list-type=2&prefix={STUDY_PREFIX}/reports/`
   and parse `<Key>` entries ending in `.json`. The prefix scopes the listing to
   this study only. Pagination is handled via `NextContinuationToken`.
2. **`manifest.json` fallback**: if listing fails (e.g. listing not enabled),
   fetch
   `https://storage.googleapis.com/{bucket}/{STUDY_PREFIX}/reports/manifest.json`,
   a JSON array of report object paths.
3. **Manual fallback**: upload a report JSON from disk, or paste a report URL.

Individual reports are fetched at
`https://storage.googleapis.com/{bucket}/{key}`.

The bucket and study are derived automatically (see "Self-locating viewer"),
with manual override hooks in `js/config.js`.

## Rendered sections

`metadata`, `model_architecture`, `training_telemetry`,
`evaluation_metrics` (global + per-head), and every entry under
`visualizations` (each a `fig.to_json()` Plotly serialization, drawn with
`Plotly.newPlot`).

## Deploying

Use `../upload_web_app.py` to upload this whole folder under the study's
`app/` folder, preserving the relative structure with correct content types.
Run from the repo root:

```
python scripts/studies/upload_web_app.py
```

This uploads to `{OBSERVATION_SET_NAME}/app/` (default
`lookback_lookahead_gru_singlehead_quantile/app/`). The `WEB_DIR` is resolved from the
script's own location (`__file__`), so the command works from any working
directory. To deploy a different study's viewer, override the prefix:

```
python scripts/studies/upload_web_app.py --prefix OtherStudy/app
```

`--bucket`, `--web-dir`, and `--dry-run` are also overridable.
