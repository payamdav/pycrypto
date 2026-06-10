# Observation Report Viewer (SPA)

A static Single Page Application that renders the master-JSON observation
reports produced by `lookback_lookahead_nn.py` and stored in the public GCS
bucket `gs://payamdpycryptoreports/`.

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
   `https://storage.googleapis.com/payamdpycryptoreports?list-type=2`
   and parse `<Key>` entries ending in `.json`. Pagination is handled via
   `NextContinuationToken`.
2. **`manifest.json` fallback**: if listing fails (e.g. listing not enabled),
   fetch `https://storage.googleapis.com/payamdpycryptoreports/manifest.json`,
   a JSON array of report object paths.
3. **Manual fallback**: upload a report JSON from disk, or paste a report URL.

Individual reports are fetched at
`https://storage.googleapis.com/payamdpycryptoreports/{key}`.

The bucket name lives in `js/config.js` (`APP_CONFIG.BUCKET_NAME`).

## Rendered sections

`metadata`, `model_architecture`, `training_telemetry`,
`evaluation_metrics` (global + per-head), and every entry under
`visualizations` (each a `fig.to_json()` Plotly serialization, drawn with
`Plotly.newPlot`).

## Deploying

Use `../upload_web_app.py` to upload this whole folder to the bucket under the
`app/` prefix, preserving the relative structure with correct content types.
