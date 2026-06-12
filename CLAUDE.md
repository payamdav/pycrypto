# Coding Agent Instructions

Before writing any code, read all files listed below. They contain the authoritative instructions, naming conventions, dataset schemas, and idea specifications that every implementation in this repository must follow.

## Required Reading

### General

- @agents/general/paths_and_files.md — repository folder structure, file placement rules, and decision logic for where to create new files
- @agents/general/rules.md — mandatory rules for all agents: dependency management, notebook pip installs, requirements.txt, repository cloning in notebooks
- @agents/general/indicators.md — indicators package: available functions (ma, wma, vwma, rsi_1_1, stddev, rolling_robust_z_score), signatures, behavior, and usage examples
- @agents/general/nn_training_performance.md — neural-net training performance techniques: free speedups (GPU-resident data, fewer host↔device syncs, cudnn.benchmark, CUDA-stream concurrency for many small models, separate load/compute timing) vs result-changing choices (batch size, seeds, architecture, optimizer); apply whenever training/evaluating nets without altering the learning result

### Datasets

- @agents/datasets/assets.md — canonical list of supported crypto assets and their lowercase folder names
- @agents/datasets/huggingface_candles.md — HuggingFace candles dataset identity, column schema, URL pattern, access methods, and the `load_range()` helper
- @agents/datasets/huggingface_depth_snapshot.md — HuggingFace depth snapshot dataset identity, folder/file structure, Parquet schema, access methods, and the `load_range()` helper

### Packages

- @agents/packages/gcs_tools.md — GCS tools package (`packages/tools/google_cloud_storage_tools/`): exported functions, setup/import instructions, credential resolution (Colab/Kaggle/RunPod/local), and usage examples
- @agents/packages/runpod_tools.md — RunPod tools package (`packages/tools/runpod_tools/`): environment detection (`is_running_environment_runpod`), secret retrieval (`get_secret`), and pod self-termination (`pod_self_terminate`); RunPod env-var and secret conventions

### Ideas

- @agents/ideas/idea_look_back_look_ahead.md — look-back / look-ahead windowing pattern: parameters, conventions (`last_candle`, `current_time`, `price_l`), inclusive vs exclusive date boundaries, loop / vectorized / chunked-vectorized modes, historic indicator boundary extension
- @agents/ideas/idea_normalize_based_on_last_price_clip.md — price and time normalization inside look-back / look-ahead windows: core formula, hard vs tanh clipping, k derivation, vectorized 2-D application, time normalization vector, inverse transform
- @agents/ideas/anchored_expanding_window.md — anchored expanding window pattern: left/right anchor, step, count, operation; expanding-window + running-accumulator logic, output ordering per anchor, numba implementation guidance
- @agents/ideas/evaluation_metrics_pipeline_blueprint.md — standard evaluation and metrics pipeline: all `Eval_` and `Report_` modules (convergence plots, tabular error metrics, certainty distribution, error distribution curve, bucketed confusion matrix/rates, prediction heatmap, directional accuracy, rolling temporal error, feature attribution map, training telemetry); **must be read and implemented whenever a task involves training, evaluating, or reporting on a model**
- @agents/ideas/headless_observation_reporting.md — report packaging and GCS export protocol: master JSON schema (metadata, model architecture, training telemetry, evaluation metrics, Plotly visualizations), GCS bucket hierarchy (`gs://payamdpycryptoreports/{Observation_Set_Name}/`), file naming convention, Plotly-only visualization requirement, and the `generate_and_upload_report()` reference implementation; **must be read and followed whenever a training run produces an observation that needs to be stored or reported**
