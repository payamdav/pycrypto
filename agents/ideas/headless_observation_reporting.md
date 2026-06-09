# Specification: Headless Observation Reporting & Telemetry Export

This document provides explicit instructions for coding agents executing automated machine learning training runs. The objective is to package every training run ("Observation") into a single, comprehensive JSON file containing all metadata, metrics, and serialized interactive Plotly figures. This file must be uploaded to a hierarchical Google Cloud Storage (GCS) bucket to be consumed later by a Single Page Application (SPA) dashboard.

---

## 1. Cloud Storage Architecture & Routing

All reports must be pushed to the designated public GCS bucket using a strict hierarchical folder structure. 

* **Target Bucket:** `gs://payamdpycryptoreports`
* **Hierarchy Rule:** Reports must be grouped into folders based on their `Observation_Set_Name`.
* **File Naming Convention:** `{Observation_Set_Name}_{Observation_Name}.json`
* **Absolute Path Format:** `gs://payamdpycryptoreports/{Observation_Set_Name}/{Observation_Set_Name}_{Observation_Name}.json`

*Example:* If running a batch over different assets called "Q3_Asset_Sweep", and the current run is for Bitcoin, the path is:
`gs://payamdpycryptoreports/Q3_Asset_Sweep/Q3_Asset_Sweep_BTC_Base_LSTM.json`

---

## 2. Visualization Protocol (Strictly Plotly)

The SPA dashboard requires interactive charting. **The coding agent must NOT use `matplotlib` or `seaborn` for the final report generation.** * All charts (`Eval_Convergence_Plot`, `Eval_Precision_Threshold_Sweep`, `Eval_Certainty_Distribution`, etc.) must be generated using `plotly.graph_objects` or `plotly.express`.
* Once the Plotly figure object (`fig`) is created, the agent must serialize it to a JSON string using:
  `chart_json = fig.to_json()`
* These serialized strings will be embedded directly into the master JSON payload.

---

## 3. The Master JSON Schema

The coding agent must construct a Python dictionary matching the following structural schema before converting it to JSON and uploading. No top-level keys should be omitted.

### Schema Blueprint:
```json
{
  "metadata": {
    "observation_set_name": "String (e.g., 'Asset_Sweep_v1')",
    "observation_name": "String (e.g., 'BTC_LSTM_2_64')",
    "execution_timestamp": "ISO 8601 UTC Datetime",
    "asset_identifier": "String (e.g., 'BTCUSDT')",
    "data_date_range": {
      "start": "YYYY-MM-DD HH:MM",
      "end": "YYYY-MM-DD HH:MM"
    }
  },
  "model_architecture": {
    "model_type": "String (e.g., 'Model_LSTM_Base_2_64')",
    "input_features": ["List", "of", "Feature", "Names"],
    "sequence_length": "Integer (e.g., 24)",
    "target_labels": ["List", "of", "Label", "Names"]
  },
  "training_telemetry": {
    "total_parameters": "Integer",
    "epochs_completed": "Integer",
    "batch_size": "Integer",
    "hardware_utilized": "String (e.g., 'CUDA: Tesla T4')",
    "total_train_time_seconds": "Float"
  },
  "evaluation_metrics": {
    "global_metrics": {
      "test_huber_loss": "Float",
      "test_mse": "Float"
    },
    "per_head_metrics": {
      "target_1": {
        "mae": "Float",
        "mse": "Float",
        "directional_accuracy_pct": "Float"
      }
      // ... repeated for all target heads
    }
  },
  "visualizations": {
    "Eval_Convergence_Plot": "JSON String (Plotly serialization)",
    "Eval_Precision_Threshold_Sweep": "JSON String (Plotly serialization)",
    "Eval_Certainty_Distribution": "JSON String (Plotly serialization)",
    "Eval_Error_Distribution_Curve": "JSON String (Plotly serialization)",
    "Eval_Prediction_Heatmap": "JSON String (Plotly serialization)"
  }
}



# ==========================================
# FINAL REPORT GENERATION & GCS EXPORT
# ==========================================
import json
import datetime
from google.cloud import storage
import plotly.graph_objects as go

def generate_and_upload_report(observation_set, observation_name, metrics_dict, plotly_figs_dict, metadata_dict):
    # 1. Construct the Master Dictionary
    master_report = {
        "metadata": {
            "observation_set_name": observation_set,
            "observation_name": observation_name,
            "execution_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            **metadata_dict # Unpack asset, dates, features, telemetry, etc.
        },
        "evaluation_metrics": metrics_dict,
        "visualizations": {}
    }

    # 2. Serialize Plotly Figures
    for fig_name, fig_obj in plotly_figs_dict.items():
        master_report["visualizations"][fig_name] = json.loads(fig_obj.to_json())

    # 3. Convert to formatted JSON string
    json_payload = json.dumps(master_report, indent=2)
    
    # 4. Upload to GCS Hierarchical Structure
    bucket_name = "payamdpycryptoreports"
    file_path = f"{observation_set}/{observation_set}_{observation_name}.json"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    # Upload from memory as application/json
    blob.upload_from_string(json_payload, content_type="application/json")
    
    print(f"Successfully exported observation report to: gs://{bucket_name}/{file_path}")

# Agent Execution: Call this function at the absolute end of the pipeline.