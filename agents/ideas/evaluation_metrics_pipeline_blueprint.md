# Evaluation & Metrics Pipeline Blueprint

This document defines the standard vocabulary and operational requirements for the evaluation and reporting modules. These definitions are designed as strict instructions for the coding agent to execute specific analytical blocks on a single-target regression model where the label space is bounded strictly between `[-1.0, 1.0]`.

---

## 1. Core Model Performance Metrics (`Eval_`)

These modules measure the foundational accuracy, convergence health, and predictive distribution of the trained network.

### `Eval_Convergence_Plot`
* **Description:** A diagnostic line chart visualizing the learning health of the network during the training phase.
* **Execution:** Plots the **Training Loss** and **Validation Loss** across the y-axis against the number of **Epochs** on the x-axis. 
* **Purpose:** To explicitly identify the exact epoch where the model stops generalizing and begins memorizing noise (overfitting), indicated by the training curve continuing to drop while the validation curve flattens or rises.

### `Eval_Tabular_Error_Metrics`
* **Description:** A clean, terminal-friendly text block that reports the absolute out-of-sample performance of the model on the blind Test Set.
* **Execution:** Calculates and prints the overall Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss for the single target label. 
* **Purpose:** Provides a rapid, numeric baseline to compare different model architectures (e.g., comparing a Dense baseline against an advanced LSTM) without needing to interpret charts.

### `Eval_Certainty_Distribution`
* **Description:** A histogram detailing the frequency of the model's continuous predictions across the `[-1.0, 1.0]` spectrum.
* **Execution:** Plots the density of the validation or test predictions. The x-axis spans `-1.0` to `1.0`, and the y-axis shows the volume of predictions falling into those zones.
* **Purpose:** Diagnoses model confidence. A healthy model prioritizing extreme moves should show peaks near the boundaries (e.g., `0.7` to `1.0`). A failing or "uncertain" model will exhibit a massive cluster of predictions dead-centered around `0.0`, indicating it has learned to just guess the mean to minimize loss safely.

---

## 2. Advanced Error & Boundary Analytics (`Eval_`)

These modules provide deep-dive diagnostics into how the model behaves across specific label ranges, translating raw loss numbers into practical operational intelligence.

### `Eval_Error_Distribution_Curve`
* **Description:** A line or bar chart that isolates how much the model struggles at different absolute market states.
* **Execution:** Slices the continuous `[-1.0, 1.0]` space into 21 uniform bins (step size `0.1`). It maps the *actual* labels to these bins, calculates the Mean Absolute Error (MAE) specifically for the samples within each bin, and plots the resulting error rate across the x-axis.
* **Purpose:** Global MAE is often misleading. This plot reveals if the model is highly accurate during flat, noisy periods (near `0.0`) but completely fails to predict massive structural moves (near `0.9`), or vice versa. 

### `Eval_Bucketed_Confusion_Matrix`
* **Description:** A tabular grid adapting the classic classification confusion matrix for bounded continuous regression.
* **Execution:** Rounds both the true labels and the model's predictions to the nearest `0.1` increment, creating 21 discrete buckets from `-1.0` to `1.0`. It generates a 21x21 data table showing the raw integer counts of how predictions map to reality (e.g., "Out of 1000 instances where the model predicted `0.9`, 100 were actually `0.9`, 200 were `0.8`, and 5 were `0.0`").
* **Purpose:** Allows absolute tracking of False Positives mapped to their exact severity.

### `Eval_Bucketed_Confusion_Rates`
* **Description:** The percentage-normalized variant of the Bucketed Confusion Matrix.
* **Execution:** Replaces the raw integer counts in the 21x21 table with percentage values based on the total number of samples in that predicted row. 
* **Purpose:** Standardizes the matrix so you can quickly read structural probabilities (e.g., "When the model outputs a `0.8`, there is an `85%` chance the true label is $> 0.6$").

### `Eval_Prediction_Heatmap`
* **Description:** The visual counterpart to the Bucketed Confusion Matrix, utilizing color density to instantly highlight the model's prediction accuracy and bias.
* **Execution:** Generates a 2D seaborn/matplotlib colored heatmap of the 21x21 grid. The x-axis is the "True Label Bin", the y-axis is the "Predicted Label Bin". Cells where the predicted bin matches the true bin (the diagonal line running from bottom-left to top-right) should glow brightest. 
* **Purpose:** A perfect model yields a bright, perfectly straight diagonal line. A smeared or clustered heatmap instantly reveals systemic bias (e.g., if the model systematically under-predicts positive moves, the glow will sit below the diagonal axis).

---

## 3. Time-Series & Trading Specific Metrics (`Eval_`)

These modules treat the data as a chronological sequence of market events to measure real-world directional edge.

### `Eval_Directional_Accuracy`
* **Description:** A metric that ignores the absolute value of the error and focuses strictly on whether the model got the *direction* (sign) of the market right.
* **Execution:** Scans all predictions and true labels. If the True Label is positive (e.g., `0.5`) and the Prediction is positive (e.g., `0.1`), it counts as a Win, even though the MAE is large. It outputs the percentage of times the model correctly predicted the sign (`+` or `-`).
* **Purpose:** In trading, predicting `0.2` when the reality is `0.9` has a terrible MSE, but it correctly identifies an upward regime. Conversely, predicting `0.1` when reality is `-0.1` has a tiny MSE, but causes a losing trade. This metric reveals the actual directional edge.

### `Eval_Rolling_Temporal_Error`
* **Description:** A time-series line chart showing how the model's error changes chronologically over the dataset.
* **Execution:** Calculates the Mean Absolute Error (MAE) on a rolling window (e.g., every 1,440 minutes / 1 day) across the validation/test set and plots it as a continuous line over time.
* **Purpose:** Market regimes change (bull, bear, sideways). Global metrics hide this. This plot reveals if the model is perfectly accurate for 6 months, loses its edge during a 1-month high-volatility period, and then recovers. It highlights *when* the model fails.

### `Eval_Feature_Attribution_Map` (Advanced)
* **Description:** An interpretability heatmap that explains *why* the model made a specific prediction.
* **Execution:** Uses an algorithm like Integrated Gradients or SHAP on the input tensor `(time_steps, features)`. For a specific high-confidence prediction, it generates a 2D color map showing exactly which of the historical timesteps and which specific features pushed the model to output a high value.
* **Purpose:** Crucial for trust and debugging. If new contextual features or sidecars are injected into the architecture, this map will immediately prove whether the model is actually utilizing them or ignoring them in favor of raw price data.

---

## 4. Operational & Telemetry Reporting (`Report_`)

These modules summarize the physical compute cost and complexity required to achieve the analytical results.

### `Report_Training_Telemetry`
* **Description:** A comprehensive text or markdown block summarizing the hardware, time, and scale footprint of the training run.
* **Execution:** Automatically records and outputs a tabular list containing:
  * Total trainable parameters of the active model.
  * Total rows of data processed (Train / Val / Test splits).
  * Sequence length and feature count per sample.
  * Mini-batch size utilized.
  * Total epochs completed before Early Stopping triggered.
  * Total wall-clock time required for convergence.
  * Hardware utilized (e.g., `CPU`, `CUDA: Tesla T4`).
* **Purpose:** Crucial for architectural ablation. If a lighter model achieves a similar Huber Loss to a deep model but trains in half the time with fewer parameters, this telemetry report provides the exact metrics needed to justify switching for production efficiency.