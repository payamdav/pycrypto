# Create a notebook for lstm neural network training and study the result
create a notebook at /notebooks/studies/lookback_lookahead_features_labels/lookback_lookahead_model.ipynb

running environment would be google colab or kaggle
use GPUs if available
during training there is a eed to sometimes have a report including timing, number of batches and epochs and model performance params in a single line
to become familiar with how data prepared and where stored look at all files in /notebooks/studies/lookback_lookahead_features_labels/
also take a look at /packages/asset_analyzer/asset_snapshot_lookback_lookahead_normalize.py to be familiar with features and labels and how they store in file
at one of top cells after importing required packages define an ASSET variable defaults to "btcusdt" ( other assets ready but commented ) and based on that load "fl_data_{ASSET}" file
the features are f_e_vwap and f_e_n_imbalances that becomes on a shape of (24, 2)
labels are l_e_vwap and l_e_n_imbalances that become a vector with length 8

between training data and validation and test data there must be at least 1440 * 7 item distant
use different cells for different tasks in a way that I easily can change asset, retrain or change some parameters then continue.
put a way to save the trained model on the bucket but comment that in its cell
I may need to change the percent for training and validation and test so define a variable for that in flexible way
read the following carefully and do the job
 

# Technical Specification: Multi-Output LSTM Regression Model for Market Regime Prediction

## 1. Dataset & Tensor Dimensions

* **Total Dataset Size:** $\approx 1,200,000$ rows (minute-level candles spanning 29 months).
* **Input Tensor Shape:** `(batch_size, 24, 2)`
* *Sequence Length ($T$):* 24 timesteps (representing engineered rolling 24-hour lookback data).
* *Features ($F$):* 2 continuous features, fully scaled and normalized between $-1$ and $1$.


* **Target Tensor Shape:** `(batch_size, 8)`
* *Labels ($L$):* 8 distinct, continuous target horizons/metrics, fully scaled and normalized between $-1$ and $1$.


* **Data Split Strategy:** Chronological Walk-Forward validation (e.g., 70% Train, 15% Validation, 15% Test) to prevent lookahead bias. Purge overlapping boundary samples between splits.

---

## 2. Model Architecture & Hyperparameters

Because financial time-series data has a low signal-to-noise ratio, the model must remain lean to avoid severe overfitting. A massive model will simply memorize noise.

```
Input Vector (24, 2) 
      │
      ▼
LSTM Layer 1 (64 hidden units, returns sequences) ──► Dropout (0.3)
      │
      ▼
LSTM Layer 2 (64 hidden units, returns final state) ──► Dropout (0.3)
      │
      ▼
Dense Output Layer (8 linear units) ──► TanH Activation ──► Output Vector (8,)

```

### Components:

* **Layer 1:** LSTM layer with **64 hidden units**, configured to `return_sequences=True`.
* **Dropout 1:** Spatial or standard Dropout rate of **0.3** applied immediately after Layer 1.
* **Layer 2:** LSTM layer with **64 hidden units**, configured to `return_sequences=False` (extracting only the final hidden state vector of the sequence).
* **Dropout 2:** Dropout rate of **0.3** applied after Layer 2.
* **Output Head:** A single Dense (Linear) layer mapping from 64 hidden units to **8 output units** (representing the 8 multi-head targets simultaneously).
* **Activation Function:** **$TanH$ (Hyperbolic Tangent)** applied directly to the final output layer. Because $TanH$ is mathematically bounded strictly between $-1$ and $1$, it naturally mirrors your target distribution and stabilizes backpropagation gradient scaling.

---

## 3. Loss Function & Optimization Strategy

To fulfill the operational goal of building a "sniper" model that prioritizes eliminating False Positives for high-valued targets ($> 0.7$), the optimization suite must be tuned as follows:

* **Primary Loss Function:** **Huber Loss (Smooth L1 Loss)**. Financial datasets are prone to extreme anomalies and sudden volatility shifts. Huber loss acts like Mean Squared Error (MSE) for small errors, but scales linearly like Mean Absolute Error (MAE) for massive outlier errors. This prevents a single flash-crash anomaly from corrupting the model's entire gradient history.
* **Optimizer:** **AdamW** with an initial learning rate of $1\times10^{-3}$ and a weight decay of $1\times10^{-4}$ to enforce L2 regularization on the LSTM weights.
* **Learning Rate Scheduler:** `ReduceLROnPlateau` monitoring the validation loss, dropping the learning rate by a factor of 0.5 if performance plateaus for 3 epochs.
* **Batch Size:** **1024 or 2048**. Large batch sizes are highly computationally efficient on modern cloud accelerators (Kaggle/Colab T4 or P100 GPUs) and provide smoother gradient updates over a 1.2-million-row sequence canvas.

---

## 4. Inference & Thresholding Logic Requirements

The notebook must include a post-training validation segment dedicated to operational execution mapping:

1. **Continuous Evaluation:** The model outputs continuous predictions $\hat{y} \in [-1, 1]$ across all 8 heads.
2. **Precision-Driven Thresholding:** The notebook must scan inference execution thresholds ($\tau$) on the validation split from $0.50$ to $0.95$ in steps of $0.01$.
3. **Execution Rule:** A trade signal is triggered for head $i$ if and only if $\hat{y}_i \ge \tau$.
4. **Metric Optimization:** The agent must plot a Precision-Recall curve specifically evaluating the chosen threshold $\tau$ to isolate the point where the False Positive rate drops to your exact acceptable risk tolerance, completely disregarding total trade volume (Recall) in favor of predictive certainty.