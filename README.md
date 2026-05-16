
# Spectral-Bridge
### Transformer-Based Signal Reconstruction using In-Context Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/Architecture-Transformers-orange)
![SignalProcessing](https://img.shields.io/badge/Domain-SignalProcessing-success)
![Competition](https://img.shields.io/badge/IITRoorkee-3rdPlace-purple)

🏆 Developed for the Spectral Bridge Challenge at Cognizance, IIT Roorkee  
🥉 Achieved 3rd Place in the competition

---

# Overview

This project focuses on reconstructing missing portions of audio signals using sparse observed waveform points.

The challenge involves predicting missing target points from limited context observations while preserving the underlying spectral structure of the signal.

The proposed solution combines:

- Transformer-based neural architectures
- Gaussian Process interpolation priors
- Neural Process representations
- Context-aware normalization
- In-context learning strategies

The framework is designed to reconstruct sparse waveforms accurately under limited observation settings.

---

# Problem Description

Each audio sample consists of:

- 100 time steps
- 20 observed context points
- 80 missing target points

The model must infer the missing waveform values using only the provided context points.

Important constraints:

- Each sample is independent
- No information can be transferred between samples
- Prediction must rely only on the sample-specific context observations

---

# Dataset

Dataset used:

**Spectral Graffiti Dataset**  
https://www.kaggle.com/datasets/fda137/spectral-graffiti/data

Dataset properties:

- ~80,000 samples
- 100 time points per sample
- 20 context points
- 80 target points
- Sampling rate: 1kHz

---

# Methodology

The solution follows a hybrid reconstruction pipeline combining classical signal interpolation and deep neural learning.

## 1. Gaussian Process Features

Multiple Gaussian Process predictions with different kernel scales are computed to capture signal behavior across multiple frequencies.

These GP priors help the model estimate waveform continuity and smoothness.

---

## 2. Neural Process Architecture

A transformer-based architecture processes the context points and predicts missing targets.

The model dynamically learns latent spectral representations for each waveform sample.

---

## 3. Context Normalization

Signals are normalized using context statistics:

```python
value_norm = (value - mean_context) / std_context
````

This stabilizes training across samples with varying amplitudes and signal distributions.

---

## 4. Training Strategy

Training pipeline includes:

* Huber Loss
* AdamW Optimizer
* Learning Rate Warmup
* Cosine Learning Rate Decay
* Gradient Clipping
* Early Stopping

These strategies improve convergence stability and generalization performance.

---

# Evaluation Metric

Competition evaluation uses:

## Mean Squared Error (MSE)

computed only on hidden target points where:

```python
Is_Context = 0
```

## Validation Performance

| Metric | Score   |
| ------ | ------- |
| MSE    | ~0.0018 |
| RMSE   | ~0.043  |

---

# Inference Pipeline

The inference process follows:

1. Extract context points
2. Compute Gaussian Process interpolation features
3. Pass features through the transformer model
4. Predict missing waveform targets
5. Generate submission CSV

## Output Format

```csv
Sample_ID, Time_ms, Predicted_Value
```

Only rows with:

```python
Is_Context = 0
```

are included in final submissions.

---

# Tech Stack

* Python
* PyTorch
* Transformers
* Gaussian Processes
* Neural Processes
* NumPy
* Pandas
* Matplotlib
* Signal Processing

---

# Repository Structure

```bash
.
├── spectral_bridge_solution.ipynb
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── utils/
└── submissions/
```

---

# How to Run

## 1. Open Notebook

```bash
spectral_bridge_solution.ipynb
```

---

## 2. Run All Cells Sequentially

The notebook will:

* preprocess the dataset
* train the reconstruction model
* perform validation
* generate predictions
* export final submission files

---

# Applications

* Audio signal reconstruction
* Sparse waveform prediction
* Neural signal interpolation
* In-context learning research
* Time-series forecasting
* Spectral analysis systems

---

# Research Highlights

* Hybrid GP + Transformer reconstruction framework
* Context-aware signal representation learning
* Sparse waveform interpolation under low-observation settings
* Robust signal reconstruction using neural attention mechanisms
* Competition-grade inference pipeline
