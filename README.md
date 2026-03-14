# Spectral Bridge – Signal Reconstruction using In-Context Learning

## Overview

This project was developed for the **Spectral Bridge Challenge** conducted at **Cognizance, IIT Roorkee**.

The goal of the challenge is to reconstruct missing portions of audio signals using a few observed context points. Each audio sample contains sparse observations of a waveform, and the task is to predict the missing target points.

This project implements a learning-based approach that combines neural attention mechanisms with classical interpolation priors to reconstruct signals accurately.

---

## Problem Description

Each audio sample consists of:

- **100 time steps**
- **20 observed context points**
- **80 missing target points**

The model must infer the missing waveform values using the provided context.

Important constraints:

- Each sample is independent.
- No information can be transferred between samples.
- Prediction must be based only on the context points within that sample.

---

## Dataset

Dataset used:

Spectral Graffiti Dataset  
https://www.kaggle.com/datasets/fda137/spectral-graffiti/data

Dataset properties:

- ~80,000 samples
- 100 time points per sample
- 20 context points
- 80 target points to reconstruct
- Sampling rate: 1kHz

---

## Methodology

The solution follows a hybrid approach combining classical signal reconstruction and neural learning.

Key components:

### 1. Gaussian Process Features
Multiple Gaussian Process predictions with different kernel scales are computed to capture signal structure at multiple frequencies.

### 2. Neural Process Architecture
A transformer-based architecture is used to process the context points and predict missing targets.

The model learns to infer the spectral signature of each sample dynamically.

### 3. Context Normalization
Signals are normalized using context statistics:

value_norm = (value − mean_context) / std_context

This stabilizes training across samples with different amplitudes.

### 4. Training Strategy

Training uses:

- Huber loss for stability
- AdamW optimizer
- Learning rate warmup and cosine decay
- Gradient clipping
- Early stopping based on validation loss

---

## Evaluation Metric

The competition evaluates models using:

Mean Squared Error (MSE)

computed only on the hidden target points where:

Is_Context = 0

Validation performance achieved:

MSE ≈ 0.0018  
RMSE ≈ 0.043

---

## Inference Pipeline

The inference process:

1. Extract context points for each sample
2. Compute GP interpolation features
3. Pass context and features through the trained model
4. Predict missing target values
5. Generate submission CSV

Output format:

Sample_ID, Time_ms, Predicted_Value

Only rows where Is_Context = 0 are included in submission.

---

## Repository Structure

## How to Run

1. Open the notebook:

spectral_bridge_solution.ipynb

2. Run all cells sequentially.

3. The notebook will:

- train the model
- perform validation
- generate the final submission file.
