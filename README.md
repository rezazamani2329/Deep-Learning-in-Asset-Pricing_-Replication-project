# Deep Learning in Asset Pricing — Replication Project

## Overview

This repository contains a replication and extension of the paper **“Deep Learning in Asset Pricing”** by Luyang Chen, Markus Pelger, and Jason Zhu.

The project studies how deep learning can be used to estimate nonlinear asset-pricing models from a large set of firm-level characteristics and macroeconomic information.

The central idea is to learn a **stochastic discount factor (SDF)** directly from data while incorporating the economic restriction implied by the no-arbitrage condition.

This project will progressively reproduce the main components of the original methodology, beginning with data preparation and benchmark models and then moving toward neural-network, recurrent-network, and adversarial asset-pricing models.

---

## Reference Paper

**Chen, L., Pelger, M., & Zhu, J.**

**Deep Learning in Asset Pricing**

*Management Science*, Volume 70, Issue 2, pp. 714–750.

DOI: `10.1287/mnsc.2023.4695`

The original paper develops a deep-learning framework that combines:

* Firm-level characteristics
* Macroeconomic information
* Nonlinear neural networks
* Recurrent neural networks for economic states
* No-arbitrage asset-pricing restrictions
* Adversarial learning for constructing informative test assets

---

## Project Objectives

The main objectives of this replication are to:

1. Reconstruct the asset-pricing data pipeline used in the paper.
2. Build benchmark linear asset-pricing models.
3. Estimate nonlinear stochastic discount factors using neural networks.
4. Incorporate macroeconomic information through recurrent neural networks.
5. Implement the adversarial asset-pricing framework.
6. Evaluate the models using both statistical and economic performance measures.
7. Compare the replication results with the findings reported in the original paper.
8. Explore possible extensions and robustness checks.

---

## Economic Framework

The project is based on the fundamental no-arbitrage condition

$$
E_t[M_{t+1}R_{i,t+1}] = 0,
$$

where

* \(M_{t+1}\) is the stochastic discount factor,
* \(R_{i,t+1}\) represents asset returns, and
* the conditional expectation is based on information available at time \(t\).

The stochastic discount factor can be represented through a portfolio of asset returns,

$$
M_{t+1}
=
1-
\sum_{i=1}^{N_t}
\omega_{i,t}R_{i,t+1},
$$

where the portfolio weights are learned as nonlinear functions of firm characteristics and aggregate economic information:

$$
\omega_{i,t}
=
g_{\theta}(X_{i,t},H_t).
$$

Here,

* \(X_{i,t}\) represents firm-specific characteristics,
* \(H_t\) represents the state of the economy, and
* \(g_{\theta}\) is a neural network parameterized by \(\theta\).

---

## Model Architecture

The full replication will be developed incrementally.

### 1. Benchmark Models

Traditional asset-pricing and linear models will provide benchmarks against which the deep-learning models can be evaluated.

### 2. Feedforward Neural Network

A feedforward neural network will learn nonlinear relationships between firm characteristics and SDF portfolio weights.

Conceptually,

```text
Firm Characteristics
        |
        v
Feedforward Neural Network
        |
        v
SDF Portfolio Weights
        |
        v
Stochastic Discount Factor
```

### 3. Macroeconomic State Network

Time-series information about the economy will be incorporated using a recurrent neural-network architecture.

```text
Macroeconomic Variables
          |
          v
        LSTM
          |
          v
Hidden Economic State
          |
          +------------------+
                             |
Firm Characteristics        |
          |                  |
          +------------------+
                  |
                  v
          Neural Network
                  |
                  v
          SDF Portfolio Weights
```

### 4. Adversarial Network

The full model introduces an adversarial network that searches for conditioning functions and test assets that expose the largest pricing errors.

The SDF network attempts to minimize these pricing errors while the adversarial network identifies the most informative moment conditions.

This creates a minimax learning problem similar in spirit to a Generative Adversarial Network.

---

## Replication Roadmap

The project will be developed in the following stages:

### Phase 1 — Data Construction

* Collect stock-return data
* Collect firm-level characteristics
* Collect macroeconomic variables
* Merge stock, characteristic, and macro data
* Handle missing observations
* Normalize characteristics
* Construct training, validation, and test samples
* Verify the data against descriptive statistics from the paper

### Phase 2 — Exploratory Analysis

* Examine the cross-section of stock returns
* Analyze characteristic distributions
* Study missing-data patterns
* Examine macroeconomic variables
* Investigate temporal and cross-sectional variation

### Phase 3 — Linear Benchmarks

* Linear SDF models
* Traditional factor-model benchmarks
* Characteristic-based linear specifications
* Out-of-sample evaluation

### Phase 4 — Feedforward Neural-Network SDF

* Construct the neural-network SDF
* Define the asset-pricing loss
* Train the model
* Tune hyperparameters
* Evaluate out-of-sample performance

### Phase 5 — Macro State Model

* Construct macroeconomic time-series inputs
* Implement the LSTM
* Estimate latent economic states
* Combine macro states with firm characteristics
* Evaluate the incremental contribution of macroeconomic information

### Phase 6 — Adversarial Asset-Pricing Model

* Construct the adversarial network
* Generate conditional test assets
* Implement the minimax objective
* Develop alternating training procedures
* Evaluate pricing errors and portfolio performance

### Phase 7 — Replication and Robustness

* Compare results with the original paper
* Analyze SDF portfolio performance
* Evaluate pricing errors
* Evaluate explained variation
* Study feature importance
* Analyze economic states
* Perform robustness checks

---

## Repository Structure

The intended project structure is:

```text
Deep-Learning-in-Asset-Pricing_-Replication-project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_linear_baseline.ipynb
│   ├── 04_feedforward_sdf.ipynb
│   ├── 05_lstm_macro_state.ipynb
│   ├── 06_gan_asset_pricing.ipynb
│   └── 07_results_replication.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── results/
│   ├── figures/
│   └── tables/
│
└── paper/
    └── notes/
```

The structure may evolve as the replication progresses.

---

## Evaluation

Model performance will be evaluated using both statistical and economic criteria.

Key metrics will include:

* Out-of-sample pricing errors
* Explained variation
* Sharpe ratio of the learned SDF portfolio
* Cross-sectional predictive performance
* Stability across time periods
* Performance across different market conditions

Additional diagnostics will be added as the project develops.

---

## Technology Stack

The project will primarily use:

```text
Python
NumPy
pandas
PyTorch
scikit-learn
SciPy
statsmodels
Matplotlib
Jupyter Notebook
Git / GitHub
```

Additional libraries may be introduced as required.

---

## Reproducibility

The project is designed to maintain a reproducible workflow.

The general pipeline will follow:

```text
Raw Data
   |
   v
Data Cleaning
   |
   v
Feature Construction
   |
   v
Train / Validation / Test Split
   |
   v
Benchmark Models
   |
   v
Deep Learning Models
   |
   v
Model Evaluation
   |
   v
Replication Tables and Figures
```

Random seeds, model configurations, preprocessing rules, and evaluation procedures will be documented whenever possible.

Large proprietary or licensed datasets will not be committed directly to the repository.

---

## Current Status

**Work in Progress**

The project is currently in the initial replication stage.

Current development sequence:

```text
[ ] Project setup
[ ] Data acquisition
[ ] Data cleaning and preprocessing
[ ] Exploratory data analysis
[ ] Linear benchmark
[ ] Feedforward neural-network SDF
[ ] LSTM macroeconomic state model
[ ] Adversarial asset-pricing model
[ ] Out-of-sample evaluation
[ ] Comparison with original paper
[ ] Robustness analysis
[ ] Final replication results
```

---

## Potential Extensions

After completing the core replication, possible extensions include:

* Alternative neural-network architectures
* Transformer-based macroeconomic state representations
* Alternative characteristic sets
* Alternative normalization procedures
* Different training and validation windows
* Transaction-cost adjustments
* Portfolio turnover analysis
* Regime-dependent performance
* Interpretability and feature-attribution analysis
* Comparison with modern machine-learning asset-pricing models

These extensions will be considered only after establishing a reliable baseline replication.

---

## Disclaimer

This repository is an **independent academic replication project**.

It is not the official implementation of Chen, Pelger, and Zhu and is not affiliated with the original authors or their institutions.

The goal is to understand, reproduce, and evaluate the methodology presented in the research paper.

---

## Author

**Reza Zamani**

UC Berkeley
Master of Financial Engineering

Research interests include quantitative finance, asset pricing, machine learning, deep learning, portfolio management, and systematic investment strategies.
