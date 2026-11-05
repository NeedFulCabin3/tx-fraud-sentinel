# TX Fraud Sentinel

> Production-ready scikit-learn machine learning pipeline for detecting fraudulent financial transactions under extreme class imbalance.

## Overview

In real-world financial transaction monitoring systems, fraud accounts for a minuscule fraction of overall volume (typically under one percent). Standard machine learning classifiers trained on such severely skewed data default to predicting the majority class, yielding misleadingly high accuracy while missing every fraudulent event. `tx-fraud-sentinel` solves this operational friction by combining class-weighted estimators, median-aware outlier scaling, and dynamic precision-recall threshold tuning into a cohesive, reproducible Python pipeline.

## How It Works

The execution pipeline operates through a deterministic multi-stage workflow:

1. **Synthetic Data Ingestion:** Generates a tabular transaction dataset mimicking credit card ledgers with a severe one percent minority class ratio via `make_classification`.
2. **Stratified Splitting:** Divides data into training and testing partitions using stratification (`stratify=y`) to preserve the exact minority class distribution across splits.
3. **Pipeline Encapsulation:** Combines preprocessing and modeling into an sklearn `Pipeline` to prevent data leakage during cross-validation and feature scaling.
4. **Median-Aware Scaling:** Utilizes `RobustScaler` to center data using medians and interquartile ranges, neutralizing distortion from extreme transaction amount outliers.
5. **Class-Wighted Estimation:** Instantiates a `RandomForestClassifier` with `class_weight='balanced'` to penalize misclassifications of the rare fraud class automatically.
6. **Threshold Optimization:** Computes precision-recall curves across continuous probability scores to dynamically locate the exact decision threshold that maximizes the F1-score, moving past arbitrary zero-point-five cutoffs.
7. **Interactive Evaluation:** Provides a command-line evaluation loop allowing engineers to test mock probability scores against both default and optimized decision boundaries in real time.

## Key Features

* **Data Leakage Prevention:** Encapsulates scaling and estimation steps inside a unified scikit-learn pipeline object.
* **Outlier-Resilient Scaling:** Normalizes transaction amounts and timestamps without being skewed by extreme financial outliers.
* **Native Class Imbalance Handling:** Employs algorithmic class re-weighting alongside parallelized tree bagging (`n_jobs=-1`).
* **Dynamic F1 Threshold Tuning:** Automatically calculates optimal classification boundaries using precision-recall curve metrics rather than relying on default assumptions.
* **Interactive Terminal Testing:** Features a built-in evaluation loop for testing probability scores on demand.

## Tech Stack & Core Dependencies Breakdown

* **Python 3.10+:** Target runtime environment utilizing modern typing and exception handling patterns.
* **NumPy (`1.20+`):** Core array manipulation and numerical vector operations.
* **Pandas (`1.3+`):** Tabular data structure construction, feature naming, and DataFrame manipulation.
* **scikit-learn (`1.0+`):** Provides machine learning algorithms (`RandomForestClassifier`), preprocessing utilities (`RobustScaler`), pipeline architecture, and evaluation metrics (`precision_recall_curve`, `roc_auc_score`, `classification_report`).

## Environment & Web-Based Quick Start

You can run this project directly in your browser using GitHub Codespaces or inside a local virtual environment.

### GitHub Codespaces (Browser-Only Setup)
1. Navigate to the repository home page on GitHub.
2. Click the green **Code** button and select the **Codespaces** tab.
3. Click **Create codespace on main** to spin up a cloud-hosted development container.
4. Once the terminal loads, install dependencies and run the pipeline.

### Local Virtual Environment Setup
```bash
# Clone repository and enter directory
git clone [https://github.com/your-username/tx-fraud-sentinel.git](https://github.com/your-username/tx-fraud-sentinel.git)
cd tx-fraud-sentinel

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install numpy pandas scikit-learn

# Execute the pipeline script
python main.py
```

## Repository Structure

```bash
tx-fraud-sentinel/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap

**[ ] Type Stubs & Static Analysis:** Introduce strict type hints across all functions and integrate mypy into the CI pipeline.

**[ ] Configurable YAML Profiles:** Externalize pipeline hyper-parameters (such as n_estimators, max_depth, and test split ratios) into an external configuration file.

**[ ] Structured Logging Migration:** Replace standard print statements with Python's built-in logging module to capture execution timestamps and severity levels.