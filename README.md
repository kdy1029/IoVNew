# Lightweight Intrusion Detection System for CAN Bus Security

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project presents a complete machine learning pipeline for developing and evaluating lightweight Intrusion Detection Systems (IDS) for the Controller Area Network (CAN) bus, a critical component in modern vehicles. The primary goal is to create an effective and efficient IDS model capable of running on resource-constrained embedded platforms, such as Neural Processing Units (NPUs).

The project covers the full workflow: data preprocessing, leakage-aware model training, evaluation, visualization, artifact generation, and INT8 TensorFlow Lite export for on-device deployment. This work is based on research published in the IEEE VTC2026-Spring conference.

## Publication

A portion of this research has been accepted for publication:

**Evaluating Lightweight IDS Models for CAN Bus Security on Embedded Platforms.**  
*Naman Sharadkumar Jain, Inseo Hwang, Junggab Son, Daeyoung Kim*  
In the 2026 IEEE 103rd Vehicular Technology Conference: VTC2026-Spring, June 2026.

## Key Features

- **Comprehensive Data Handling**: Preprocessing scripts for the CICIoV2024 dataset, handling DoS, spoofing, and benign CAN traffic.
- **Robust Model Evaluation**: Training and evaluation for classical ML models and neural models, including Logistic Regression, LinearSVC, Decision Tree, ExtraTrees, MLP, and LSTM variants.
- **Data Leakage Prevention**: Uses `GroupShuffleSplit` and row-level hashing so identical packets do not appear in both training and testing sets.
- **NPU-Ready Model Export**: Exports the MLP model to TensorFlow Lite INT8 format for deployment on edge devices.
- **Result Visualization**: Generates ROC curves, precision-recall curves, coefficient plots, and runtime/performance comparison charts.

## Project Structure

```text
.
|-- artifacts/                  # Exported models, scalers, metadata, and TFLite output
|-- data/                       # Local CICIoV2024 CSV files (not committed)
|-- figures/                    # Generated plots and charts
|-- scripts/
|   |-- train_evaluate.py       # Main runnable training/evaluation entrypoint
|   |-- export_artifacts.py     # Exports sklearn models, thresholds, metrics, and board params
|   `-- export_tflite.py        # Exports the INT8 TensorFlow Lite MLP model
|-- src/
|   |-- data.py                 # Data loading, cleaning, grouped splitting, sequence building
|   |-- evaluate.py             # Metrics, threshold tuning, and score utilities
|   |-- export_tflite.py        # Artifact and TensorFlow Lite export logic
|   |-- models.py               # Model definitions and constructors
|   `-- plots.py                # Plotting utilities
|-- build_dataset.py            # Optional helper to combine decimal CSV files
|-- train_evaluate_models.py    # Backward-compatible wrapper
|-- train_and_quantize.py       # Backward-compatible artifact export wrapper
|-- requirements.txt
`-- README.md
```

## Getting Started

### Prerequisites

- Python 3.10+ recommended
- CICIoV2024 decimal CSV files placed locally under `data/decimal/`

The dataset and generated artifacts can be large and are intentionally not required to be committed.

### Installation

```sh
git clone https://github.com/kdy1029/IoVNew.git
cd IoVNew
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Prepare Data

Place the CICIoV2024 decimal CSV files in `data/decimal/`:

```text
data/decimal/decimal_benign.csv
data/decimal/decimal_DoS.csv
data/decimal/decimal_spoofing-GAS.csv
data/decimal/decimal_spoofing-RPM.csv
data/decimal/decimal_spoofing-SPEED.csv
data/decimal/decimal_spoofing-STEERING_WHEEL.csv
```

If you need a combined decimal CSV for inspection:

```sh
python build_dataset.py
```

### Train and Evaluate Models

```sh
python scripts/train_evaluate.py
```

This runs the leakage-aware training and evaluation pipeline, including threshold tuning and generated plots. The compatibility wrapper also works:

```sh
python train_evaluate_models.py
```

### Export Board Artifacts

```sh
python scripts/export_artifacts.py
```

This saves `scaler.npz`, `thresholds.json`, `metrics.csv`, model objects, and board-friendly parameters under `artifacts/`.

### Export TFLite Model

```sh
python scripts/export_tflite.py
```

This trains the MLP using the same preprocessing path and writes `artifacts/mlp_int8.tflite`, `artifacts/scaler.npz`, and `artifacts/meta.json`.

### Generate Figures

```sh
python plot_roc_curve.py
python plot_model_performance_cpu.py
python plot_model_performance_gpu.py
python plot_runtime_imx.py
```

## Configuration

No secrets are required. `.env.example` documents the expected local paths for users who prefer environment-based configuration, but the scripts use the repository-local defaults unless changed in code.

## Results

This project demonstrates the feasibility of deploying a high-performance IDS on embedded systems. The quantized INT8 MLP model achieves a strong balance between detection accuracy and computational efficiency.

- **MLP Performance**: The final model achieves an F1-score of **0.8331** and a ROC-AUC of **0.9299** on the test set.
- **Quantization Impact**: The model is successfully converted to a TFLite INT8 format, significantly reducing its size and making it compatible with hardware accelerators like NPUs, with minimal performance degradation.

Detailed performance metrics and confusion matrices are printed during script execution, and plots are saved in `figures/`.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{jain2026evaluating,
  title={Evaluating Lightweight IDS Models for CAN Bus Security on Embedded Platforms},
  author={Jain, Naman Sharadkumar and Hwang, Inseo and Son, Junggab and Kim, Daeyoung},
  booktitle={2026 IEEE 103rd Vehicular Technology Conference: VTC2026-Spring},
  year={2026},
  month={June},
  organization={IEEE}
}
```

