# Lightweight Intrusion Detection System for CAN Bus Security

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project presents a complete machine learning pipeline for developing and evaluating lightweight Intrusion Detection Systems (IDS) for the Controller Area Network (CAN) bus, a critical component in modern vehicles. The primary goal is to create an effective and efficient IDS model capable of running on resource-constrained embedded platforms, such as Neural Processing Units (NPUs).

The project covers the entire workflow: from data preprocessing and exploratory analysis to training multiple ML/DL models, and finally, quantizing the best-performing model into a TFLite INT8 format for on-device deployment. This work is based on the research published in the IEEE VTC2026-Spring conference.

##  academically published

A portion of this research has been accepted for publication:

**Evaluating Lightweight IDS Models for CAN Bus Security on Embedded Platforms.**
*Naman Sharadkumar Jain, Inseo Hwang, Junggab Son, Daeyoung Kim*
In the 2026 IEEE 103rd Vehicular Technology Conference: VTC2026-Spring, June 2026.

## Key Features

- **Comprehensive Data Handling**: Preprocessing scripts for the CICIoV2024 dataset, handling various attack types (DoS, Spoofing) and benign traffic.
- **Robust Model Evaluation**: Rigorous training and evaluation of multiple models, including:
  - **Classical ML**: Logistic Regression, LinearSVC, Decision Tree, ExtraTrees
  - **Deep Learning**: MLP (Multi-Layer Perceptron), LSTM
- **Data Leakage Prevention**: Employs `GroupShuffleSplit` to ensure that identical data packets do not appear in both training and testing sets, ensuring a more realistic evaluation.
- **NPU-Ready Model Export**: The final MLP model is quantized to **INT8** using post-training quantization and exported to the **TensorFlow Lite (`.tflite`)** format, making it suitable for deployment on edge devices with NPUs.
- **Result Visualization**: Generates ROC curves, Precision-Recall curves, and performance comparison charts for clear analysis.

## Project Structure

```
.
├── artifacts/              # Exported models (logreg.joblib, mlp_int8.tflite), scaler, and metadata
├── data/                   # Raw dataset files (.csv)
├── figures/                # Saved plots and charts from visualization
├── .gitignore
├── main.py                 # Main script for training and evaluating all models
├── train_and_export.py     # Script to train and export the final TFLite model
├── plot_roc_curve.py       # Script for visualizing model performance
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- The required Python libraries are listed in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```sh
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### How to Run the Pipeline

1.  **Place Datasets**: Place your CAN bus dataset CSV files inside the `data/` directory.

2.  **Train and Evaluate All Models**: Run the main script to see a comparative analysis of all implemented models.
    ```sh
    python main.py
    ```

3.  **Train and Export the Final TFLite Model**: This script trains the MLP, quantizes it to INT8, and saves the `.tflite` file and scaler in the `artifacts/` directory.
    ```sh
    python train_and_export.py
    ```

4.  **Visualize Results**: Generate and save performance plots in the `figures/` directory.
    ```sh
    python plot_roc_curve.py
    ```

## Results

This project successfully demonstrates the feasibility of deploying a high-performance IDS on embedded systems. The quantized INT8 MLP model achieves a strong balance between detection accuracy and computational efficiency.

- **MLP Performance**: The final model achieves an F1-score of **0.8331** and a ROC-AUC of **0.9299** on the test set.
- **Quantization Impact**: The model is successfully converted to a TFLite INT8 format, significantly reducing its size and making it compatible with hardware accelerators like NPUs, with minimal performance degradation.

Detailed performance metrics and confusion matrices are printed to the console during script execution, and plots are saved in the `figures/` directory.

## Citation

If you find this work useful in your research, please consider citing our paper:

```
@inproceedings{jain2026evaluating,
  title={Evaluating Lightweight IDS Models for CAN Bus Security on Embedded Platforms},
  author={Jain, Naman Sharadkumar and Hwang, Inseo and Son, Junggab and Kim, Daeyoung},
  booktitle={2026 IEEE 103rd Vehicular Technology Conference: VTC2026-Spring},
  year={2026},
  month={June},
  organization={IEEE}
}
```
