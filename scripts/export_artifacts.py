#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export_tflite import train_artifact_models


if __name__ == "__main__":
    train_artifact_models()
