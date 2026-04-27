#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export_tflite import export_mlp_int8


if __name__ == "__main__":
    export_mlp_int8()
