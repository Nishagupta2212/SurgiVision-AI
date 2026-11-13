"""Sanity-check the environment and the downloaded model.

Confirms that:
  1. PyTorch can see the GPU (or falls back to CPU).
  2. The weights load as a YOLO *detection* model.
  3. The class names are surgical instruments, not COCO objects.
  4. Inference runs end to end.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8s_cholec80.pt"

# Classes listed on the model card. Compared case-insensitively.
EXPECTED_CLASSES = {"bag", "bipolar", "clipper", "grasper", "hook", "irrigator", "scissors"}


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python scripts/download_model.py")

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    print(f"Model task      : {model.task}")
    print("Class names     :")
    for class_id, name in model.names.items():
        print(f"  {class_id}: {name}")

    found = {name.lower() for name in model.names.values()}
    if found == EXPECTED_CLASSES:
        print("Class check     : OK - matches the model card")
    else:
        print(f"Class check     : MISMATCH - missing {EXPECTED_CLASSES - found}, extra {found - EXPECTED_CLASSES}")

    # A black frame contains no instruments, so we expect zero boxes.
    blank_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.predict(blank_frame, device=device, verbose=False)[0]
    print(f"Blank-frame test: {len(result.boxes)} detections (expected 0)")
    print(f"Inference time  : {result.speed['inference']:.1f} ms (first call includes warm-up)")


if __name__ == "__main__":
    main()
