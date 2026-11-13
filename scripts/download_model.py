"""Download the pretrained surgical instrument detector from Hugging Face.

Model: cesaraha/yolov8s-surgical-instrument-detection-cholec80
License: CC BY-NC-SA 4.0 (non-commercial, research/education only).
The weights are downloaded at setup time instead of committed to Git.
The app also downloads them automatically on first run, so this script is optional.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import download_model  # noqa: E402


def main() -> None:
    model_path = download_model()
    size_mb = model_path.stat().st_size / 1e6
    print(f"Model saved to: {model_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
