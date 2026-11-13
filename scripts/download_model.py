"""Download the pretrained surgical instrument detector from Hugging Face.

Model: cesaraha/yolov8s-surgical-instrument-detection-cholec80
License: CC BY-NC-SA 4.0 (non-commercial, research/education only).
The weights are downloaded at setup time instead of committed to Git.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "cesaraha/yolov8s-surgical-instrument-detection-cholec80"
FILENAME = "yolov8s_cholec80.pt"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODELS_DIR)
    size_mb = Path(model_path).stat().st_size / 1e6
    print(f"Model saved to: {model_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
