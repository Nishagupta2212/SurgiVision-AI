"""Unit tests for detector.py.

Tests that need the model weights are skipped if the weights have not been downloaded.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import (  # noqa: E402
    DETECTION_COLUMNS,
    MODEL_PATH,
    detect_image,
    load_model,
    process_video,
    summarize_detections,
)

EXPECTED_CLASSES = {"Bag", "Bipolar", "Clipper", "Grasper", "Hook", "Irrigator", "Scissors"}
SAMPLE_IMAGES = Path(__file__).resolve().parent.parent / "samples" / "images"

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model weights not downloaded")


@pytest.fixture(scope="module")
def model():
    return load_model()


def write_synthetic_video(path: Path, frames: int = 10, size=(320, 240), fps: int = 25) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


# ---- summarize_detections (no model needed) -------------------------------------------

def test_summary_of_empty_detections_is_empty_with_columns():
    summary = summarize_detections(pd.DataFrame(columns=DETECTION_COLUMNS), source_fps=25)
    assert summary.empty
    assert "visible_s" in summary.columns


def test_summary_counts_and_visible_time():
    detections = pd.DataFrame([
        {"frame": 0, "time_s": 0.0, "instrument": "Hook", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 1, "y2": 1},
        {"frame": 1, "time_s": 0.04, "instrument": "Hook", "confidence": 0.7, "x1": 0, "y1": 0, "x2": 1, "y2": 1},
        {"frame": 1, "time_s": 0.04, "instrument": "Grasper", "confidence": 0.5, "x1": 0, "y1": 0, "x2": 1, "y2": 1},
    ])
    summary = summarize_detections(detections, source_fps=25).set_index("instrument")
    assert summary.loc["Hook", "detections"] == 2
    assert summary.loc["Hook", "mean_confidence"] == pytest.approx(0.8)
    assert summary.loc["Hook", "frames_present"] == 2
    assert summary.loc["Hook", "visible_s"] == pytest.approx(0.08)
    assert summary.loc["Grasper", "detections"] == 1


def test_image_summary_has_no_video_columns():
    summary = summarize_detections(pd.DataFrame(columns=DETECTION_COLUMNS))
    assert "visible_s" not in summary.columns


# ---- model-dependent tests -------------------------------------------------------------

@needs_model
def test_model_detects_surgical_instruments_not_coco_objects(model):
    assert model.task == "detect"
    assert set(model.names.values()) == EXPECTED_CLASSES


@needs_model
def test_blank_image_has_no_detections(model):
    result = detect_image(model, np.zeros((480, 854, 3), dtype=np.uint8))
    assert result.detections.empty
    assert result.annotated_bgr.shape == (480, 854, 3)


@needs_model
def test_detections_respect_confidence_threshold(model):
    images = sorted(SAMPLE_IMAGES.glob("*.png")) if SAMPLE_IMAGES.exists() else []
    if not images:
        pytest.skip("no sample images in samples/images")
    for image_path in images:
        result = detect_image(model, cv2.imread(str(image_path)), conf=0.5)
        assert (result.detections["confidence"] >= 0.5).all()
        assert set(result.detections["instrument"]) <= EXPECTED_CLASSES


@needs_model
def test_process_video_writes_every_frame(model, tmp_path):
    video = write_synthetic_video(tmp_path / "input.mp4", frames=10)
    result = process_video(model, video, tmp_path / "output")
    assert result.frames_processed == 10
    assert len(result.per_frame_counts) == 10
    assert result.output_path.exists()

    capture = cv2.VideoCapture(str(result.output_path))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert written == 10


@needs_model
def test_process_video_rejects_unreadable_file(model, tmp_path):
    bad_file = tmp_path / "not_a_video.mp4"
    bad_file.write_bytes(b"this is not a video")
    with pytest.raises(ValueError):
        process_video(model, bad_file, tmp_path / "output")
