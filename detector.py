"""Core detection logic for SurgiVision Lite.

Kept free of Streamlit code so it can be unit-tested and reused from scripts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolov8s_cholec80.pt"
# CPU keeps setup simple (no CUDA install) and runs on any laptop.
DEVICE = "cpu"
DETECTION_COLUMNS = ["frame", "time_s", "instrument", "confidence", "x1", "y1", "x2", "y2"]

# VP8/WebM plays in all modern browsers and ships with the opencv-python wheel.
# H.264 would need the separate OpenH264 DLL, so it is not used. mp4v is a last resort:
# the file downloads fine but browsers may not play it inline.
VIDEO_CODECS = [("VP80", ".webm"), ("mp4v", ".mp4")]


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python scripts/download_model.py")
    return YOLO(str(model_path))


def _result_to_rows(result, frame: int, time_s: float) -> list[dict]:
    """Convert one Ultralytics result into plain detection rows."""
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        rows.append({
            "frame": frame,
            "time_s": round(time_s, 3),
            "instrument": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return rows


@dataclass
class ImageResult:
    annotated_bgr: np.ndarray
    detections: pd.DataFrame
    inference_ms: float  # model forward pass only, as reported by Ultralytics
    total_ms: float      # preprocessing + inference + postprocessing


def detect_image(model: YOLO, image_bgr: np.ndarray, conf: float = 0.25) -> ImageResult:
    start = time.perf_counter()
    result = model.predict(image_bgr, conf=conf, device=DEVICE, verbose=False)[0]
    total_ms = (time.perf_counter() - start) * 1000
    detections = pd.DataFrame(_result_to_rows(result, frame=0, time_s=0.0), columns=DETECTION_COLUMNS)
    return ImageResult(result.plot(), detections, result.speed["inference"], total_ms)


@dataclass
class VideoResult:
    output_path: Path
    detections: pd.DataFrame
    per_frame_counts: pd.DataFrame  # one row per frame: frame, time_s, instruments
    frames_processed: int
    source_fps: float
    duration_s: float
    processing_time_s: float
    processing_fps: float
    mean_inference_ms: float


def _open_writer(base_path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    for fourcc, ext in VIDEO_CODECS:
        path = base_path.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("No browser-compatible video codec (H.264 or VP8) is available in this OpenCV build.")


def process_video(
    model: YOLO,
    input_path: Path,
    output_base: Path,
    conf: float = 0.25,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoResult:
    """Run detection on every frame and write an annotated video.

    `output_base` is a path without the final extension; the codec decides .mp4 or .webm.
    """
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0  # some containers report 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, output_path = _open_writer(Path(output_base), source_fps, (width, height))
    rows: list[dict] = []
    counts: list[dict] = []
    inference_ms: list[float] = []
    frame_index = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frame_index / source_fps
            result = model.predict(frame, conf=conf, device=DEVICE, verbose=False)[0]
            frame_rows = _result_to_rows(result, frame_index, time_s)
            rows.extend(frame_rows)
            counts.append({"frame": frame_index, "time_s": round(time_s, 3), "instruments": len(frame_rows)})
            inference_ms.append(result.speed["inference"])
            writer.write(result.plot())
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, total_frames)
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("The video contains no readable frames.")

    processing_time_s = time.perf_counter() - start
    return VideoResult(
        output_path=output_path,
        detections=pd.DataFrame(rows, columns=DETECTION_COLUMNS),
        per_frame_counts=pd.DataFrame(counts),
        frames_processed=frame_index,
        source_fps=source_fps,
        duration_s=frame_index / source_fps,
        processing_time_s=processing_time_s,
        processing_fps=frame_index / processing_time_s,
        mean_inference_ms=float(np.mean(inference_ms)),
    )


def summarize_detections(detections: pd.DataFrame, source_fps: float | None = None) -> pd.DataFrame:
    """Per-instrument statistics.

    Without tracking we cannot count *unique* instruments, so for video we report how many
    frames each instrument class appears in and the equivalent on-screen time.
    """
    columns = ["instrument", "detections", "mean_confidence", "min_confidence", "max_confidence"]
    if source_fps:
        columns += ["frames_present", "visible_s"]
    if detections.empty:
        return pd.DataFrame(columns=columns)

    grouped = detections.groupby("instrument")
    summary = pd.DataFrame({
        "detections": grouped.size(),
        "mean_confidence": grouped["confidence"].mean().round(3),
        "min_confidence": grouped["confidence"].min().round(3),
        "max_confidence": grouped["confidence"].max().round(3),
    })
    if source_fps:
        summary["frames_present"] = grouped["frame"].nunique()
        summary["visible_s"] = (summary["frames_present"] / source_fps).round(2)
    return summary.sort_values("detections", ascending=False).reset_index()
