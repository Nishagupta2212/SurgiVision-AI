"""Measure detection results and speed on local sample media.

Prints the numbers used in the README so every reported figure is reproducible:
    python scripts/benchmark.py samples/video52_test_clip.mp4 --images samples/images
"""
import argparse
import sys
import tempfile
from pathlib import Path

import platform

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import DEVICE, detect_image, load_model, process_video, summarize_detections  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Video file to process")
    parser.add_argument("--images", type=Path, help="Folder of images to run detection on")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    print(f"Device: {DEVICE} ({platform.processor()}) | conf={args.conf}\n")
    model = load_model()

    if args.images:
        print("Images")
        for image_path in sorted(args.images.glob("*.png")):
            result = detect_image(model, cv2.imread(str(image_path)), args.conf)
            found = ", ".join(f"{r.instrument} {r.confidence:.2f}" for r in result.detections.itertuples()) or "none"
            print(f"  {image_path.name}: {found}  ({result.inference_ms:.1f} ms inference)")
        print()

    with tempfile.TemporaryDirectory() as tmp:
        result = process_video(model, args.video, Path(tmp) / "out", args.conf)
        print(f"Video: {args.video.name}")
        print(f"  Frames processed      : {result.frames_processed} ({result.duration_s:.1f} s @ {result.source_fps:.0f} fps)")
        print(f"  Processing time       : {result.processing_time_s:.2f} s")
        print(f"  Processing FPS        : {result.processing_fps:.1f} (end to end, incl. decode/draw/encode)")
        print(f"  Mean inference/frame  : {result.mean_inference_ms:.1f} ms")
        counts = result.per_frame_counts["instruments"]
        print(f"  Frames with >=1 tool  : {(counts > 0).sum()} / {len(counts)}")
        print(f"  Output codec          : {result.output_path.suffix}")
        print(summarize_detections(result.detections, result.source_fps).to_string(index=False))


if __name__ == "__main__":
    main()
