# SurgiVision Lite

**AI-based surgical instrument detection for laparoscopic images and recorded videos.**

Upload an image or a video and the app finds surgical instruments, draws labelled bounding boxes with confidence scores, counts them, shows simple statistics, and lets you download the annotated result.

> ⚠️ **Educational prototype.** Not a medical device and not validated for clinical use.

## Features

- Upload images (`jpg`, `png`, `bmp`) or videos (`mp4`, `avi`, `mov`, `mkv`, `webm`)
- Detects 7 laparoscopic instruments: **Bag, Bipolar, Clipper, Grasper, Hook, Irrigator, Scissors**
- Bounding boxes, labels and confidence scores drawn on the output
- Instrument counts and per-instrument statistics (count, mean/min/max confidence)
- Video: per-frame instrument chart, on-screen time per instrument, processing time, FPS, inference time
- Adjustable confidence threshold
- Download the annotated image/video and a CSV of every detection
- Handles bad input (corrupted images, unreadable videos) with clear error messages

## Tech stack

| Part | Used for |
|---|---|
| Python 3.11 | Application code |
| Ultralytics YOLOv8 | Object detection model |
| OpenCV | Reading/writing images and video frames |
| Pandas | Detection tables and statistics |
| Streamlit | Web interface |
| pytest | Unit tests |

## Model

This project does **not** train a model. It uses publicly available pretrained weights:

- **Model:** [`cesaraha/yolov8s-surgical-instrument-detection-cholec80`](https://huggingface.co/cesaraha/yolov8s-surgical-instrument-detection-cholec80) (YOLOv8s, 640×640)
- **Trained on:** Cholec80 laparoscopic cholecystectomy videos 41–45
- **License:** CC BY-NC-SA 4.0 (non-commercial)
- **Author-reported test metrics** (from the model card, *not measured by this project*): mAP@50 0.685, mAP@50-95 0.392. The card notes weaker results for Clipper and Scissors, and that Grasper is sometimes confused with Hook.

`scripts/check_model.py` verifies the downloaded weights are a detection model whose classes are surgical instruments (not ordinary COCO objects like "person" or "car").

## Project structure

```
SurgiVision-AI/
├── app.py                  # Streamlit user interface
├── detector.py             # Detection, video processing and statistics (no UI code)
├── scripts/
│   ├── download_model.py   # Downloads the pretrained weights from Hugging Face
│   ├── check_model.py      # Verifies the model and its class names
│   └── benchmark.py        # Runs on sample media and prints measured results
├── tests/
│   └── test_detector.py    # Unit tests
├── .streamlit/config.toml  # Upload size limit
├── requirements.txt        # Python dependencies (CPU-only PyTorch)
├── packages.txt            # Linux system libraries for cloud deployment
└── LICENSE                 # AGPL-3.0 (required by Ultralytics)
```

`models/`, `samples/` and `outputs/` are git-ignored: the weights and dataset frames are under non-commercial licenses and are downloaded, not committed.

## Setup

```powershell
git clone https://github.com/Nishagupta2212/SurgiVision-AI.git
cd SurgiVision-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py     # optional: the app also downloads the weights on first run
python scripts/check_model.py         # should print the 7 instrument classes
```

## Run

```powershell
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy (Streamlit Community Cloud, free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repository `Nishagupta2212/SurgiVision-AI`, branch `main`, main file `app.py`.
3. **Advanced settings** → Python version **3.11**.
4. Deploy. The first build installs CPU-only PyTorch and the app downloads the model weights on first load.

`packages.txt` installs the Linux libraries OpenCV needs. Free hosting has limited memory, so use short video clips.

> Vercel is not suitable: it runs short-lived serverless functions, while Streamlit needs a long-running server, and PyTorch exceeds Vercel's function size limit.

## Test

```powershell
pytest -v
```

## Results

All numbers below were measured by `scripts/benchmark.py` on a laptop CPU (Intel Core, Windows 11, 8 GB RAM), confidence threshold 0.25. Test frames come from [CholecSeg8k](https://huggingface.co/datasets/minwoosun/CholecSeg8k) (CC BY-NC-SA 4.0), using only Cholec80 videos **not** used to train the model (videos 41–45 excluded).

**Images (854×480):**

| Frame | Detections | Inference |
|---|---|---|
| video01 frame 28700 | Hook 0.80 | 233 ms (first call, includes warm-up) |
| video12 frame 19620 | Hook 0.76 | 152 ms |
| video17 frame 1683 | Grasper 0.71 | 139 ms |
| video26 frame 1975 | none | 141 ms |
| video37 frame 728 | Grasper 0.78, Grasper 0.49, Grasper 0.26 | 170 ms |
| video55 frame 628 | Grasper 0.87, Hook 0.83, Grasper 0.55 | 164 ms |

These are model outputs, not accuracy measurements. Accuracy (mAP) needs ground-truth boxes, which these frames do not have.

## Limitations

- Trained on one type of surgery (laparoscopic cholecystectomy) from a single hospital dataset; it will not generalise to robotic instruments or other procedures without retraining.
- Counts are **per frame**. There is no tracking, so the same instrument across 100 frames is counted 100 times, not once.
- CPU-only for simplicity, so video is processed slower than real time.
- No accuracy measurement on our own data (no labelled test set).

## Possible improvements

- Add object tracking (e.g. ByteTrack) to count unique instruments and draw their paths
- Fine-tune on labelled data to measure precision/recall/mAP directly
- GPU support for faster video processing

## Credits and licenses

- Code: AGPL-3.0 (see `LICENSE`), as required by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- Model weights: [cesaraha](https://huggingface.co/cesaraha/yolov8s-surgical-instrument-detection-cholec80), CC BY-NC-SA 4.0.
- Sample frames: CholecSeg8k (Hong et al., 2020), derived from Cholec80 (Twinanda et al., 2016), CC BY-NC-SA 4.0.
