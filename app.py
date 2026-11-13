"""SurgiVision Lite - Streamlit UI for surgical instrument detection.

Run with:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import DEVICE, MODEL_PATH, detect_image, load_model, process_video, summarize_detections

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
OUTPUT_DIR = Path(tempfile.gettempdir()) / "surgivision_outputs"

st.set_page_config(page_title="SurgiVision Lite", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


def render_sidebar() -> float:
    st.sidebar.header("Settings")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                             help="Detections below this confidence are discarded.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    model = get_model()
    st.sidebar.markdown(
        f"**Weights:** `{MODEL_PATH.name}`  \n"
        f"**Architecture:** YOLOv8s (Ultralytics)  \n"
        f"**Device:** {DEVICE.upper()}  \n"
        f"**Classes:** {', '.join(model.names.values())}"
    )
    st.sidebar.caption(
        "Pretrained weights by [cesaraha](https://huggingface.co/cesaraha/"
        "yolov8s-surgical-instrument-detection-cholec80), trained on Cholec80 "
        "(CC BY-NC-SA 4.0). Non-commercial use only."
    )
    return conf


def decode_image(uploaded_file) -> np.ndarray | None:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def render_summary(summary, empty_message: str) -> None:
    if summary.empty:
        st.info(empty_message)
        return
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Detections per instrument**")
        st.bar_chart(summary.set_index("instrument")["detections"])
    with right:
        st.markdown("**Statistics**")
        st.dataframe(summary, hide_index=True, width="stretch")


def image_tab(conf: float) -> None:
    uploaded = st.file_uploader("Upload an image", type=IMAGE_TYPES, key="image_upload")
    if not uploaded:
        st.caption("Supported formats: " + ", ".join(IMAGE_TYPES))
        return

    image = decode_image(uploaded)
    if image is None:
        st.error("This file could not be read as an image. It may be corrupted.")
        return

    result = detect_image(get_model(), image, conf)
    summary = summarize_detections(result.detections)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Instruments detected", len(result.detections))
    m2.metric("Instrument types", result.detections["instrument"].nunique())
    m3.metric("Inference time", f"{result.inference_ms:.1f} ms")
    m4.metric("Total processing time", f"{result.total_ms:.1f} ms")

    left, right = st.columns(2)
    left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
    right.image(cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB), caption="Detections",
                width="stretch")

    render_summary(summary, "No instruments detected at this confidence threshold.")
    if not result.detections.empty:
        with st.expander("All detections (bounding boxes in pixels)"):
            st.dataframe(result.detections.drop(columns=["frame", "time_s"]), hide_index=True,
                         width="stretch")

    stem = Path(uploaded.name).stem
    ok, png = cv2.imencode(".png", result.annotated_bgr)
    d1, d2 = st.columns(2)
    if ok:
        d1.download_button("Download annotated image", png.tobytes(), f"{stem}_detected.png", "image/png")
    d2.download_button("Download detections (CSV)", result.detections.to_csv(index=False),
                       f"{stem}_detections.csv", "text/csv")


def video_tab(conf: float) -> None:
    uploaded = st.file_uploader("Upload a video", type=VIDEO_TYPES, key="video_upload")
    if not uploaded:
        st.caption("Supported formats: " + ", ".join(VIDEO_TYPES))
        return

    # Results are kept in session_state so download buttons (which rerun the script)
    # don't throw away a finished analysis.
    run_key = (uploaded.name, uploaded.size, conf)
    if st.button("Analyze video", type="primary"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        input_path = OUTPUT_DIR / f"input_{Path(uploaded.name).name}"
        input_path.write_bytes(uploaded.getvalue())
        progress = st.progress(0.0, text="Processing frames...")

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                progress.progress(min(done / total, 1.0), text=f"Processing frame {done} of {total}")

        try:
            result = process_video(get_model(), input_path, OUTPUT_DIR / f"{Path(uploaded.name).stem}_detected",
                                   conf, on_progress)
        except (ValueError, RuntimeError) as error:
            progress.empty()
            st.error(str(error))
            return
        finally:
            input_path.unlink(missing_ok=True)
        progress.empty()
        st.session_state["video_result"] = (run_key, result)

    stored = st.session_state.get("video_result")
    if not stored or stored[0] != run_key:
        st.caption("Click **Analyze video** to run detection on every frame.")
        return
    result = stored[1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frames processed", result.frames_processed)
    m2.metric("Video duration", f"{result.duration_s:.1f} s")
    m3.metric("Processing time", f"{result.processing_time_s:.1f} s")
    m4.metric("Processing FPS", f"{result.processing_fps:.1f}")
    m5.metric("Mean inference / frame", f"{result.mean_inference_ms:.1f} ms")

    counts = result.per_frame_counts
    frames_with_tools = int((counts["instruments"] > 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames with ≥1 instrument", f"{frames_with_tools} ({frames_with_tools / len(counts):.0%})")
    c2.metric("Max instruments in one frame", int(counts["instruments"].max()))
    c3.metric("Mean instruments per frame", f"{counts['instruments'].mean():.2f}")

    st.video(str(result.output_path))
    st.markdown("**Instruments detected per frame**")
    st.line_chart(counts.set_index("time_s")["instruments"], x_label="Time (s)", y_label="Instruments")

    summary = summarize_detections(result.detections, result.source_fps)
    render_summary(summary, "No instruments detected in this video at this confidence threshold.")
    st.caption("Counts are per frame. Without tracking, the same instrument seen in many frames is "
               "counted once per frame, so 'detections' is not the number of unique instruments.")

    d1, d2 = st.columns(2)
    mime = "video/mp4" if result.output_path.suffix == ".mp4" else "video/webm"
    d1.download_button("Download annotated video", result.output_path.read_bytes(),
                       result.output_path.name, mime)
    d2.download_button("Download detections (CSV)", result.detections.to_csv(index=False),
                       f"{result.output_path.stem}.csv", "text/csv")


def main() -> None:
    st.title("🩺 SurgiVision Lite")
    st.markdown("AI-based surgical instrument detection for laparoscopic images and recorded videos.")
    st.warning("Educational prototype only. Not a medical device and not validated for clinical use.", icon="⚠️")

    try:
        conf = render_sidebar()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    image, video = st.tabs(["🖼️ Image", "🎞️ Video"])
    with image:
        image_tab(conf)
    with video:
        video_tab(conf)


if __name__ == "__main__":
    main()
