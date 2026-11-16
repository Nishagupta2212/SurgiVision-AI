"""SurgiVision Lite - Streamlit UI for surgical instrument detection.

Run with:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics.utils.plotting import colors

from detector import DEVICE, MODEL_PATH, detect_image, load_model, process_video, summarize_detections

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
OUTPUT_DIR = Path(tempfile.gettempdir()) / "surgivision_outputs"
DEMO_DIR = Path(__file__).resolve().parent / "assets" / "demo"
DEMO_IMAGES = {
    "Grasper + Hook (video 55)": DEMO_DIR / "sample_grasper_hook.png",
    "Multiple graspers (video 37)": DEMO_DIR / "sample_graspers.png",
    "Hook (video 12)": DEMO_DIR / "sample_hook.png",
}
DEMO_VIDEO = DEMO_DIR / "sample_clip.mp4"
DEMO_VIDEO_NAME = "Demo clip (video 52, 13 s)"

# What each class is used for in laparoscopic cholecystectomy (gallbladder removal).
INSTRUMENT_INFO = {
    "Bag": "Specimen bag for removing tissue",
    "Bipolar": "Bipolar forceps for coagulation",
    "Clipper": "Applies clips to the cystic duct and artery",
    "Grasper": "Holds and retracts tissue",
    "Hook": "Electrocautery hook for dissection",
    "Irrigator": "Irrigation and suction of the field",
    "Scissors": "Cuts tissue and clipped structures",
}

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
.console-header {
    border: 1px solid #1E3A4C; border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    background: linear-gradient(135deg, #0F1E2E 0%, #0A0F1A 60%, #0D2A2A 100%);
}
.console-header h1 { font-size: 1.9rem; margin: 0; color: #F1F5F9; letter-spacing: 0.5px; }
.console-header p { margin: 0.3rem 0 0.8rem 0; color: #94A3B8; }
.pill {
    display: inline-block; padding: 0.2rem 0.7rem; margin: 0 0.4rem 0.3rem 0; border-radius: 999px;
    font-size: 0.78rem; border: 1px solid #1E3A4C; background: #0F1A2A; color: #CBD5E1;
}
.pill .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot.green { background: #22C55E; box-shadow: 0 0 6px #22C55E; }
.dot.teal { background: #2DD4BF; }
div[data-testid="stMetric"] {
    background: #111A2B; border: 1px solid #1E3A4C; border-left: 3px solid #2DD4BF;
    border-radius: 10px; padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] p { color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.6px; }
.section-title {
    color: #2DD4BF; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; margin: 1.2rem 0 0.5rem 0; border-bottom: 1px solid #1E3A4C; padding-bottom: 0.3rem;
}
.legend-row { display: flex; align-items: center; margin: 0.25rem 0; font-size: 0.85rem; }
.legend-swatch { width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; flex-shrink: 0; }
.legend-desc { color: #94A3B8; font-size: 0.75rem; margin-left: 20px; margin-top: -2px; }
.feed-label {
    font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem;
}
.feed-label .rec { color: #EF4444; }
.footer { color: #64748B; font-size: 0.75rem; text-align: center; margin-top: 2.5rem;
    border-top: 1px solid #1E3A4C; padding-top: 0.8rem; }
/* Robots: header surgical robot, sidebar assistant, empty-state helper. */
.console-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.robot-hero { width: 250px; flex-shrink: 0; }
.robot-hero .arm { transform-box: view-box; animation: sway 4s ease-in-out infinite; }
.robot-hero .a1 { transform-origin: 52px 38px; }
.robot-hero .a2 { transform-origin: 92px 38px; animation-delay: -1s; animation-duration: 5s; }
.robot-hero .a3 { transform-origin: 168px 38px; animation-delay: -2s; }
.robot-hero .a4 { transform-origin: 208px 38px; animation-delay: -3s; animation-duration: 4.5s; }
.beam { animation: beam 2s ease-in-out infinite; }
.led { animation: led 1.6s ease-in-out infinite; }
.bot-eyes { transform-box: fill-box; transform-origin: center; animation: blink 5s infinite; }
.bot-antenna { animation: led 1.2s ease-in-out infinite; }
@keyframes sway { 0%, 100% { transform: rotate(-3deg); } 50% { transform: rotate(3deg); } }
@keyframes beam { 0%, 100% { opacity: 0.08; } 50% { opacity: 0.3; } }
@keyframes led { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
@keyframes blink { 0%, 92%, 100% { transform: scaleY(1); } 95% { transform: scaleY(0.1); } }
@media (prefers-reduced-motion: reduce) { .arm, .beam, .led, .bot-eyes, .bot-antenna { animation: none; } }
@media (max-width: 760px) { .robot-hero { display: none; } }
.bot-card {
    display: flex; align-items: center; gap: 0.7rem; padding: 0.6rem 0.8rem; margin-bottom: 0.8rem;
    border: 1px solid #1E3A4C; border-radius: 12px; background: #0F1A2A;
}
.bot-card b { color: #F1F5F9; font-size: 0.9rem; }
.bot-card .bot-sub { color: #94A3B8; font-size: 0.75rem; line-height: 1.3; }
.bot-note {
    display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 1rem; margin: 0.5rem 0;
    border: 1px solid #1E3A4C; border-left: 3px solid #2DD4BF; border-radius: 10px; background: #111A2B;
    color: #CBD5E1; font-size: 0.9rem;
}
.bot-head { flex-shrink: 0; }
/* Roaming robots: rovers drive along the bottom edge, stop to talk, then leave; a drone patrols the top. */
.block-container { padding-bottom: 6rem; }
.rover, .drone { position: fixed; left: 0; z-index: 1000; }
.rover { bottom: 4px; width: 60px; animation: roam-right 24s linear infinite; }
.rover.r2 { animation-name: roam-left; animation-duration: 30s; animation-delay: -12s; }
.rover.r2 svg { transform: scaleX(-1); }
.rover:hover { animation-play-state: paused; cursor: pointer; }
.rover .bubble {
    position: absolute; bottom: 66px; left: 50%; transform: translateX(-50%); white-space: nowrap;
    background: #111A2B; border: 1px solid #2DD4BF; color: #E2E8F0; border-radius: 10px;
    padding: 0.25rem 0.6rem; font-size: 0.75rem; opacity: 0; pointer-events: none;
    animation: talk-right 24s linear infinite;
}
.rover.r2 .bubble { animation-name: talk-left; animation-duration: 30s; animation-delay: -12s; }
.rover:hover .bubble { opacity: 1 !important; animation-play-state: paused; }
.wheel { transform-box: fill-box; transform-origin: center; animation: spin 0.8s linear infinite; }
.scan-eye { animation: scan 1.8s ease-in-out infinite alternate; }
.arm-wave { transform-box: fill-box; transform-origin: left bottom; }
.rover:hover .arm-wave { animation: wave 0.35s ease-in-out infinite alternate; }
.drone { top: 70px; width: 72px; pointer-events: none; animation: fly 32s ease-in-out infinite; animation-delay: -6s; }
.drone svg { animation: hover 1.4s ease-in-out infinite alternate; }
.rotor { transform-box: fill-box; transform-origin: center; animation: rotor 0.15s linear infinite alternate; }
@keyframes roam-right {
    0% { transform: translateX(-90px); } 35%, 50% { transform: translateX(38vw); }
    100% { transform: translateX(calc(100vw + 90px)); }
}
@keyframes roam-left {
    0% { transform: translateX(calc(100vw + 90px)); } 40%, 55% { transform: translateX(62vw); }
    100% { transform: translateX(-90px); }
}
@keyframes talk-right { 0%, 36% { opacity: 0; } 38%, 48% { opacity: 1; } 50%, 100% { opacity: 0; } }
@keyframes talk-left { 0%, 41% { opacity: 0; } 43%, 53% { opacity: 1; } 55%, 100% { opacity: 0; } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes scan { from { transform: translateX(0); } to { transform: translateX(9px); } }
@keyframes wave { from { transform: rotate(0deg); } to { transform: rotate(-35deg); } }
@keyframes fly {
    0% { transform: translate(-100px, 0); } 25% { transform: translate(30vw, 40px); }
    50% { transform: translate(60vw, -10px); } 75% { transform: translate(85vw, 50px); }
    100% { transform: translate(calc(100vw + 100px), 0); }
}
@keyframes hover { from { transform: translateY(-4px); } to { transform: translateY(4px); } }
@keyframes rotor { from { transform: scaleX(1); } to { transform: scaleX(0.4); } }
@media (prefers-reduced-motion: reduce) { .rover, .drone { display: none; } }
</style>
"""

# Original illustration of a multi-arm patient-side surgical robot: grasper, endoscope, hook and scissors.
ROBOT_HERO = (
    '<svg class="robot-hero" viewBox="0 0 260 170" xmlns="http://www.w3.org/2000/svg" role="img" '
    'aria-label="Surgical robot with four instrument arms">'
    '<ellipse cx="130" cy="163" rx="80" ry="5" fill="#000" opacity=".35"/>'
    '<rect x="121" y="38" width="18" height="104" rx="4" fill="#162235" stroke="#2DD4BF" stroke-opacity=".35"/>'
    '<rect x="125" y="66" width="10" height="30" rx="2" fill="#0A0F1A"/>'
    '<circle class="led" cx="130" cy="56" r="2.5" fill="#22C55E"/>'
    '<rect x="92" y="138" width="76" height="18" rx="5" fill="#1E293B" stroke="#2DD4BF" stroke-opacity=".4"/>'
    '<circle cx="104" cy="159" r="4" fill="#334155"/><circle cx="156" cy="159" r="4" fill="#334155"/>'
    '<rect x="38" y="28" width="184" height="10" rx="5" fill="#334155"/>'
    '<g class="arm a1"><circle cx="52" cy="38" r="5" fill="#1E293B" stroke="#94A3B8"/>'
    '<line x1="52" y1="38" x2="44" y2="82" stroke="#94A3B8" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="44" y1="82" x2="66" y2="118" stroke="#CBD5E1" stroke-width="3.5" stroke-linecap="round"/>'
    '<circle cx="44" cy="82" r="4.5" fill="#0A0F1A" stroke="#2DD4BF" stroke-width="2"/>'
    '<path d="M66 118 l5 9 M66 118 l-3 10" stroke="#2DD4BF" stroke-width="2" stroke-linecap="round"/></g>'
    '<g class="arm a2"><circle cx="92" cy="38" r="5" fill="#1E293B" stroke="#94A3B8"/>'
    '<line x1="92" y1="38" x2="88" y2="78" stroke="#94A3B8" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="88" y1="78" x2="106" y2="112" stroke="#CBD5E1" stroke-width="3.5" stroke-linecap="round"/>'
    '<circle cx="88" cy="78" r="4.5" fill="#0A0F1A" stroke="#2DD4BF" stroke-width="2"/>'
    '<polygon class="beam" points="106,112 90,142 124,142" fill="#2DD4BF"/>'
    '<circle cx="106" cy="112" r="3" fill="#F1F5F9"/></g>'
    '<g class="arm a3"><circle cx="168" cy="38" r="5" fill="#1E293B" stroke="#94A3B8"/>'
    '<line x1="168" y1="38" x2="172" y2="78" stroke="#94A3B8" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="172" y1="78" x2="154" y2="112" stroke="#CBD5E1" stroke-width="3.5" stroke-linecap="round"/>'
    '<circle cx="172" cy="78" r="4.5" fill="#0A0F1A" stroke="#2DD4BF" stroke-width="2"/>'
    '<path d="M154 112 l-4 8 l4 2" stroke="#2DD4BF" stroke-width="2" fill="none" stroke-linecap="round"/></g>'
    '<g class="arm a4"><circle cx="208" cy="38" r="5" fill="#1E293B" stroke="#94A3B8"/>'
    '<line x1="208" y1="38" x2="216" y2="82" stroke="#94A3B8" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="216" y1="82" x2="194" y2="118" stroke="#CBD5E1" stroke-width="3.5" stroke-linecap="round"/>'
    '<circle cx="216" cy="82" r="4.5" fill="#0A0F1A" stroke="#2DD4BF" stroke-width="2"/>'
    '<path d="M194 118 l-5 9 M194 118 l3 10" stroke="#2DD4BF" stroke-width="2" stroke-linecap="round"/></g>'
    '<circle cx="130" cy="33" r="9" fill="#0A0F1A" stroke="#2DD4BF" stroke-width="2"/>'
    '<circle class="led" cx="130" cy="33" r="3" fill="#2DD4BF"/>'
    '</svg>'
)

# Small robot assistant head; SIZE is replaced with the pixel size.
ROBOT_HEAD = (
    '<svg class="bot-head" width="SIZE" height="SIZE" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" '
    'aria-hidden="true">'
    '<line x1="24" y1="5" x2="24" y2="11" stroke="#94A3B8" stroke-width="2"/>'
    '<circle class="bot-antenna" cx="24" cy="4" r="3" fill="#2DD4BF"/>'
    '<rect x="3" y="20" width="5" height="9" rx="2" fill="#334155"/>'
    '<rect x="40" y="20" width="5" height="9" rx="2" fill="#334155"/>'
    '<rect x="8" y="11" width="32" height="26" rx="8" fill="#1E293B" stroke="#2DD4BF" stroke-width="1.5"/>'
    '<g class="bot-eyes"><circle cx="18" cy="23" r="3.2" fill="#2DD4BF"/>'
    '<circle cx="30" cy="23" r="3.2" fill="#2DD4BF"/></g>'
    '<path d="M18 31 Q24 35 30 31" stroke="#94A3B8" stroke-width="1.8" fill="none" stroke-linecap="round"/>'
    '<rect x="16" y="38" width="16" height="6" rx="2" fill="#334155"/>'
    '</svg>'
)

# Wheeled helper robot that drives across the bottom of the page.
ROVER = (
    '<svg viewBox="0 0 64 64" width="60" height="60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<line x1="32" y1="6" x2="32" y2="12" stroke="#94A3B8" stroke-width="2"/>'
    '<circle class="bot-antenna" cx="32" cy="5" r="2.5" fill="#2DD4BF"/>'
    '<rect x="18" y="12" width="28" height="18" rx="6" fill="#1E293B" stroke="#2DD4BF" stroke-width="1.5"/>'
    '<rect x="22" y="17" width="20" height="7" rx="3.5" fill="#0A0F1A"/>'
    '<circle class="scan-eye" cx="27" cy="20.5" r="2.2" fill="#2DD4BF"/>'
    '<rect x="29" y="30" width="6" height="4" fill="#334155"/>'
    '<rect x="14" y="34" width="36" height="15" rx="4" fill="#162235" stroke="#334155"/>'
    '<path d="M32 37 v8 M28 41 h8" stroke="#2DD4BF" stroke-width="2.2" stroke-linecap="round"/>'
    '<g class="arm-wave"><line x1="50" y1="38" x2="58" y2="30" stroke="#CBD5E1" stroke-width="3" '
    'stroke-linecap="round"/><path d="M58 30 l3 -3 M58 30 l4 1" stroke="#2DD4BF" stroke-width="1.8" '
    'stroke-linecap="round"/></g>'
    '<g class="wheel"><circle cx="22" cy="55" r="6" fill="#334155" stroke="#94A3B8"/>'
    '<path d="M16 55 h12 M22 49 v12" stroke="#0A0F1A" stroke-width="1.5"/></g>'
    '<g class="wheel"><circle cx="42" cy="55" r="6" fill="#334155" stroke="#94A3B8"/>'
    '<path d="M36 55 h12 M42 49 v12" stroke="#0A0F1A" stroke-width="1.5"/></g>'
    '</svg>'
)

# Hovering scanner drone that patrols near the top of the page.
DRONE = (
    '<svg viewBox="0 0 80 56" width="72" height="50" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<line x1="26" y1="20" x2="10" y2="11" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>'
    '<line x1="54" y1="20" x2="70" y2="11" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>'
    '<ellipse class="rotor" cx="10" cy="9" rx="9" ry="2" fill="#CBD5E1"/>'
    '<ellipse class="rotor" cx="70" cy="9" rx="9" ry="2" fill="#CBD5E1"/>'
    '<polygon class="beam" points="40,32 26,56 54,56" fill="#2DD4BF"/>'
    '<ellipse cx="40" cy="24" rx="16" ry="9" fill="#1E293B" stroke="#2DD4BF" stroke-width="1.5"/>'
    '<circle class="led" cx="40" cy="24" r="4" fill="#2DD4BF"/>'
    '</svg>'
)

ROBOT_PARADE = (
    f'<div class="rover r1"><div class="bubble">Scanning for instruments…</div>{ROVER}</div>'
    f'<div class="rover r2"><div class="bubble">Drop a video or image above ⬆</div>{ROVER}</div>'
    f'<div class="drone">{DRONE}</div>'
)

st.set_page_config(page_title="SurgiVision Lite", page_icon="🩺", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model (the first run downloads ~22 MB of weights)...")
def get_model():
    return load_model()


def class_color_hex(class_id: int) -> str:
    """Same colour Ultralytics uses for this class's bounding boxes."""
    r, g, b = colors(class_id, bgr=False)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def feed_label(text: str, live: bool = False) -> None:
    rec = '<span class="rec">●</span> ' if live else ""
    st.markdown(f'<div class="feed-label">{rec}{text}</div>', unsafe_allow_html=True)


def robot_head(size: int) -> str:
    return ROBOT_HEAD.replace("SIZE", str(size))


def robot_says(html: str) -> None:
    """Empty-state / hint message delivered by the robot assistant."""
    st.markdown(f'<div class="bot-note">{robot_head(36)}<div>{html}</div></div>', unsafe_allow_html=True)


def render_header(model) -> None:
    st.markdown(
        f"""
        <div class="console-header">
          <div>
          <h1>🩺 SurgiVision Lite</h1>
          <p>Surgical instrument detection and video analysis for laparoscopic and robot-assisted surgery footage</p>
          <span class="pill"><span class="dot green"></span>Model online</span>
          <span class="pill"><span class="dot teal"></span>YOLOv8s · {len(model.names)} instrument classes</span>
          <span class="pill"><span class="dot teal"></span>Compute: {DEVICE.upper()}</span>
          <span class="pill">⚠️ Educational prototype · not for clinical use</span>
          </div>
          {ROBOT_HERO}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model) -> float:
    st.sidebar.markdown(
        f'<div class="bot-card">{robot_head(44)}<div><b>SV-1 Vision Assistant</b>'
        '<div class="bot-sub">Watching the endoscope feed for instruments. Tune my sensitivity below.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### ⚙️ Detection settings")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                             help="Detections below this confidence are hidden. Higher = fewer, surer detections.")

    st.sidebar.markdown("### 🛠️ Instrument legend")
    legend = "".join(
        f'<div class="legend-row"><span class="legend-swatch" style="background:{class_color_hex(i)}"></span>'
        f'<b>{name}</b></div><div class="legend-desc">{INSTRUMENT_INFO.get(name, "")}</div>'
        for i, name in model.names.items()
    )
    st.sidebar.markdown(legend, unsafe_allow_html=True)

    st.sidebar.markdown("### 📦 Model")
    st.sidebar.caption(
        f"`{MODEL_PATH.name}` · YOLOv8s by [cesaraha](https://huggingface.co/cesaraha/"
        "yolov8s-surgical-instrument-detection-cholec80), trained on Cholec80. "
        "CC BY-NC-SA 4.0, non-commercial use only."
    )
    return conf


def render_summary(summary: pd.DataFrame, empty_message: str) -> None:
    if summary.empty:
        robot_says(empty_message)
        return
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Detections per instrument**")
        st.bar_chart(summary.set_index("instrument")["detections"], color="#2DD4BF", height=260)
    with right:
        st.markdown("**Instrument statistics**")
        st.dataframe(
            summary, hide_index=True, width="stretch",
            column_config={
                "instrument": "Instrument",
                "detections": "Detections",
                "mean_confidence": st.column_config.ProgressColumn("Mean confidence", min_value=0, max_value=1,
                                                                   format="%.2f"),
                "min_confidence": st.column_config.NumberColumn("Min", format="%.2f"),
                "max_confidence": st.column_config.NumberColumn("Max", format="%.2f"),
                "frames_present": "Frames present",
                "visible_s": st.column_config.NumberColumn("On screen (s)", format="%.1f"),
            },
        )


def robot_report(detections: pd.DataFrame, total_frames: int | None = None) -> None:
    """Robot assistant's one-line verdict: instance counts for an image, frames present for a video."""
    if detections.empty:
        return
    if total_frames is None:
        counts = detections["instrument"].value_counts()
        found = " · ".join(f"<b>{name}</b> ×{n}" for name, n in counts.items())
        robot_says(f"Analysis complete! I spotted {found} in this frame.")
    else:
        frames = detections.groupby("instrument")["frame"].nunique().sort_values(ascending=False)
        found = " · ".join(f"<b>{name}</b> in {n}" for name, n in frames.items())
        robot_says(f"Analysis complete! Across {total_frames} frames I saw {found} frames.")


def image_analysis(model, conf: float, name: str, data: bytes, key: str) -> None:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("This file could not be read as an image. It may be corrupted.")
        return

    with st.spinner("🤖 Robots are scanning the frame..."):
        result = detect_image(model, image, conf)

    robot_report(result.detections)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Instruments detected", len(result.detections))
    m2.metric("Instrument types", result.detections["instrument"].nunique())
    m3.metric("Inference time", f"{result.inference_ms:.0f} ms")
    m4.metric("Total processing", f"{result.total_ms:.0f} ms")

    section("Endoscope view")
    left, right = st.columns(2)
    with left:
        feed_label("Input frame")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    with right:
        feed_label("AI detection overlay", live=True)
        st.image(cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")

    section("Analytics")
    render_summary(summarize_detections(result.detections),
                   "No instruments detected at this confidence threshold. Try lowering it in the sidebar.")
    if not result.detections.empty:
        with st.expander("Raw detections (bounding boxes in pixels)"):
            st.dataframe(result.detections.drop(columns=["frame", "time_s"]), hide_index=True, width="stretch")

    section("Export")
    stem = Path(name).stem.replace(" ", "_")
    ok, png = cv2.imencode(".png", result.annotated_bgr)
    d1, d2 = st.columns(2)
    if ok:
        d1.download_button("⬇️ Annotated image (PNG)", png.tobytes(), f"{stem}_detected.png", "image/png",
                           width="stretch", key=f"{key}_png")
    d2.download_button("⬇️ Detections (CSV)", result.detections.to_csv(index=False),
                       f"{stem}_detections.csv", "text/csv", width="stretch", key=f"{key}_image_csv")


def presence_timeline(detections: pd.DataFrame) -> pd.DataFrame:
    """Frames per second in which each instrument is visible (0 to fps)."""
    if detections.empty:
        return pd.DataFrame()
    per_second = detections.assign(second=detections["time_s"].astype(int))
    return (per_second.groupby(["second", "instrument"])["frame"].nunique()
            .unstack(fill_value=0).sort_index())


def video_analysis(model, conf: float, name: str, data: bytes | None, key: str) -> None:
    """Analyse an uploaded video (data) or the bundled demo clip (data=None)."""
    # Results are kept in session_state so download buttons (which rerun the script)
    # don't throw away a finished analysis.
    run_key = (name, len(data) if data else 0, conf)
    state_key = f"{key}_video_result"
    if st.button("▶️ Run analysis", type="primary", key=f"{key}_run"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if data is None:
            input_path, cleanup = DEMO_VIDEO, False
        else:
            input_path, cleanup = OUTPUT_DIR / f"input_{Path(name).name}", True
            input_path.write_bytes(data)
        progress = st.progress(0.0, text="Processing frames...")

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                progress.progress(min(done / total, 1.0), text=f"Analyzing frame {done} of {total}")

        try:
            result = process_video(model, input_path, OUTPUT_DIR / f"{Path(name).stem.replace(' ', '_')}_detected",
                                   conf, on_progress)
        except (ValueError, RuntimeError) as error:
            progress.empty()
            st.error(str(error))
            return
        finally:
            if cleanup:
                input_path.unlink(missing_ok=True)
        progress.empty()
        st.session_state[state_key] = (run_key, result)

    stored = st.session_state.get(state_key)
    if not stored or stored[0] != run_key:
        robot_says("Click <b>Run analysis</b> and I'll detect instruments in every frame. "
                   "On CPU, expect roughly a few frames per second.")
        return
    result = stored[1]
    counts = result.per_frame_counts
    frames_with_tools = int((counts["instruments"] > 0).sum())

    robot_report(result.detections, result.frames_processed)
    section("Processing performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frames processed", result.frames_processed)
    m2.metric("Video duration", f"{result.duration_s:.1f} s")
    m3.metric("Processing time", f"{result.processing_time_s:.1f} s")
    m4.metric("Processing FPS", f"{result.processing_fps:.1f}")
    m5.metric("Inference / frame", f"{result.mean_inference_ms:.0f} ms")

    section("Annotated surgical video")
    feed_label("AI detection overlay", live=True)
    st.video(str(result.output_path))

    section("Instrument activity")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames with ≥1 instrument", f"{frames_with_tools / len(counts):.0%}")
    c2.metric("Max instruments in a frame", int(counts["instruments"].max()))
    c3.metric("Mean instruments / frame", f"{counts['instruments'].mean():.2f}")

    timeline = presence_timeline(result.detections)
    if not timeline.empty:
        st.markdown("**Instrument presence timeline** (frames per second in which each instrument is visible)")
        palette = {name: class_color_hex(i) for i, name in model.names.items()}
        st.bar_chart(timeline, x_label="Time (s)", y_label="Frames visible",
                     color=[palette[c] for c in timeline.columns], height=280)
    st.markdown("**Instruments detected per frame**")
    st.line_chart(counts.set_index("time_s")["instruments"], x_label="Time (s)", y_label="Instruments",
                  color="#2DD4BF", height=220)

    render_summary(summarize_detections(result.detections, result.source_fps),
                   "No instruments detected in this video at this confidence threshold.")
    st.caption("Counts are per frame. Without tracking, an instrument seen in many frames is counted once per "
               "frame, so 'Detections' is not the number of unique instruments.")

    section("Export")
    d1, d2 = st.columns(2)
    mime = "video/mp4" if result.output_path.suffix == ".mp4" else "video/webm"
    d1.download_button("⬇️ Annotated video", result.output_path.read_bytes(), result.output_path.name, mime,
                       width="stretch", key=f"{key}_video")
    d2.download_button("⬇️ Detections (CSV)", result.detections.to_csv(index=False),
                       f"{result.output_path.stem}.csv", "text/csv", width="stretch", key=f"{key}_video_csv")


def upload_tab(model, conf: float) -> None:
    robot_says("Drop an endoscope <b>image</b> or a surgical <b>video</b> below and my robot team will find "
               "the instruments.")
    uploaded = st.file_uploader("Upload an image or video", type=IMAGE_TYPES + VIDEO_TYPES, key="upload")
    if not uploaded:
        st.caption(f"Images: {', '.join(IMAGE_TYPES)} · Videos: {', '.join(VIDEO_TYPES)} · "
                   "No file handy? Try the **Demo samples** tab.")
        return
    if Path(uploaded.name).suffix.lower().lstrip(".") in IMAGE_TYPES:
        image_analysis(model, conf, uploaded.name, uploaded.getvalue(), key="upload")
    else:
        video_analysis(model, conf, uploaded.name, uploaded.getvalue(), key="upload")


def demo_tab(model, conf: float) -> None:
    options = [name for name, path in DEMO_IMAGES.items() if path.exists()]
    if DEMO_VIDEO.exists():
        options.append(DEMO_VIDEO_NAME)
    if not options:
        robot_says("No demo files are bundled with this copy of the app.")
        return
    choice = st.radio("Demo sample", options, horizontal=True, key="demo_source")
    if choice == DEMO_VIDEO_NAME:
        video_analysis(model, conf, choice, None, key="demo")
    else:
        image_analysis(model, conf, choice, DEMO_IMAGES[choice].read_bytes(), key="demo")


def about_tab() -> None:
    section("How it works")
    steps = st.columns(4)
    for col, (title, text) in zip(steps, [
        ("1 · Input", "Endoscope image or recorded surgical video"),
        ("2 · Frames", "OpenCV decodes the video frame by frame"),
        ("3 · Detection", "YOLOv8 finds instruments, with class and confidence"),
        ("4 · Analytics", "Pandas computes counts, presence time and FPS"),
    ]):
        col.markdown(f"**{title}**  \n{text}")

    section("Why this matters in surgical robotics")
    st.markdown(
        "- In minimally invasive and robot-assisted surgery, the **endoscope video is the surgeon's main view** "
        "of the operation.\n"
        "- Knowing **where instruments are in the image** underpins features studied in surgical robotics, "
        "such as camera control that follows the instruments, alerts when a tool leaves the field of view, "
        "and video-based training and review.\n"
        "- Recorded procedures are used for **surgeon training and case review**. Automatic instrument "
        "statistics make long recordings easier to search and summarise."
    )

    section("Limitations")
    st.markdown(
        "- Trained on one procedure type (laparoscopic cholecystectomy, Cholec80). It is not trained on "
        "robotic instruments and would need retraining with such data.\n"
        "- No tracking: counts are per frame, not unique instruments.\n"
        "- Author-reported accuracy on the model card: mAP@50 0.685. Clipper and Scissors are weaker classes.\n"
        "- CPU inference, so video is processed slower than real time."
    )


def main() -> None:
    try:
        model = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    render_header(model)
    conf = render_sidebar(model)

    st.markdown(ROBOT_PARADE, unsafe_allow_html=True)

    upload, demo, about = st.tabs(["📤 Upload & analyze", "🧪 Demo samples", "ℹ️ About"])
    with upload:
        upload_tab(model, conf)
    with demo:
        demo_tab(model, conf)
    with about:
        about_tab()

    st.markdown(
        '<div class="footer">SurgiVision Lite · Educational prototype, not a medical device · '
        "Demo media: CholecSeg8k / Cholec80 (CC BY-NC-SA 4.0)</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
