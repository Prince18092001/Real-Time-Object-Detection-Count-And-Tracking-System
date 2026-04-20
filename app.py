from __future__ import annotations

import platform
from collections import Counter

import streamlit as st

CV2_IMPORT_ERROR: str | None = None
try:
    import cv2
except Exception as exc:  # pragma: no cover - environment specific
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)

DETECTION_IMPORT_ERROR: str | None = None
try:
    from src.detection import DetectionState, load_model, process_frame
except Exception as exc:  # pragma: no cover - environment specific
    DetectionState = None
    load_model = None
    process_frame = None
    DETECTION_IMPORT_ERROR = str(exc)


st.set_page_config(
    page_title="Real-Time Object Detection Count & Tracking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 24%),
                    linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
                color: #e5eefb;
            }
            [data-testid="stSidebar"] {
                background: rgba(8, 15, 30, 0.88);
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }
            .hero-card {
                background: rgba(15, 23, 42, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 24px;
                padding: 1.4rem 1.6rem;
                box-shadow: 0 20px 60px rgba(2, 6, 23, 0.35);
            }
            .metric-card {
                background: rgba(15, 23, 42, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 18px;
                padding: 1rem 1.1rem;
            }
            .stButton > button {
                width: 100%;
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%);
                color: white;
                font-weight: 700;
                padding: 0.65rem 1rem;
            }
            .stButton > button:hover {
                filter: brightness(1.06);
                border-color: rgba(255, 255, 255, 0.24);
            }
            .small-note {
                color: #94a3b8;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(stats: dict[str, int | float]) -> None:
    cols = st.columns(4)
    cards = [
        ("Objects in Frame", stats.get("frame_count", 0)),
        ("Tracked IDs", stats.get("tracked_ids", 0)),
        ("Crossing Count", stats.get("line_count", 0)),
        ("FPS", round(float(stats.get("fps", 0.0)), 1)),
    ]
    for column, (label, value) in zip(cols, cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:0.85rem;">{label}</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#f8fafc;line-height:1.15;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def supports_webcam_mode() -> bool:
    return platform.system() == "Windows"


def main() -> None:
    inject_styles()

    if CV2_IMPORT_ERROR is not None:
        st.error("OpenCV failed to import in this deployment environment.")
        st.code(CV2_IMPORT_ERROR)
        st.info("Install opencv-python-headless in requirements.txt and redeploy.")
        st.stop()

    if DETECTION_IMPORT_ERROR is not None:
        st.error("Detection module failed to import.")
        st.code(DETECTION_IMPORT_ERROR)
        st.info("Check OpenCV/Numpy dependency versions and redeploy.")
        st.stop()

    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:0.92rem;letter-spacing:0.08em;text-transform:uppercase;color:#38bdf8;font-weight:700;">Real-Time Vision Dashboard</div>
            <h1 style="margin:0.35rem 0 0.4rem 0;font-size:2.1rem;color:#f8fafc;">Real-Time Object Detection, Count & Tracking System</h1>
            <p style="margin:0;color:#cbd5e1;max-width:900px;">
                Open your camera directly from this dashboard and inspect live tracking results, object counts,
                and line-crossing analytics in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Run Controls")
    st.sidebar.subheader("Live Camera")
    camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    image_size = st.sidebar.selectbox("Inference size", [640, 768, 960], index=0)
    line_position = st.sidebar.slider("Counting line position", 0.2, 0.8, 0.5, 0.05)
    max_track_distance = st.sidebar.slider("Tracking distance", 20, 120, 60, 5)

    guidance = (
        "Webcam mode is available only when running this app on Windows."
        if not supports_webcam_mode()
        else "Press Start detection to open the camera directly. Ensure no other app is using it."
    )
    st.sidebar.markdown(
        f"""
        <div class="small-note">
        {guidance}
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_button = st.sidebar.button("Start detection")
    reset_button = st.sidebar.button("Reset counters")

    metrics_block = st.empty()

    if reset_button:
        st.session_state.pop("detection_state", None)
        st.session_state.pop("last_stats", None)
        st.session_state.pop("summary_data", None)

    state = st.session_state.get("detection_state")
    if not isinstance(state, DetectionState):
        state = DetectionState()
        st.session_state["detection_state"] = state

    last_stats = st.session_state.get("last_stats", {"frame_count": 0, "tracked_ids": 0, "line_count": 0, "fps": 0.0})
    with metrics_block.container():
        render_metrics(last_stats)

    left_col, right_col = st.columns([2.0, 1.0])

    with left_col:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    with right_col:
        st.markdown(
            """
            <div class="hero-card">
                <h3 style="margin-top:0;color:#f8fafc;">Dashboard Guide</h3>
                <ol style="color:#cbd5e1;line-height:1.75;padding-left:1.2rem;">
                    <li>Set camera index and detection parameters in the sidebar.</li>
                    <li>Press Start detection to open the camera feed.</li>
                    <li>Review the live frame, counts, IDs, and crossing analytics.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not run_button:
        st.info("Press Start detection to open the camera and run real-time tracking.")
        return

    try:
        model = load_model()
    except Exception as exc:
        st.error("Model download/load failed.")
        st.code(str(exc))
        st.info("Redeploy and retry. If this persists, check outbound internet access for GitHub/CDN URLs.")
        return

    if platform.system() == "Windows":
        capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(int(camera_index))
    if not capture.isOpened():
        st.error("Could not access the webcam. Close other apps using the camera and try again.")
        return

    try:
        status_placeholder.info("Detection running. Close the page or refresh to stop the current session.")
        frame_rate_counter = cv2.getTickCount()
        frame_counter = 0
        frame_histogram: Counter[str] = Counter()

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            annotated, stats = process_frame(
                frame=frame,
                model=model,
                state=state,
                confidence=confidence,
                image_size=image_size,
                line_position=line_position,
                max_track_distance=max_track_distance,
            )

            frame_counter += 1
            frame_histogram.update(stats["class_counts"])
            elapsed = (cv2.getTickCount() - frame_rate_counter) / cv2.getTickFrequency()
            fps = frame_counter / elapsed if elapsed > 0 else 0.0

            status_placeholder.caption(
                f"Camera: {int(camera_index)} | Frame: {frame_counter} | FPS: {fps:.1f} | Classes in frame: {dict(stats['class_counts'])}"
            )
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            st.session_state["last_stats"] = {
                "frame_count": stats["frame_count"],
                "tracked_ids": len(state.active_ids),
                "line_count": state.line_cross_count,
                "fps": fps,
            }
            with metrics_block.container():
                render_metrics(st.session_state["last_stats"])

        st.session_state["summary_data"] = dict(frame_histogram)
        st.success("Detection finished.")
    finally:
        capture.release()

    if "summary_data" in st.session_state and st.session_state["summary_data"]:
        st.subheader("Class summary")
        st.bar_chart(st.session_state["summary_data"])


if __name__ == "__main__":
    main()
