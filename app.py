from __future__ import annotations

import platform
import time
from threading import Lock
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

WEBRTC_IMPORT_ERROR: str | None = None
try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception as exc:  # pragma: no cover - environment specific
    av = None
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = str(exc)


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
    camera_backend_options = ["Browser camera (WebRTC)"]
    if supports_webcam_mode():
        camera_backend_options.append("Device camera (OpenCV)")
    camera_backend = st.sidebar.selectbox("Camera backend", camera_backend_options)

    st.sidebar.subheader("Live Camera")
    camera_index = 0
    if camera_backend == "Device camera (OpenCV)":
        camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

    default_process_every_n = 3 if camera_backend == "Browser camera (WebRTC)" else 1
    process_every_n = st.sidebar.slider("Process every N frames", min_value=1, max_value=5, value=default_process_every_n, step=1)
    confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    image_size = st.sidebar.selectbox("Inference size", [640, 768, 960], index=0)
    line_position = st.sidebar.slider("Counting line position", 0.2, 0.8, 0.5, 0.05)
    max_track_distance = st.sidebar.slider("Tracking distance", 20, 120, 60, 5)

    guidance = (
        "Allow browser camera permission and click Start in the camera panel below. WebRTC runs at lower resolution/FPS for stability."
        if camera_backend == "Browser camera (WebRTC)"
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

    run_button = camera_backend == "Device camera (OpenCV)" and st.sidebar.button("Start detection")
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
                    <li>Select Browser camera (WebRTC) for cloud deployment.</li>
                    <li>Set camera index and detection parameters in the sidebar.</li>
                    <li>Use Start detection for OpenCV mode or Start in the WebRTC panel.</li>
                    <li>Review the live frame, counts, IDs, and crossing analytics.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if camera_backend == "Device camera (OpenCV)" and not run_button:
        st.info("Press Start detection to open the camera and run real-time tracking.")
        return

    try:
        model = load_model()
    except Exception as exc:
        st.error("Model download/load failed.")
        st.code(str(exc))
        st.info("Redeploy and retry. If this persists, check outbound internet access for GitHub/CDN URLs.")
        return

    if camera_backend == "Browser camera (WebRTC)":
        if WEBRTC_IMPORT_ERROR is not None:
            st.error("Browser camera dependencies failed to import.")
            st.code(WEBRTC_IMPORT_ERROR)
            st.info("Install streamlit-webrtc in requirements.txt and redeploy.")
            return

        webrtc_stats = {"frame_count": 0, "tracked_ids": 0, "line_count": 0, "fps": 0.0}
        stats_lock = Lock()
        fps_state = {"last_ts": time.perf_counter()}
        runtime_state = {"frame_idx": 0, "last_error": ""}

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            runtime_state["frame_idx"] += 1
            should_process = runtime_state["frame_idx"] % process_every_n == 0

            try:
                if should_process:
                    annotated, stats = process_frame(
                        frame=image,
                        model=model,
                        state=state,
                        confidence=confidence,
                        image_size=image_size,
                        line_position=line_position,
                        max_track_distance=max_track_distance,
                    )
                else:
                    annotated = image
                    stats = {"frame_count": 0}

                now = time.perf_counter()
                elapsed = now - fps_state["last_ts"]
                fps_state["last_ts"] = now
                fps = (1.0 / elapsed) if elapsed > 0 else 0.0

                with stats_lock:
                    webrtc_stats["frame_count"] = int(stats.get("frame_count", 0))
                    webrtc_stats["tracked_ids"] = len(state.active_ids)
                    webrtc_stats["line_count"] = state.line_cross_count
                    webrtc_stats["fps"] = fps
                    runtime_state["last_error"] = ""

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as exc:  # pragma: no cover - runtime specific
                with stats_lock:
                    runtime_state["last_error"] = str(exc)
                return av.VideoFrame.from_ndarray(image, format="bgr24")

        with left_col:
            webrtc_ctx = webrtc_streamer(
                key="realtime-object-detection-webrtc",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 360},
                        "frameRate": {"ideal": 15, "max": 20},
                    },
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
                    ]
                },
                video_frame_callback=video_frame_callback,
                video_html_attrs={"autoPlay": True, "controls": False, "muted": True},
                async_processing=True,
            )

        if webrtc_ctx.state.playing:
            status_placeholder.caption("Browser camera is live. Detection overlay is running in real time.")
            with stats_lock:
                with metrics_block.container():
                    render_metrics(webrtc_stats)
                if runtime_state["last_error"]:
                    st.warning(f"Frame processing warning: {runtime_state['last_error']}")
        else:
            status_placeholder.info("Click Start in the WebRTC panel to begin real-time detection.")
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
