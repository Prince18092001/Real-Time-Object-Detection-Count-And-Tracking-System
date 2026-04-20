# Real-Time Object Detection, Count & Tracking System

This project provides a Streamlit dashboard for running real-time object detection and tracking with YOLO.

## Features

- Streamlit dashboard with source selection
- Webcam and uploaded video input options
- Real-time object detection and tracking
- Per-class counts and track IDs
- Line-crossing counter for simple counting analytics

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
streamlit run app.py
```

## Notes

- The first run may download YOLO weights automatically.
- On Windows, webcam access usually works best when no other app is using the camera.
