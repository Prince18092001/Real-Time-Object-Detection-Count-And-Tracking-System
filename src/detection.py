from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

MODEL_DIR = Path("models")
PROTO_PATH = MODEL_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODEL_DIR / "mobilenet_iter_73000.caffemodel"

PROTO_URLS = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "https://cdn.jsdelivr.net/gh/chuanqi305/MobileNet-SSD/deploy.prototxt",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt",
)
WEIGHTS_URLS = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
)


@dataclass
class DetectionState:
    active_ids: set[int] = field(default_factory=set)
    counted_ids: set[int] = field(default_factory=set)
    tracks: dict[int, tuple[int, int]] = field(default_factory=dict)
    line_cross_count: int = 0
    next_track_id: int = 1


def _download_file(urls: tuple[str, ...], target_path: Path) -> None:
    errors: list[str] = []
    for url in urls:
        for _ in range(3):
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=30) as response, target_path.open("wb") as output:
                    output.write(response.read())
                if target_path.exists() and target_path.stat().st_size > 0:
                    return
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url}: {exc}")

    joined = "\n".join(errors[-6:])
    raise RuntimeError(
        f"Failed to download required model file '{target_path.name}'.\n"
        f"Please check internet access and retry.\n{joined}"
    )


def _ensure_model_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PROTO_PATH.exists():
        _download_file(PROTO_URLS, PROTO_PATH)
    if not WEIGHTS_PATH.exists():
        _download_file(WEIGHTS_URLS, WEIGHTS_PATH)


@lru_cache(maxsize=1)
def load_model() -> cv2.dnn_Net:
    _ensure_model_files()
    return cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(WEIGHTS_PATH))


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (52, 211, 153),
        (56, 189, 248),
        (251, 191, 36),
        (248, 113, 113),
        (167, 139, 250),
        (34, 197, 94),
    ]
    return palette[track_id % len(palette)]


def _assign_tracks(
    detections: list[tuple[int, int, int, int, str, float]],
    state: DetectionState,
    max_distance: int,
) -> list[tuple[int, int, int, int, str, float, int, tuple[int, int], tuple[int, int] | None]]:
    unmatched_track_ids = set(state.tracks.keys())
    current_tracks: dict[int, tuple[int, int]] = {}
    assigned = []

    for x1, y1, x2, y2, label, confidence in detections:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        best_track_id = None
        best_distance = float("inf")

        for track_id in unmatched_track_ids:
            previous = state.tracks[track_id]
            distance = np.hypot(center[0] - previous[0], center[1] - previous[1])
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is not None and best_distance <= max_distance:
            track_id = best_track_id
            unmatched_track_ids.remove(track_id)
        else:
            track_id = state.next_track_id
            state.next_track_id += 1

        previous_center = state.tracks.get(track_id)
        current_tracks[track_id] = center
        assigned.append((x1, y1, x2, y2, label, confidence, track_id, center, previous_center))

    state.tracks = current_tracks
    state.active_ids = set(current_tracks.keys())
    return assigned


def process_frame(
    frame: np.ndarray,
    model: cv2.dnn_Net,
    state: DetectionState,
    confidence: float,
    image_size: int,
    line_position: float,
    max_track_distance: int,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = frame.shape[:2]
    line_y = int(height * line_position)

    annotated = frame.copy()
    class_counts: Counter[str] = Counter()

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (image_size, image_size)),
        0.007843,
        (image_size, image_size),
        127.5,
    )
    model.setInput(blob)
    detections_tensor = model.forward()

    detections: list[tuple[int, int, int, int, str, float]] = []
    for i in range(detections_tensor.shape[2]):
        score = float(detections_tensor[0, 0, i, 2])
        if score < confidence:
            continue

        class_id = int(detections_tensor[0, 0, i, 1])
        if class_id < 0 or class_id >= len(CLASSES):
            continue

        box = detections_tensor[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append((x1, y1, x2, y2, CLASSES[class_id], score))

    assigned_tracks = _assign_tracks(detections=detections, state=state, max_distance=max_track_distance)

    for x1, y1, x2, y2, label, confidence_value, track_id, center, previous_center in assigned_tracks:
        if previous_center is not None and previous_center[1] < line_y <= center[1] and track_id not in state.counted_ids:
            state.counted_ids.add(track_id)
            state.line_cross_count += 1

        class_counts[label] += 1
        color = _color_for_id(track_id)
        caption = f"{label} #{track_id} {confidence_value:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, center, 4, color, -1)
        caption_width = max(len(caption) * 8, 120)
        cv2.rectangle(annotated, (x1, max(y1 - 26, 0)), (x1 + caption_width, y1), color, -1)
        cv2.putText(
            annotated,
            caption,
            (x1 + 4, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 23, 42),
            1,
            cv2.LINE_AA,
        )

    cv2.line(annotated, (0, line_y), (width, line_y), (0, 165, 255), 2)
    cv2.putText(
        annotated,
        f"Count line: {state.line_cross_count}",
        (12, max(line_y - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated, {
        "class_counts": class_counts,
        "frame_count": len(assigned_tracks),
    }
