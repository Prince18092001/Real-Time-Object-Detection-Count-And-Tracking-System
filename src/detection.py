from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionState:
    active_ids: set[int] = field(default_factory=set)
    counted_ids: set[int] = field(default_factory=set)
    last_centers: dict[int, tuple[int, int]] = field(default_factory=dict)
    line_cross_count: int = 0


@lru_cache(maxsize=1)
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    return YOLO(model_name)


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


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    state: DetectionState,
    confidence: float,
    image_size: int,
    line_position: float,
    tracker_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = frame.shape[:2]
    line_y = int(height * line_position)

    results = model.track(
        frame,
        persist=True,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
        tracker=tracker_name,
    )

    annotated = frame.copy()
    class_counts: Counter[str] = Counter()
    current_ids: set[int] = set()

    if not results:
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 165, 255), 2)
        return annotated, {"class_counts": class_counts, "frame_count": 0}

    result = results[0]
    boxes = result.boxes
    names = result.names if hasattr(result, "names") else model.names

    if boxes is not None and len(boxes) > 0:
        ids = boxes.id
        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            cls_id = int(boxes.cls[index].item())
            label = names.get(cls_id, str(cls_id))
            confidence_value = float(boxes.conf[index].item())

            if ids is not None and ids[index] is not None:
                track_id = int(ids[index].item())
                current_ids.add(track_id)
                color = _color_for_id(track_id)
            else:
                track_id = -1
                color = (148, 163, 184)

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            center_x = int((x1_i + x2_i) / 2)
            center_y = int((y1_i + y2_i) / 2)

            if track_id != -1:
                previous_center = state.last_centers.get(track_id)
                if previous_center is not None and previous_center[1] < line_y <= center_y and track_id not in state.counted_ids:
                    state.counted_ids.add(track_id)
                    state.line_cross_count += 1
                state.last_centers[track_id] = (center_x, center_y)

            class_counts[label] += 1
            caption = f"{label} #{track_id if track_id != -1 else '?'} {confidence_value:.2f}"

            cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), color, 2)
            cv2.circle(annotated, (center_x, center_y), 4, color, -1)
            caption_width = max(len(caption) * 8, 120)
            cv2.rectangle(annotated, (x1_i, max(y1_i - 26, 0)), (x1_i + caption_width, y1_i), color, -1)
            cv2.putText(
                annotated,
                caption,
                (x1_i + 4, max(y1_i - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 23, 42),
                1,
                cv2.LINE_AA,
            )

    state.active_ids = current_ids
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
        "frame_count": len(boxes) if boxes is not None else 0,
    }
