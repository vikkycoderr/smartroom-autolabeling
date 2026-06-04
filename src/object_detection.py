import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def get_video_fps(video_path: Path) -> float:
    """
    Get the video's FPS so we can convert frame index to seconds.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")

    return fps


def run_yolo_on_video(
    video_path: str,
    output_path: str = "outputs/yolo_detections.jsonl",
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.50,
) -> list:
    """
    Run pretrained YOLO on a video.

    This saves one JSON line per detected object:
    - frame index
    - timestamp in seconds
    - object class
    - confidence
    - bounding box
    """

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = get_video_fps(video_path)

    # Free pretrained YOLO model.
    # yolov8n.pt is the small/nano pretrained detection model.
    model = YOLO(model_name)

    detections = []

    results = model.predict(
        source=str(video_path),
        conf=conf_threshold,
        stream=True,
        verbose=False,
    )

    frame_idx = 0

    with open(output_path, "w") as f:
        for result in results:
            time_sec = frame_idx / fps

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection = {
                        "frame_idx": frame_idx,
                        "time_sec": round(time_sec, 3),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox_xyxy": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ],
                    }

                    detections.append(detection)
                    f.write(json.dumps(detection) + "\n")

            frame_idx += 1

    return detections


def summarize_object_windows(
    detections: list,
    min_confidence: float = 0.25,
) -> list:
    """
    Summarize when each object class appears in the video.

    Example output:
    person appears from 3.2 sec to 7.8 sec.
    """

    grouped = defaultdict(list)

    for detection in detections:
        if detection["confidence"] >= min_confidence:
            grouped[detection["class_name"]].append(detection)

    summaries = []

    for class_name, items in grouped.items():
        times = [item["time_sec"] for item in items]
        confidences = [item["confidence"] for item in items]

        summaries.append(
            {
                "class_name": class_name,
                "first_seen_sec": min(times),
                "last_seen_sec": max(times),
                "num_detections": len(items),
                "avg_confidence": round(sum(confidences) / len(confidences), 4),
            }
        )

    return sorted(summaries, key=lambda x: x["first_seen_sec"])


def save_json(data, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    video_path = (
        "data/sample_dataset/day_01_2026-06-01/"
        "rec_20260601_003/streams/camera_main.mp4"
    )

    detections = run_yolo_on_video(
        video_path=video_path,
        output_path="outputs/yolo_detections.jsonl",
        model_name="yolov8n.pt",
        conf_threshold=0.50,
    )

    object_windows = summarize_object_windows(
        detections=detections,
        min_confidence=0.50,
    )

    save_json(object_windows, "outputs/yolo_object_windows.json")

    print(f"Saved {len(detections)} detections to outputs/yolo_detections.jsonl")
    print("Object time windows:")

    for window in object_windows:
        print(window)