from pathlib import Path


def load_video_stream(recording_dir: Path, stream_id: str, stream_info: dict) -> dict:
    """
    Week 1 video loader.

    Uses metadata to:
    1. Find the video file path
    2. Check if the file exists
    3. Open the video with OpenCV
    4. Read actual FPS, frame count, and resolution
    """

    relative_path = stream_info.get("path")
    full_path = recording_dir / relative_path

    result = {
        "stream_id": stream_id,
        "modality": stream_info.get("modality"),
        "relative_path": relative_path,
        "full_path": str(full_path),
        "exists": full_path.exists(),

        "codec_metadata": stream_info.get("codec"),
        "fps_metadata": stream_info.get("fps"),
        "frame_count_metadata": stream_info.get("frame_count"),
        "resolution_metadata": stream_info.get("resolution"),
        "sync_offset_ms": stream_info.get("sync_offset_ms"),

        "status": "missing_file"
    }

    if not full_path.exists():
        return result

    try:
        import cv2
    except ImportError:
        result["status"] = "loaded_without_opencv"
        result["note"] = "OpenCV is not installed. File exists, but video properties were not checked."
        return result

    cap = cv2.VideoCapture(str(full_path))

    if not cap.isOpened():
        result["status"] = "cannot_open_video"
        return result

    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    frame_count_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    result.update({
        "fps_actual": fps_actual,
        "frame_count_actual": frame_count_actual,
        "resolution_actual": [width_actual, height_actual],
        "status": "loaded"
    })

    return result