from pathlib import Path

from loaders.metadata_loader import load_metadata
from loaders.video_loader import load_video_stream


def main():
    recording_dir = Path(
        "data/sample_dataset/day_01_2026-06-01/rec_20260601_003"
    )

    metadata = load_metadata(recording_dir)

    stream_id = "camera_main"
    stream_info = metadata["streams"][stream_id]

    video_result = load_video_stream(
        recording_dir=recording_dir,
        stream_id=stream_id,
        stream_info=stream_info
    )

    print("Metadata loaded:")
    print({
        "recording_id": metadata.get("recording_id"),
        "scenario_id": metadata.get("scenario_id"),
        "space": metadata.get("space"),
        "privacy_level": metadata.get("privacy_level")
    })

    print("\nVideo loaded:")
    print(video_result)


if __name__ == "__main__":
    main()