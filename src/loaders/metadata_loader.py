import json
from pathlib import Path


def load_metadata(recording_dir: Path) -> dict:
    metadata_path = recording_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata