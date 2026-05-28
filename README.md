# WINLAB Smart Room AutoLabeling

## Week 1 Goals

This repository implements the **AutoLabeling layer** for the WINLAB Smart Room project. Week 1 focuses on:

1. **Define a provisional dataset directory schema** – Establish a standardized layout for recording folders with metadata and multi-modal streams.
2. **Load one modality plus metadata** – Successfully ingest video (camera_main) and associated metadata from a sample recording.
3. **Keep the loader flexible and adaptable** – Design an adapter/loader layer that can accommodate future changes to the raw input format from the upstream Smart Room data team.
4. **Convert loaded data into evidence artifacts** – Prepare the foundation for transforming loaded recordings into evidence artifacts for downstream analysis.

## Project Context

The upstream Smart Room data team sends recording folders containing:
- `metadata.json` – Recording metadata and stream definitions
- Stream files – Video, audio, environmental sensors, motion sensors, etc.

Because the raw input format may evolve, this repo uses an **adapter pattern** to abstract away format-specific logic.

## Repository Structure

```
smartroom-autolabeling/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_dataset/
│       └── day_01_2026-06-01/
│           └── rec_20260601_003/
│               ├── metadata.json
│               └── streams/
├── src/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   └── recording_folder_adapter.py
│   ├── loaders/
│   │   ├── metadata_loader.py
│   │   └── video_loader.py
│   ├── schemas/
│   │   ├── loaded_packet.py
│   │   └── evidence_artifact.py
│   └── main.py
└── outputs/
```

## Adapter Pattern

The **adapter layer** maps recording folder paths and metadata to stream resources:

- **base_adapter.py** – Abstract interface for adapters
- **recording_folder_adapter.py** – Concrete adapter for filesystem-based recordings

Future adapters could support cloud storage, databases, or different folder layouts.

## Loader Layer

Loaders handle modality-specific ingestion:

- **metadata_loader.py** – Parses `metadata.json`
- **video_loader.py** – Loads video file; optionally extracts basic info (duration, frame count, codec) using OpenCV

## Schemas

- **loaded_packet.py** – Dataclass for loaded recording data
- **evidence_artifact.py** – Evidence artifact schema (future use) with fields for claims, confidence, privacy level, QC status, etc.

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Sample

```bash
python src/main.py
```

This loads the sample recording (`data/sample_dataset/day_01_2026-06-01/rec_20260601_003/`) and writes output to `outputs/loaded_packet.json`.

## Next Steps

- Add dummy stream files (e.g., placeholder video, CSV) to the sample dataset
- Extend loaders for audio, environmental, and motion modalities
- Implement evidence artifact generation from loaded packets
- Add unit tests and validation
