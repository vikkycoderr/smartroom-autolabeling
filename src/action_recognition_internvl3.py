import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """
    Build image preprocessing transform for InternVL-style models.
    """

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    """
    Pick the closest tile layout for dynamic image preprocessing.
    """

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio

        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=12,
    image_size=448,
    use_thumbnail=True,
):
    """
    Split image into tiles for InternVL visual input.
    """

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        original_width,
        original_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))

    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )

        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def sample_video_frames(
    video_path: str,
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[Image.Image]:
    """
    Sample evenly spaced frames from a video.

    This avoids sending the entire MP4 directly to the VLM.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata. fps={fps}, frames={frame_count}")

    duration_sec = frame_count / fps

    if start_sec is None:
        start_sec = 0.0

    if end_sec is None:
        end_sec = duration_sec

    start_sec = max(0.0, start_sec)
    end_sec = min(duration_sec, end_sec)

    if end_sec <= start_sec:
        cap.release()
        raise ValueError(f"Invalid clip range: {start_sec} to {end_sec}")

    frame_times = np.linspace(start_sec, end_sec, num_frames)

    frames = []

    for time_sec in frame_times:
        frame_idx = int(round(time_sec * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()

        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    cap.release()

    if not frames:
        raise ValueError("No frames could be sampled from the video.")

    return frames


def load_video_as_pixel_values(
    video_path: str,
    input_size: int = 448,
    max_num: int = 4,
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    """
    Convert sampled video frames into InternVL pixel_values.

    Returns:
    - pixel_values tensor
    - num_patches_list for each frame
    """

    transform = build_transform(input_size=input_size)

    frames = sample_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    pixel_values_list = []
    num_patches_list = []

    for frame in frames:
        tiles = dynamic_preprocess(
            frame,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )

        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)

        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    pixel_values = torch.cat(pixel_values_list, dim=0)

    return pixel_values, num_patches_list, len(frames)


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from model response.
    """

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {
            "event_label": "unknown_activity",
            "event_description": text,
            "confidence": 0.0,
            "qc_status": "needs_review",
            "failure_tags": ["json_parse_failed"],
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "event_label": "unknown_activity",
            "event_description": text,
            "confidence": 0.0,
            "qc_status": "needs_review",
            "failure_tags": ["json_parse_failed"],
        }


def load_internvl3_model(model_id: str):
    """
    Load InternVL3 model and tokenizer.

    Start with InternVL3-1B or InternVL3-2B.
    Larger models need much more GPU memory.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Do not run InternVL3 on CPU for this project. "
            "Use a GPU node where nvidia-smi works."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().cuda()

    return model, tokenizer


def run_internvl3_action_detection(
    video_path: str,
    model,
    tokenizer,
    candidate_events: list[str],
    num_frames: int = 8,
    max_tiles_per_frame: int = 4,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict:
    """
    Open-vocabulary action/event detection using InternVL3.

    This samples frames and asks the VLM to identify the visible event.
    """

    pixel_values, num_patches_list, sampled_frame_count = load_video_as_pixel_values(
        video_path=video_path,
        input_size=448,
        max_num=max_tiles_per_frame,
        num_frames=num_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    pixel_values = pixel_values.to(torch.float16).cuda()

    candidate_text = "\n".join(f"- {event}" for event in candidate_events)

    frame_tokens = ""
    for i in range(sampled_frame_count):
        frame_tokens += f"Frame{i + 1}: <image>\n"

    question = f"""
{frame_tokens}

You are an open-vocabulary action/event recognition model.

Analyze these sampled video frames in temporal order and identify the main visible event.

Candidate events:
{candidate_text}

Instructions:
- Choose the most visually supported event.
- If the exact event is not listed, create a short natural-language event label.
- Do not guess tiny object interactions unless clearly visible.
- If the frames are unclear, return "unknown_activity".
- Return ONLY valid JSON.
- Do not include markdown or explanations outside JSON.

Required JSON format:
{{
  "event_label": "short_event_label",
  "event_description": "one sentence describing what happens",
  "confidence": 0.0,
  "evidence": {{
    "visible_subjects": [],
    "motion_cues": [],
    "object_cues": []
  }},
  "qc_status": "pass_or_needs_review",
  "failure_tags": []
}}
""".strip()

    generation_config = {
        "max_new_tokens": 512,
        "do_sample": False,
    }

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

    parsed_event = extract_json_from_text(response)

    return {
        "raw_model_output": response,
        "parsed_event": parsed_event,
        "sampled_frame_count": sampled_frame_count,
        "num_patches_list": num_patches_list,
    }


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

    # Start small. Try 2B later if 1B works.
    model_id = "OpenGVLab/InternVL3-1B"

    candidate_events = [
        "a person is walking through the room",
        "a person is sitting down",
        "a person is standing up",
        "a person is using a computer",
        "a person is picking up an object",
        "a person is taking an object out of their pocket",
        "a person is leaving the room",
        "a car is moving",
        "a car is stopped",
        "a pedestrian is crossing the street",
        "no important activity is happening",
        "unknown_activity",
    ]

    print(f"Loading model: {model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model, tokenizer = load_internvl3_model(model_id)

    result = run_internvl3_action_detection(
        video_path=video_path,
        model=model,
        tokenizer=tokenizer,
        candidate_events=candidate_events,
        num_frames=8,
        max_tiles_per_frame=4,
        start_sec=None,
        end_sec=None,
    )

    final_result = {
        "experiment": "internvl3_open_vocabulary_action_detection",
        "model_id": model_id,
        "video_path": video_path,
        "candidate_events": candidate_events,
        "internvl3_result": result,
    }

    output_path = "outputs/internvl3_action_detection_result.json"
    save_json(final_result, output_path)

    print("InternVL3 action detection result:")
    print(json.dumps(final_result, indent=2))

    print(f"Saved result to {output_path}")