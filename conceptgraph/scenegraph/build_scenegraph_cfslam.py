"""Build a scene graph from a segment-based map and multi-view captions."""

import base64
import gc
import gzip
import hashlib
import json
import math
import os
import pickle as pkl
import stat
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Union
from textwrap import wrap
from urllib.parse import urlsplit
from conceptgraph.utils.general_utils import prjson

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm
from transformers import logging as hf_logging

# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()

# OpenAI-compatible API settings. Keep credentials out of source control and
# provide them through a private file (preferred) or the environment at runtime.
from openai import OpenAI


def validate_openai_base_url(value: str) -> str:
    """Reject insecure or credential-bearing API base URLs."""
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OPENAI_BASE_URL must not contain credentials, a query, or a fragment")
    return value.rstrip("/")


def read_float_environment_default(name: str, fallback: float) -> float:
    """Return a numeric environment default without blocking CLI overrides."""
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback
    return value if math.isfinite(value) else fallback


def read_int_environment_default(name: str, fallback: int) -> int:
    """Return an integer environment default without blocking CLI overrides."""
    try:
        return int(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://www.autodl.art/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
OPENAI_TIMEOUT = read_float_environment_default("OPENAI_TIMEOUT", 120.0)
OPENAI_MAX_RETRIES = read_int_environment_default("OPENAI_MAX_RETRIES", 0)
OPENAI_API_KEY_FILE = os.getenv("OPENAI_API_KEY_FILE")


def read_openai_api_key(api_key_file: Union[str, None] = None) -> str:
    """Read a credential from a private file, with an environment fallback.

    A file path is safe to expose as a CLI argument; the credential itself is
    not.  Explicit file configuration takes precedence over OPENAI_API_KEY.
    """
    configured_file = api_key_file or os.getenv("OPENAI_API_KEY_FILE")
    if configured_file:
        path = Path(configured_file).expanduser()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to open the configured OpenAI API key file ({type(exc).__name__})"
            ) from None
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError("OPENAI API key file must be a regular file")
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                raise RuntimeError(
                    "OPENAI API key file must not be readable or writable by group/others"
                )
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                fd = -1
                api_key = stream.read(65537)
        finally:
            if fd >= 0:
                os.close(fd)
        if len(api_key) > 65536:
            raise RuntimeError("OPENAI API key file is unexpectedly large")
        api_key = api_key.strip()
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is configured. Use --openai-api-key-file, set "
            "OPENAI_API_KEY_FILE, or export OPENAI_API_KEY."
        )
    if any(character.isspace() for character in api_key):
        raise RuntimeError("OpenAI API key configuration must contain exactly one token")
    return api_key


def make_openai_client(
    *,
    api_key_file: Union[str, None] = None,
    base_url: str = OPENAI_BASE_URL,
    max_retries: int = OPENAI_MAX_RETRIES,
) -> OpenAI:
    if max_retries < 0:
        raise ValueError("OPENAI_MAX_RETRIES must be non-negative")
    return OpenAI(
        api_key=read_openai_api_key(api_key_file),
        base_url=validate_openai_base_url(base_url),
        max_retries=max_retries,
    )


def request_openai_text(
    client: OpenAI,
    messages: list[dict],
    timeout: float,
    model: str = OPENAI_MODEL,
) -> str:
    """Call the OpenAI-compatible Responses API and return its text output."""
    safe_error_message = None
    try:
        response = client.responses.create(
            model=model,
            input=messages,
            store=False,
            timeout=timeout,
        )
        output_text = response.output_text or ""
    except Exception as exc:
        # SDK exceptions can retain the request object, including Base64 image
        # payloads. Retain only scalar safe fields, then leave the except block
        # before raising so the new exception has no hidden __context__ link.
        status_code = getattr(exc, "status_code", None)
        status_suffix = f", HTTP {status_code}" if isinstance(status_code, int) else ""
        safe_error_message = (
            f"OpenAI Responses request failed ({type(exc).__name__}{status_suffix}); "
            "request and response bodies were omitted"
        )
    if safe_error_message is not None:
        raise RuntimeError(safe_error_message)
    return output_text


@dataclass
class ProgramArgs:
    mode: Literal[
        "extract-node-captions",
        "refine-node-captions",
        "build-scenegraph",
        "generate-scenegraph-json",
        "annotate-scenegraph",
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Private credential file path; never pass the credential itself here
    openai_api_key_file: Union[str, None] = OPENAI_API_KEY_FILE

    # OpenAI-compatible API root URL
    openai_base_url: str = OPENAI_BASE_URL

    # Text model for node refinement and relation inference
    openai_model: str = OPENAI_MODEL

    # Vision model for per-view object captions
    openai_vision_model: Union[str, None] = OPENAI_VISION_MODEL

    # Per-request timeout in seconds
    openai_timeout: float = OPENAI_TIMEOUT

    # SDK retry count; zero avoids hidden duplicate paid requests
    openai_max_retries: int = OPENAI_MAX_RETRIES

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Device to use
    device: str = "cuda:0"

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Candidate generator. The legacy mode preserves the original 3D MST.
    relation_mode: Literal["legacy-3d-mst", "multiview-2d-3d"] = "legacy-3d-mst"

    # Maximum hybrid candidates; hybrid mode fails instead of truncating.
    max_relation_candidates: int = 100

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 4

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "red_outline"

    # Image detail sent to the OpenAI Responses vision model
    openai_image_detail: Literal["low", "high", "auto", "original"] = "high"

    # Resize each crop so its longest side is at most this many pixels
    openai_image_max_size: int = 1024

    # JPEG quality used for the in-memory Base64 image payload
    openai_jpeg_quality: int = 90

    # Save local red-outline crop grids for visual inspection
    save_caption_debug: bool = False

    # Print prompts and model responses during caption refinement
    print_openai_responses: bool = False


def validate_openai_runtime_args(
    args,
    *,
    require_text_model: bool = False,
    require_vision_model: bool = False,
) -> None:
    """Validate and normalize the configurable provider settings in-place."""
    args.openai_base_url = validate_openai_base_url(args.openai_base_url)
    args.openai_model = (args.openai_model or "").strip()
    args.openai_vision_model = (args.openai_vision_model or args.openai_model).strip()
    if require_text_model and not args.openai_model:
        raise ValueError("openai_model must not be empty")
    if require_vision_model and not args.openai_vision_model:
        raise ValueError("openai_vision_model must not be empty")
    if not math.isfinite(args.openai_timeout) or args.openai_timeout <= 0:
        raise ValueError("openai_timeout must be positive")
    if args.openai_max_retries < 0:
        raise ValueError("openai_max_retries must be non-negative")


def make_openai_client_from_args(args) -> OpenAI:
    return make_openai_client(
        api_key_file=args.openai_api_key_file,
        base_url=args.openai_base_url,
        max_retries=args.openai_max_retries,
    )


def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            serialized = list(loaded_data["objects"] or [])
            if getattr(args, "relation_mode", "legacy-3d-mst") == "multiview-2d-3d":
                serialized.extend(list(loaded_data.get("bg_objects") or []))
            scene_map.load_serializable(serialized)
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")
        return loaded_data



def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """Draw a red outline around the masked object."""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        raise ValueError("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)
    
    return image_crop, mask_crop

def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """Save a debug grid showing exactly which crops were captioned."""
    
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    output_buffer = BytesIO()
    try:
        fig.tight_layout()
        fig.savefig(output_buffer, format="png")
    finally:
        plt.close(fig)
    save_bytes_atomic(output_buffer.getvalue(), savedir / f"{idx_obj}.png")



def pil_image_to_data_url(image: Image.Image, max_size: int, jpeg_quality: int) -> str:
    """Encode a PIL image as an in-memory JPEG data URL."""
    if max_size <= 0:
        raise ValueError("openai_image_max_size must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("openai_jpeg_quality must be between 1 and 100")

    image = image.convert("RGB")
    if max(image.size) > max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def select_caption_detections(obj: dict, max_views: int) -> list[dict]:
    """Select valid, temporally diverse object views for visual captioning."""
    if max_views <= 0:
        raise ValueError("max_detections_per_object must be positive")

    required_fields = ("color_path", "mask", "xyxy", "conf")
    if any(field not in obj for field in required_fields):
        missing = [field for field in required_fields if field not in obj]
        raise KeyError(f"Object is missing caption fields: {missing}")

    num_detections = min(len(obj[field]) for field in required_fields)
    candidates_by_path = {}

    for idx_det in range(num_detections):
        image_path = Path(str(obj["color_path"][idx_det]))
        if not image_path.is_file():
            continue

        mask = np.asarray(obj["mask"][idx_det], dtype=bool)
        xyxy = np.asarray(obj["xyxy"][idx_det], dtype=float).reshape(-1)
        if mask.ndim != 2 or xyxy.size != 4 or not np.isfinite(xyxy).all():
            continue

        image_height, image_width = mask.shape
        x1 = max(0, int(np.floor(xyxy[0])))
        y1 = max(0, int(np.floor(xyxy[1])))
        x2 = min(image_width, int(np.ceil(xyxy[2])))
        y2 = min(image_height, int(np.ceil(xyxy[3])))
        bbox_width, bbox_height = x2 - x1, y2 - y1
        bbox_area = bbox_width * bbox_height
        if bbox_area <= 0:
            continue

        mask_area = int(mask[y1:y2, x1:x2].sum())
        mask_fill_ratio = mask_area / bbox_area
        if mask_area < 100 or mask_fill_ratio < 0.1:
            continue

        padding = int(np.clip(round(0.2 * max(bbox_width, bbox_height)), 20, 50))
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(image_width, x2 + padding)
        crop_y2 = min(image_height, y2 + padding)
        crop_width, crop_height = crop_x2 - crop_x1, crop_y2 - crop_y1
        # Thin structural surfaces (for example a ceiling strip at the image
        # boundary) can be valid even when one crop dimension is modest.
        if min(crop_width, crop_height) < 48 or crop_width * crop_height < 70 * 70:
            continue

        confidence = float(np.asarray(obj["conf"][idx_det]).reshape(-1)[0])
        if not np.isfinite(confidence):
            continue
        quality = confidence * mask_area

        if "image_idx" in obj and idx_det < len(obj["image_idx"]):
            try:
                frame_order = int(np.asarray(obj["image_idx"][idx_det]).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                frame_order = idx_det
        else:
            frame_order = idx_det

        candidate = {
            "idx_det": idx_det,
            "image_path": str(image_path),
            "xyxy": [float(value) for value in xyxy],
            "padding": padding,
            "confidence": confidence,
            "mask_area": mask_area,
            "mask_fill_ratio": mask_fill_ratio,
            "quality": quality,
            "frame_order": frame_order,
        }

        # A map can contain duplicate detections from the same frame. Keep the
        # view with the largest confidence-weighted visible mask.
        previous = candidates_by_path.get(str(image_path))
        if previous is None or candidate["quality"] > previous["quality"]:
            candidates_by_path[str(image_path)] = candidate

    candidates = sorted(
        candidates_by_path.values(),
        key=lambda item: (item["frame_order"], item["idx_det"]),
    )
    if len(candidates) <= max_views:
        return candidates

    # Split the trajectory into temporal bins and take the best visible view
    # from each bin. This avoids selecting only adjacent high-confidence frames.
    selected = []
    for indices in np.array_split(np.arange(len(candidates)), max_views):
        selected.append(max((candidates[int(i)] for i in indices), key=lambda item: item["quality"]))
    return sorted(selected, key=lambda item: (item["frame_order"], item["idx_det"]))


def save_bytes_atomic(value: bytes, filename):
    """Atomically create a private file without a world-readable write window."""
    filename = Path(filename)
    filename.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{filename.name}.",
        suffix=".tmp",
        dir=str(filename.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
        os.chmod(filename, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def save_json_atomic(value, filename):
    """Atomically save private JSON checkpoints for interrupted API runs."""
    payload = json.dumps(value, indent=4, sort_keys=False).encode("utf-8")
    save_bytes_atomic(payload, filename)


def save_pickle_atomic(value, filename):
    """Atomically save a private pickle used by the legacy downstream stage."""
    save_bytes_atomic(pkl.dumps(value), filename)


def sha256_file(filename: Path) -> str:
    """Return a streaming SHA256 digest for cache invalidation."""
    digest = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_openai_caption(cache_file: Path, request_spec: dict) -> Union[str, None]:
    """Return a compatible cached caption, or None when it must be regenerated."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    caption = cached.get("caption")
    if cached.get("request") != request_spec or not isinstance(caption, str) or not caption.strip():
        return None
    return caption.strip()


def is_valid_caption_entry(entry, expected_id: int) -> bool:
    """Validate the object-level schema consumed by caption refinement."""
    if not isinstance(entry, dict) or entry.get("id") != expected_id:
        return False
    captions = entry.get("captions")
    low_confidences = entry.get("low_confidences")
    if not isinstance(captions, list) or not captions or not isinstance(low_confidences, list):
        return False
    if len(captions) != len(low_confidences):
        return False
    if not all(isinstance(caption, str) and caption.strip() for caption in captions):
        return False
    return all(isinstance(value, bool) for value in low_confidences)


def make_openai_caption_prompt(masking_option: str) -> str:
    """Return the provider-neutral prompt whose hash invalidates view caches."""
    marker_hint = {
        "red_outline": "The target object is enclosed by a red outline.",
        "blackout": "Only the target object remains visible; the surrounding pixels are black.",
        "none": "The target object is centered in the crop.",
    }[masking_option]
    return (
        "Describe only the target indoor object in one concise English sentence. "
        f"{marker_hint} Name the object type and visible attributes that help identify it. "
        "Do not mention the image, crop, mask, outline, or background. "
        "If identification is genuinely uncertain, say 'unclear indoor object' and describe its visible attributes."
    )


def request_openai_image_caption(
    client: OpenAI,
    image: Image.Image,
    args,
    image_url: Union[str, None] = None,
) -> str:
    """Caption one marked object crop with an OpenAI-compatible vision model."""
    prompt = make_openai_caption_prompt(args.masking_option)
    if image_url is None:
        image_url = pil_image_to_data_url(
            image,
            max_size=args.openai_image_max_size,
            jpeg_quality=args.openai_jpeg_quality,
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": args.openai_image_detail,
                },
            ],
        }
    ]
    caption = request_openai_text(
        client,
        messages,
        args.openai_timeout,
        model=args.openai_vision_model,
    )
    caption = " ".join(caption.strip().split())
    if len(caption) >= 2 and caption[0] == caption[-1] == '"':
        caption = caption[1:-1].strip()
    if caption.lower().startswith("caption:"):
        caption = caption[len("caption:"):].strip()
    if not caption:
        raise ValueError("OpenAI vision response did not contain a caption")
    return caption


def extract_node_captions(args):
    """Generate per-view object captions with the OpenAI Responses vision API."""
    from conceptgraph.slam.slam_classes import MapObjectList

    validate_openai_runtime_args(args, require_vision_model=True)
    client = None
    if args.openai_image_max_size <= 0:
        raise ValueError("openai_image_max_size must be positive")
    if not 1 <= args.openai_jpeg_quality <= 100:
        raise ValueError("openai_jpeg_quality must be between 1 and 100")

    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    cache_root = Path(args.cachedir)
    savedir_captions = cache_root / "cfslam_captions_openai"
    savedir_views = savedir_captions / "views"
    savedir_debug = cache_root / "cfslam_captions_openai_debug"
    savedir_captions.mkdir(mode=0o700, exist_ok=True, parents=True)
    savedir_views.mkdir(mode=0o700, exist_ok=True)
    os.chmod(savedir_captions, 0o700)
    os.chmod(savedir_views, 0o700)
    if args.save_caption_debug:
        savedir_debug.mkdir(mode=0o700, exist_ok=True, parents=True)
        os.chmod(savedir_debug, 0o700)

    if args.annot_inds is None:
        requested_ids = set(range(len(scene_map)))
    else:
        requested_ids = {int(idx) for idx in args.annot_inds}
        invalid_ids = sorted(idx for idx in requested_ids if idx < 0 or idx >= len(scene_map))
        if invalid_ids:
            raise ValueError(f"annot_inds contains invalid object IDs: {invalid_ids}")

    map_path = Path(args.mapfile).resolve()
    map_stat = map_path.stat()
    caption_prompt = make_openai_caption_prompt(args.masking_option)
    settings = {
        "provider": "openai-compatible-responses",
        "model": args.openai_vision_model,
        "base_url": args.openai_base_url,
        "image_detail": args.openai_image_detail,
        "image_max_size": args.openai_image_max_size,
        "jpeg_quality": args.openai_jpeg_quality,
        "masking_option": args.masking_option,
        "max_detections_per_object": args.max_detections_per_object,
        "mapfile": str(map_path),
        "mapfile_size": map_stat.st_size,
        "mapfile_mtime_ns": map_stat.st_mtime_ns,
        "mapfile_sha256": sha256_file(map_path),
        "caption_prompt_sha256": hashlib.sha256(caption_prompt.encode("utf-8")).hexdigest(),
    }
    caption_dict_list = [
        {"id": idx, "captions": [], "low_confidences": []}
        for idx in range(len(scene_map))
    ]
    available_ids = set()
    processed_ids = []
    api_requests = 0
    cache_hits = 0

    # Mark the cache incomplete before any paid request. If the process is
    # interrupted, refinement will not consume an older, stale complete file.
    manifest_file = cache_root / "cfslam_openai_caption_manifest.json"
    save_json_atomic(
        {
            "schema_version": 1,
            "state": "running",
            "complete": False,
            "settings": settings,
            "num_scene_objects": len(scene_map),
            "requested_object_ids": sorted(requested_ids),
            "available_object_ids": [],
            "processed_object_ids": [],
            "missing_object_ids": list(range(len(scene_map))),
        },
        manifest_file,
    )

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map), desc="OpenAI vision captions"):
        object_cache_file = savedir_captions / f"{idx_obj}.json"
        if idx_obj not in requested_ids:
            try:
                with open(object_cache_file, "r", encoding="utf-8") as f:
                    cached_object = json.load(f)
                entry = cached_object.get("entry")
                if (
                    cached_object.get("settings") == settings
                    and is_valid_caption_entry(entry, idx_obj)
                    and entry["captions"]
                ):
                    caption_dict_list[idx_obj] = entry
                    available_ids.add(idx_obj)
            except (AttributeError, OSError, json.JSONDecodeError):
                pass
            continue

        selected = select_caption_detections(obj, args.max_detections_per_object)
        captions = []
        low_confidences = []
        image_list = []
        confidences_list = []
        mask_list = []
        selected_indices = []

        for candidate in tqdm(selected, leave=False, desc=f"object {idx_obj}"):
            idx_det = candidate["idx_det"]
            image_path = Path(candidate["image_path"])
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")

            mask = np.asarray(obj["mask"][idx_det], dtype=bool)
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(
                    f"Image/mask shape mismatch for object {idx_obj}, detection {idx_det}: "
                    f"image={image.size}, mask={mask.shape[::-1]}"
                )

            x1, y1, x2, y2 = candidate["xyxy"]
            image_crop, mask_crop = crop_image_and_mask(
                image,
                mask,
                x1,
                y1,
                x2,
                y2,
                padding=candidate["padding"],
            )
            if image_crop is None or mask_crop is None:
                raise ValueError(f"Failed to crop object {idx_obj}, detection {idx_det}")

            if args.masking_option == "blackout":
                image_for_api = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_for_api = draw_red_outline(image_crop, mask_crop)
            else:
                image_for_api = image_crop

            image_url = pil_image_to_data_url(
                image_for_api,
                max_size=args.openai_image_max_size,
                jpeg_quality=args.openai_jpeg_quality,
            )
            stat = image_path.stat()
            request_spec = {
                "object_id": idx_obj,
                "detection_index": idx_det,
                "image_path": str(image_path),
                "image_size": stat.st_size,
                "image_mtime_ns": stat.st_mtime_ns,
                "xyxy": candidate["xyxy"],
                "padding": candidate["padding"],
                "mask_area": candidate["mask_area"],
                "mapfile": settings["mapfile"],
                "mapfile_size": settings["mapfile_size"],
                "mapfile_mtime_ns": settings["mapfile_mtime_ns"],
                "mapfile_sha256": settings["mapfile_sha256"],
                "provider": settings["provider"],
                "base_url": args.openai_base_url,
                "model": args.openai_vision_model,
                "image_detail": args.openai_image_detail,
                "image_max_size": args.openai_image_max_size,
                "jpeg_quality": args.openai_jpeg_quality,
                "masking_option": args.masking_option,
                "image_payload_sha256": hashlib.sha256(image_url.encode("ascii")).hexdigest(),
                "caption_prompt_sha256": settings["caption_prompt_sha256"],
            }
            view_cache_file = savedir_views / f"{idx_obj:04d}_{idx_det:04d}.json"
            caption = load_cached_openai_caption(view_cache_file, request_spec)
            if caption is None:
                if client is None:
                    client = make_openai_client_from_args(args)
                caption = request_openai_image_caption(
                    client,
                    image_for_api,
                    args,
                    image_url=image_url,
                )
                save_json_atomic(
                    {"request": request_spec, "caption": caption},
                    view_cache_file,
                )
                api_requests += 1
            else:
                cache_hits += 1

            low_confidence = "unclear indoor object" in caption.lower()
            captions.append(caption)
            low_confidences.append(low_confidence)
            if args.save_caption_debug:
                image_list.append(image_for_api)
            confidences_list.append(candidate["confidence"])
            if args.save_caption_debug:
                mask_list.append(mask_crop)
            selected_indices.append(idx_det)

        entry = {
            "id": idx_obj,
            "captions": captions,
            "low_confidences": low_confidences,
        }
        caption_dict_list[idx_obj] = entry
        if captions:
            available_ids.add(idx_obj)
        processed_ids.append(idx_obj)
        save_json_atomic(
            {
                "settings": settings,
                "selected_detection_indices": selected_indices,
                "entry": entry,
            },
            object_cache_file,
        )

        if args.save_caption_debug and image_list:
            plot_images_with_captions(
                image_list,
                captions,
                confidences_list,
                low_confidences,
                mask_list,
                savedir_debug,
                idx_obj,
            )

    partial_file = cache_root / "cfslam_openai_captions_partial.json"
    save_json_atomic(caption_dict_list, partial_file)

    missing_ids = sorted(set(range(len(scene_map))) - available_ids)
    complete = not missing_ids
    legacy_file = cache_root / "cfslam_llava_captions.json"
    legacy_file_preserved = legacy_file.exists()
    if complete:
        # The OpenAI filename is canonical. Never synthesize the historical
        # LLaVA filename: doing so creates stale provenance for old consumers.
        save_json_atomic(caption_dict_list, cache_root / "cfslam_openai_captions.json")

    manifest = {
        "schema_version": 1,
        "state": "complete" if complete else "partial",
        "complete": complete,
        "settings": settings,
        "num_scene_objects": len(scene_map),
        "available_object_ids": sorted(available_ids),
        "processed_object_ids": sorted(processed_ids),
        "missing_object_ids": missing_ids,
        "api_requests_this_run": api_requests,
        "view_cache_hits_this_run": cache_hits,
        "output_file": "cfslam_openai_captions.json" if complete else partial_file.name,
        "existing_legacy_file_preserved": legacy_file_preserved,
    }
    save_json_atomic(manifest, manifest_file)

    if complete:
        print(
            f"Saved OpenAI captions for {len(scene_map)} objects "
            f"({api_requests} API requests, {cache_hits} cache hits this run)."
        )
    else:
        print(
            f"Saved partial OpenAI captions; missing object IDs: {missing_ids}. "
            "Run the remaining IDs, or omit --annot-inds, before refinement."
        )


def save_json_to_file(json_str, filename):
    save_json_atomic(json_str, filename)


def parse_json_object_text(content: str) -> dict:
    """Parse a JSON object from a plain or Markdown-fenced model response."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response is empty")
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("Model response does not contain a JSON object")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def normalize_refined_object_response(response: dict) -> dict:
    """Validate the existing GPTPrompt output without adding semantic rules."""
    summary = response.get("summary")
    possible_tags = response.get("possible_tags")
    object_tag = response.get("object_tag")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Refinement response requires a non-empty summary")
    if not isinstance(object_tag, str) or not object_tag.strip():
        raise ValueError("Refinement response requires a non-empty object_tag")
    if not isinstance(possible_tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in possible_tags
    ):
        raise ValueError("Refinement response possible_tags must be a list of strings")
    return {
        "summary": " ".join(summary.split()),
        "possible_tags": list(dict.fromkeys(tag.strip() for tag in possible_tags)),
        "object_tag": " ".join(object_tag.split()),
    }


def parse_refined_object_response(content: str, object_id: int) -> dict:
    """Parse a refinement response without retaining its body in exceptions."""
    parse_error = None
    try:
        normalized = normalize_refined_object_response(parse_json_object_text(content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parse_error = type(exc).__name__
    if parse_error is not None:
        raise RuntimeError(
            f"Invalid refinement response for object {object_id} ({parse_error}); "
            "response body omitted"
        )
    return normalized


def refine_node_captions(args):
    """Refine multi-view captions into validated, resumable semantic nodes."""
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.scenegraph.GPTPrompt import GPTPrompt

    validate_openai_runtime_args(args, require_text_model=True)

    manifest_file = Path(args.cachedir) / "cfslam_openai_caption_manifest.json"
    caption_file = Path(args.cachedir) / "cfslam_openai_captions.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            caption_manifest = json.load(f)
        if caption_manifest.get("complete") is not True:
            raise RuntimeError(
                "OpenAI caption extraction is incomplete. Finish all object IDs "
                "before refine-node-captions."
            )
        if not caption_file.exists():
            raise FileNotFoundError(
                "Caption manifest is complete but cfslam_openai_captions.json is missing."
            )
        current_map_path = Path(args.mapfile).resolve()
        current_map_stat = current_map_path.stat()
        manifest_settings = caption_manifest.get("settings", {})
        current_map_identity = {
            "mapfile": str(current_map_path),
            "mapfile_size": current_map_stat.st_size,
            "mapfile_mtime_ns": current_map_stat.st_mtime_ns,
            "mapfile_sha256": sha256_file(current_map_path),
        }
        if any(manifest_settings.get(key) != value for key, value in current_map_identity.items()):
            raise RuntimeError(
                "OpenAI captions were generated from a different map file or map revision."
            )
    elif not caption_file.exists():
        caption_file = Path(args.cachedir) / "cfslam_llava_captions.json"
    if not caption_file.exists():
        raise FileNotFoundError(
            "No complete caption file found. Run extract-node-captions for all "
            "objects before refine-node-captions."
        )

    with open(caption_file, "r", encoding="utf-8") as f:
        captions = json.load(f)
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    if not isinstance(captions, list) or len(captions) != len(scene_map):
        raise ValueError(
            f"Caption file has {len(captions) if isinstance(captions, list) else 'invalid'} "
            f"entries for {len(scene_map)} scene objects"
        )
    for expected_id, entry in enumerate(captions):
        if not is_valid_caption_entry(entry, expected_id):
            raise ValueError(f"Invalid caption entry for object {expected_id}")

    gpt_messages = GPTPrompt().get_json()
    responses_savedir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses_savedir.mkdir(mode=0o700, exist_ok=True, parents=True)
    os.chmod(responses_savedir, 0o700)
    prompt_payload = json.dumps(gpt_messages, ensure_ascii=False, sort_keys=True)
    refinement_settings = {
        "schema_version": 2,
        "provider": "openai-compatible-responses",
        "base_url": args.openai_base_url,
        "model": args.openai_model,
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
        "caption_file": str(caption_file.resolve()),
        "caption_file_sha256": sha256_file(caption_file),
        "mapfile": str(Path(args.mapfile).resolve()),
        "mapfile_sha256": sha256_file(Path(args.mapfile)),
    }
    refinement_manifest_file = Path(args.cachedir) / "cfslam_openai_refinement_manifest.json"
    save_json_atomic(
        {
            "state": "running",
            "complete": False,
            "settings": refinement_settings,
            "num_scene_objects": len(scene_map),
            "processed_object_ids": [],
        },
        refinement_manifest_file,
    )

    client = None
    responses = []
    unsuccessful_responses = 0
    api_requests = 0
    cache_hits = 0
    processed_object_ids = []

    for caption_entry in tqdm(captions, desc="OpenAI node refinement"):
        object_id = int(caption_entry["id"])
        bbox = scene_map[object_id]["bbox"]
        geometry_type = scene_map[object_id].get(
            "geometry_type",
            "colmap_3d" if bbox is not None else "multiview_2d",
        )
        prompt_input = {
            "id": object_id,
            "bbox_extent": (
                np.round(bbox.extent, 3).tolist() if bbox is not None else None
            ),
            "bbox_center": (
                np.round(bbox.center, 3).tolist() if bbox is not None else None
            ),
            "geometry_type": geometry_type,
            "captions": caption_entry["captions"],
        }
        request_spec = {"settings": refinement_settings, "input": prompt_input}
        response_file = responses_savedir / f"{object_id}.json"
        record = None
        normalized_response = None
        if response_file.exists():
            try:
                with open(response_file, "r", encoding="utf-8") as f:
                    cached_record = json.load(f)
                if cached_record.get("_request") == request_spec:
                    normalized_response = parse_refined_object_response(
                        cached_record.get("response", ""), object_id
                    )
                    record = cached_record
            except (AttributeError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
                record = None
                normalized_response = None

        if record is None:
            if client is None:
                client = make_openai_client_from_args(args)
            messages = list(gpt_messages)
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(prompt_input, ensure_ascii=False),
                }
            )
            content = request_openai_text(
                client,
                messages,
                args.openai_timeout,
                model=args.openai_model,
            )
            normalized_response = parse_refined_object_response(content, object_id)
            record = {
                **prompt_input,
                "response": json.dumps(normalized_response, ensure_ascii=False),
                "_request": request_spec,
            }
            save_json_atomic(record, response_file)
            api_requests += 1
        else:
            cache_hits += 1

        if normalized_response["object_tag"].lower() in {"invalid", "fail"}:
            unsuccessful_responses += 1
        if args.print_openai_responses:
            prjson([{"role": "user", "content": prompt_input}])
            prjson(normalized_response)
            print(f"Unsuccessful responses so far: {unsuccessful_responses}")
        responses.append(json.dumps(record, ensure_ascii=False))
        processed_object_ids.append(object_id)

    save_pickle_atomic(responses, Path(args.cachedir) / "cfslam_gpt-4_responses.pkl")
    save_json_atomic(
        {
            "state": "complete",
            "complete": True,
            "settings": refinement_settings,
            "num_scene_objects": len(scene_map),
            "processed_object_ids": processed_object_ids,
            "api_requests_this_run": api_requests,
            "cache_hits_this_run": cache_hits,
            "invalid_object_count": unsuccessful_responses,
        },
        refinement_manifest_file,
    )
    print(
        f"Saved refined nodes for {len(processed_object_ids)} objects "
        f"({api_requests} API requests, {cache_hits} cache hits, "
        f"{unsuccessful_responses} invalid)."
    )


def extract_object_tag_from_json_str(json_str):
    start_str_found = False
    is_object_tag = False
    object_tag_complete = False
    object_tag = ""
    r = json_str.strip().split()
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # Searching for open parenthesis of JSON
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # Start string found. Now skip everything until the object_tag field
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # object_tag field found. Read it
        if is_object_tag and not object_tag_complete:
            if _r == '"':
                continue
            else:
                if _r.strip() in [",", "}"]:
                    break
                object_tag += f" {_r}"
                continue
    return object_tag


def _build_multiview_scenegraph(
    args,
    scene_map,
    segment_ids_to_retain,
    cachedir: Path,
):
    """Build all evidence-qualified hybrid pairs without an MST."""

    from conceptgraph.scenegraph.multiview_relations import (
        RelationThresholds,
        canonical_sha256,
        load_camera_info_from_objects,
        mask_digest,
        pair_relation_evidence,
        scene_diagonal,
    )

    thresholds = RelationThresholds()
    if args.max_relation_candidates <= 0:
        raise ValueError("max_relation_candidates must be positive")
    camera_info, camera_path = load_camera_info_from_objects(scene_map)
    camera_sha256 = (
        sha256_file(camera_path) if camera_path is not None and camera_path.is_file() else None
    )
    diagonal = scene_diagonal(scene_map)
    map_path = Path(args.mapfile).resolve()
    map_sha256 = sha256_file(map_path)
    mask_sha256 = {
        str(original_id): mask_digest(scene_map[index])
        for index, original_id in enumerate(segment_ids_to_retain)
    }

    pair_records = []
    candidates = []
    for first_index in range(len(scene_map)):
        for second_index in range(first_index + 1, len(scene_map)):
            first_id = segment_ids_to_retain[first_index]
            second_id = segment_ids_to_retain[second_index]
            evidence = pair_relation_evidence(
                scene_map[first_index],
                scene_map[second_index],
                camera_info,
                scene_diagonal_value=diagonal,
                thresholds=thresholds,
            )
            record = {
                "pair": [first_id, second_id],
                "source_object_ids": [
                    list(scene_map[first_index].get("source_object_ids") or []),
                    list(scene_map[second_index].get("source_object_ids") or []),
                ],
                "evidence": evidence,
            }
            pair_records.append(record)
            if evidence["candidate"]:
                candidates.append(record)

    relation_prompt = """
You receive two observed indoor objects and measured multi-view/3D evidence.
Return one JSON object with exactly "object_relation" and "reason".
"object_relation" must be exactly one of:
"a on b", "b on a", "a in b", "b in a", "none of these".

Use the observed directional support/contact/containment and scale-independent
3D quality together with object semantics. Distinguish ON, INSIDE, and none.
A parent hint only recalled the pair and must never force a relation. The 3D
bbox may be null for a multiview_2d object. No image is available.
""".strip()
    evidence_manifest = {
        "schema_version": 2,
        "mode": "multiview-2d-3d",
        "state": "candidates_ready",
        "thresholds": thresholds.to_dict(),
        "thresholds_sha256": canonical_sha256(thresholds.to_dict()),
        "prompt_sha256": hashlib.sha256(relation_prompt.encode("utf-8")).hexdigest(),
        "model": {
            "base_url": args.openai_base_url,
            "model": args.openai_model,
        },
        "inputs": {
            "map": str(map_path),
            "map_sha256": map_sha256,
            "camera_info": str(camera_path) if camera_path is not None else None,
            "camera_sha256": camera_sha256,
            "mask_sha256_by_object_id": mask_sha256,
        },
        "scene_diagonal": diagonal,
        "pair_count": len(pair_records),
        "candidate_count": len(candidates),
        "max_candidates": args.max_relation_candidates,
        "pairs": pair_records,
    }
    evidence_path = cachedir / "cfslam_multiview_relation_evidence.json"
    save_json_atomic(evidence_manifest, evidence_path)
    if len(candidates) > args.max_relation_candidates:
        diagnostic_path = cachedir / "cfslam_relation_candidate_overflow.json"
        save_json_atomic(
            {
                "candidate_count": len(candidates),
                "maximum": args.max_relation_candidates,
                "candidate_pairs": [record["pair"] for record in candidates],
                "evidence_manifest": str(evidence_path),
            },
            diagnostic_path,
        )
        raise RuntimeError(
            f"Hybrid relation candidates ({len(candidates)}) exceed the "
            f"configured maximum ({args.max_relation_candidates}); diagnostics: "
            f"{diagnostic_path}"
        )

    def node_payload(index: int, original_id: int) -> dict:
        obj = scene_map[index]
        bbox = obj.get("bbox")
        response = obj["caption_dict"]["response"]
        return {
            "id": original_id,
            "object_tag": response["object_tag"],
            "caption": response["summary"],
            "possible_tags": response["possible_tags"],
            "geometry_type": obj.get(
                "geometry_type",
                "colmap_3d" if bbox is not None else "multiview_2d",
            ),
            "bbox_extent": (
                np.round(bbox.extent, 3).tolist() if bbox is not None else None
            ),
            "bbox_center": (
                np.round(bbox.center, 3).tolist() if bbox is not None else None
            ),
            "point_count": int(
                obj.get("point_count", len(np.asarray(obj["pcd"].points)))
            ),
            "is_background": bool(obj.get("is_background", False)),
            "source_object_ids": list(obj.get("source_object_ids") or []),
        }

    def compact_direction(value: dict) -> dict:
        return {key: item for key, item in value.items() if key != "clusters"}

    queries = []
    pair_indices = []
    for record in candidates:
        first_id, second_id = record["pair"]
        first_index = segment_ids_to_retain.index(first_id)
        second_index = segment_ids_to_retain.index(second_id)
        evidence = record["evidence"]
        compact_evidence = {
            "first_on_second_2d": compact_direction(evidence["first_on_second_2d"]),
            "second_on_first_2d": compact_direction(evidence["second_on_first_2d"]),
            "first_in_second_2d": compact_direction(evidence["first_in_second_2d"]),
            "second_in_first_2d": compact_direction(evidence["second_in_first_2d"]),
            "scale_independent_3d": evidence["scale_independent_3d"],
            "parent_hint": evidence["parent_hint"],
            "candidate_reasons": evidence["candidate_reasons"],
        }
        queries.append(
            {
                "object1": node_payload(first_index, first_id),
                "object2": node_payload(second_index, second_id),
                "observed_evidence": compact_evidence,
            }
        )
        pair_indices.append((first_id, second_id))

    save_json_atomic(queries, cachedir / "cfslam_object_relation_queries.json")
    allowed_relations = {
        "a on b",
        "b on a",
        "a in b",
        "b in a",
        "none of these",
    }
    cache_dir = cachedir / "cfslam_object_relation_cache"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    client = None
    relations = []
    cache_hits = 0
    api_requests = 0
    relation_manifest_path = cachedir / "cfslam_object_relations_manifest.json"

    def save_relation_progress(complete: bool) -> None:
        payload = {
            "schema_version": 2,
            "mode": "multiview-2d-3d",
            "state": "complete" if complete else "running",
            "complete": complete,
            "candidate_pairs": [list(pair) for pair in pair_indices],
            "completed_pairs": [
                list(pair) for pair in pair_indices[: len(relations)]
            ],
            "relation_count": len(relations),
            "evidence_manifest": str(evidence_path),
            "evidence_manifest_sha256": sha256_file(evidence_path),
            "cache_hits_this_run": cache_hits,
            "api_requests_this_run": api_requests,
        }
        if complete:
            payload["relations_sha256"] = canonical_sha256(relations)
        save_json_atomic(payload, relation_manifest_path)

    save_relation_progress(False)

    for pair, query in zip(pair_indices, queries):
        first_id, second_id = pair
        pair_key = f"{first_id:06d}_{second_id:06d}"
        request_identity = {
            "schema_version": 2,
            "pair": [first_id, second_id],
            "query": query,
            "mask_sha256": [mask_sha256[str(first_id)], mask_sha256[str(second_id)]],
            "camera_sha256": camera_sha256,
            "thresholds_sha256": evidence_manifest["thresholds_sha256"],
            "prompt_sha256": evidence_manifest["prompt_sha256"],
            "model": evidence_manifest["model"],
        }
        request_sha256 = canonical_sha256(request_identity)
        cache_file = cache_dir / f"{pair_key}.json"
        output_dict = None
        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                candidate = cached.get("result")
                if (
                    cached.get("request_sha256") == request_sha256
                    and cached.get("request") == request_identity
                    and isinstance(candidate, dict)
                    and cached.get("result_sha256") == canonical_sha256(candidate)
                    and candidate.get("object1") == query["object1"]
                    and candidate.get("object2") == query["object2"]
                    and candidate.get("object_relation") in allowed_relations
                    and isinstance(candidate.get("reason"), str)
                    and candidate["reason"].strip()
                ):
                    output_dict = candidate
            except (AttributeError, OSError, json.JSONDecodeError):
                output_dict = None
        if output_dict is not None:
            cache_hits += 1
            relations.append(output_dict)
            save_relation_progress(False)
            continue

        if client is None:
            client = make_openai_client_from_args(args)
        content = request_openai_text(
            client,
            [
                {
                    "role": "user",
                    "content": relation_prompt
                    + "\n\n"
                    + json.dumps(query, ensure_ascii=False),
                }
            ],
            args.openai_timeout,
            model=args.openai_model,
        )
        error_type = None
        try:
            parsed = parse_json_object_text(content)
            if set(parsed) != {"object_relation", "reason"}:
                raise ValueError("unexpected relation response schema")
            relation = str(parsed["object_relation"]).strip().lower()
            reason = str(parsed["reason"]).strip()
            if relation not in allowed_relations or not reason:
                raise ValueError("invalid relation or empty reason")
            output_dict = {
                "object1": query["object1"],
                "object2": query["object2"],
                "object_relation": relation,
                "reason": reason,
            }
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error_type = type(exc).__name__
        if error_type is not None:
            raise RuntimeError(
                f"Invalid relation response for objects {first_id} and {second_id} "
                f"({error_type}); response body omitted"
            )
        save_json_atomic(
            {
                "request_sha256": request_sha256,
                "request": request_identity,
                "result_sha256": canonical_sha256(output_dict),
                "result": output_dict,
            },
            cache_file,
        )
        relations.append(output_dict)
        api_requests += 1
        save_relation_progress(False)

    save_json_atomic(relations, cachedir / "cfslam_object_relations.json")
    save_relation_progress(True)

    edges = []
    for pair, result in zip(pair_indices, relations):
        if result["object_relation"] != "none of these":
            edges.append((pair[0], pair[1], result["object_relation"]))
    save_pickle_atomic(edges, cachedir / "cfslam_scenegraph_edges.pkl")
    print(
        f"Created hybrid scenegraph with {len(scene_map)} nodes and "
        f"{len(edges)} edges ({cache_hits} pair-cache hits, "
        f"{api_requests} API requests)."
    )


def build_scenegraph(args):
    validate_openai_runtime_args(args, require_text_model=True)

    from conceptgraph.slam.slam_classes import MapObjectList

    def compute_local_overlap_matrix(objects):
        """Compute directional point-cloud overlap without importing slam.utils."""
        import faiss

        num_objects = len(objects)
        overlap_matrix = np.zeros((num_objects, num_objects), dtype=float)
        point_arrays = []
        faiss_indices = []
        aabb_bounds = []

        for obj in objects:
            points = np.asarray(obj["pcd"].points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1:] != (3,):
                points = np.empty((0, 3), dtype=np.float32)
            points = np.ascontiguousarray(points)
            point_arrays.append(points)

            if len(points):
                index = faiss.IndexFlatL2(3)
                index.add(points)
            else:
                index = None
            faiss_indices.append(index)

            box_points = np.asarray(obj["bbox"].get_box_points(), dtype=float)
            if box_points.ndim == 2 and box_points.shape[0] and box_points.shape[1] == 3:
                aabb_bounds.append((box_points.min(axis=0), box_points.max(axis=0)))
            else:
                aabb_bounds.append(None)

        distance_threshold_squared = float(args.downsample_voxel_size) ** 2
        for i in range(num_objects):
            if not len(point_arrays[i]) or aabb_bounds[i] is None:
                continue
            min_i, max_i = aabb_bounds[i]
            for j in range(num_objects):
                if i == j or faiss_indices[j] is None or aabb_bounds[j] is None:
                    continue
                min_j, max_j = aabb_bounds[j]
                # AABB is only a cheap prefilter. The actual score below uses
                # the same nearest-neighbour ratio as slam.utils.
                intersection_extent = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                if np.any(intersection_extent <= 0):
                    continue
                distances, _ = faiss_indices[j].search(point_arrays[i], 1)
                overlap_matrix[i, j] = float(
                    np.count_nonzero(distances[:, 0] < distance_threshold_squared)
                ) / len(point_arrays[i])

        return overlap_matrix

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    cachedir = Path(args.cachedir)
    refinement_manifest_path = cachedir / "cfslam_openai_refinement_manifest.json"
    refinement_manifest = None
    if refinement_manifest_path.is_file():
        with open(refinement_manifest_path, "r", encoding="utf-8") as f:
            refinement_manifest = json.load(f)
        expected_ids = list(range(len(scene_map)))
        if refinement_manifest.get("complete") is not True:
            raise RuntimeError(
                "Node refinement is incomplete; finish refine-node-captions before building."
            )
        if refinement_manifest.get("num_scene_objects") != len(scene_map):
            raise RuntimeError("Node refinement manifest does not match the scene map size.")
        if refinement_manifest.get("processed_object_ids") != expected_ids:
            raise RuntimeError("Node refinement manifest does not cover every original object ID.")
        refinement_settings = refinement_manifest.get("settings", {})
        current_map_path = Path(args.mapfile).resolve()
        if (
            refinement_settings.get("mapfile") != str(current_map_path)
            or refinement_settings.get("mapfile_sha256") != sha256_file(current_map_path)
        ):
            raise RuntimeError("Node refinement manifest belongs to a different map revision.")

    response_dir = cachedir / "cfslam_gpt-4_responses"
    response_paths = sorted(response_dir.glob("*.json")) if response_dir.is_dir() else []
    response_file_identity = [
        {
            "name": response_path.name,
            "size": response_path.stat().st_size,
            "sha256": sha256_file(response_path),
        }
        for response_path in response_paths
    ]

    # Responses must be joined to map segments by their explicit original ID.
    # File names are deliberately not used as row positions: a missing/invalid
    # file must never shift all later captions onto the wrong objects.
    responses_by_original_id = {}
    seen_response_ids = set()
    for response_path in response_paths:
        try:
            with open(response_path, "r", encoding="utf-8") as f:
                response_dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"Ignoring unreadable refinement response: {response_path.name}")
            continue

        if not isinstance(response_dict, dict):
            print(f"Ignoring non-object refinement response: {response_path.name}")
            continue
        response_id = response_dict.get("id")
        if isinstance(response_id, bool) or not isinstance(response_id, int):
            print(f"Ignoring refinement response without an integer id: {response_path.name}")
            continue
        if response_id < 0 or response_id >= len(scene_map):
            print(f"Ignoring out-of-range refinement response id {response_id}")
            continue
        if response_id in seen_response_ids:
            raise ValueError(f"Duplicate refinement response for original object {response_id}")
        seen_response_ids.add(response_id)

        if refinement_manifest is not None:
            request_spec = response_dict.get("_request")
            if (
                not isinstance(request_spec, dict)
                or request_spec.get("settings") != refinement_manifest.get("settings")
                or not isinstance(request_spec.get("input"), dict)
                or request_spec["input"].get("id") != response_id
            ):
                raise RuntimeError(
                    f"Refinement response {response_path.name} is stale or has invalid provenance."
                )

        parsed_response = response_dict.get("response")
        if isinstance(parsed_response, str):
            try:
                parsed_response = json.loads(parsed_response)
            except json.JSONDecodeError:
                parsed_response = None
        if not isinstance(parsed_response, dict):
            continue

        normalized_response = dict(response_dict)
        normalized_response["id"] = response_id
        normalized_response["response"] = parsed_response
        responses_by_original_id[response_id] = normalized_response

    invalid_original_ids = set()
    for original_id in range(len(scene_map)):
        response_dict = responses_by_original_id.get(original_id)
        if response_dict is None:
            invalid_original_ids.add(original_id)
            continue
        object_tag = response_dict["response"].get("object_tag")
        summary = response_dict["response"].get("summary")
        possible_tags = response_dict["response"].get("possible_tags")
        if (
            not isinstance(object_tag, str)
            or not object_tag.strip()
            or object_tag.strip().lower() in {"fail", "invalid"}
            or not isinstance(summary, str)
            or not isinstance(possible_tags, list)
            or any(not isinstance(tag, str) for tag in possible_tags)
        ):
            invalid_original_ids.add(original_id)
            continue
        if len(scene_map[original_id]["conf"]) < args.min_views_per_object:
            invalid_original_ids.add(original_id)

    # Low-view objects without a response are already covered above; this loop
    # also makes the criterion explicit for every original map object.
    for original_id in range(len(scene_map)):
        if len(scene_map[original_id]["conf"]) < args.min_views_per_object:
            invalid_original_ids.add(original_id)

    indices_to_remove = sorted(invalid_original_ids)
    segment_ids_to_retain = [
        original_id
        for original_id in range(len(scene_map))
        if original_id not in invalid_original_ids
    ]
    save_pickle_atomic(
        indices_to_remove,
        cachedir / "cfslam_scenegraph_invalid_indices.pkl",
    )
    print(f"Removed {len(indices_to_remove)} segments")

    object_tags = [
        responses_by_original_id[original_id]["response"]["object_tag"].strip()
        for original_id in segment_ids_to_retain
    ]

    pruned_scene_map = []
    for original_id in segment_ids_to_retain:
        segment = scene_map[original_id]
        caption_dict = responses_by_original_id[original_id]
        if caption_dict["id"] != original_id:
            raise AssertionError("Caption ID and original map ID diverged during pruning")
        segment["caption_dict"] = caption_dict
        pruned_scene_map.append(segment)
    scene_map = MapObjectList(pruned_scene_map)
    del pruned_scene_map
    gc.collect()
    num_segments = len(scene_map)

    # Save the pruned scene map (create the directory if needed)
    map_output_dir = cachedir / "map"
    map_output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(map_output_dir / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    if args.relation_mode == "multiview-2d-3d":
        _build_multiview_scenegraph(
            args,
            scene_map,
            segment_ids_to_retain,
            cachedir,
        )
        return

    print("Computing bounding box overlaps...")
    if num_segments:
        bbox_overlaps = compute_local_overlap_matrix(scene_map)
    else:
        bbox_overlaps = np.zeros((0, 0), dtype=float)

    # Keep the official ConceptGraphs baseline behavior: use the directional
    # overlap from the lower-index object and pass that similarity directly to
    # scipy's minimum spanning tree implementation.
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            overlap = float(bbox_overlaps[i, j])
            if overlap > 0.01:
                weights.append(overlap)
                rows.append(i)
                cols.append(j)
                weights.append(overlap)
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    save_pickle_atomic(components, cachedir / "cfslam_scenegraph_components.pkl")

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relation_queries = []
    tree_edges = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        for componentidx, component in enumerate(components):
            if len(component) <= 1:
                continue
            mst_rows, mst_cols = minimum_spanning_trees[componentidx].nonzero()
            for u, v in zip(mst_rows, mst_cols):
                pruned_idx1 = component[u]
                pruned_idx2 = component[v]
                original_id1 = segment_ids_to_retain[pruned_idx1]
                original_id2 = segment_ids_to_retain[pruned_idx2]
                bbox1 = scene_map[pruned_idx1]["bbox"]
                bbox2 = scene_map[pruned_idx2]["bbox"]
                input_dict = {
                    "object1": {
                        "id": original_id1,
                        "bbox_extent": np.round(bbox1.extent, 1).tolist(),
                        "bbox_center": np.round(bbox1.center, 1).tolist(),
                        "object_tag": object_tags[pruned_idx1],
                    },
                    "object2": {
                        "id": original_id2,
                        "bbox_extent": np.round(bbox2.extent, 1).tolist(),
                        "bbox_center": np.round(bbox2.center, 1).tolist(),
                        "object_tag": object_tags[pruned_idx2],
                    },
                }
                relation_queries.append(input_dict)
                tree_edges.append((original_id1, original_id2))

    default_prompt = """
    The input is a JSON object describing two objects "object1" and "object2". Produce a JSON
    object (and nothing else), with exactly two keys: "object_relation" and "reason".

    Each of the JSON fields "object1" and "object2" has these fields:
    1. bbox_extent: the 3D bounding box extents of the object
    2. bbox_center: the 3D bounding box center of the object
    3. object_tag: an extremely brief description of the object

    The "object_relation" value must be exactly one of:
    1. "a on b": object a is commonly placed on top of object b
    2. "b on a": object b is commonly placed on top of object a
    3. "a in b": object a is commonly placed inside object b
    4. "b in a": object b is commonly placed inside object a
    5. "none of these": none of the above describes the relationship

    The "reason" value must be a non-empty string explaining the choice.
    """.strip()
    allowed_relations = {"a on b", "b on a", "a in b", "b in a", "none of these"}

    def canonical_sha256(value):
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_relation_entry(entry, expected_query):
        if not isinstance(entry, dict):
            raise ValueError("Relation result must be a JSON object")
        expected_keys = {"object1", "object2", "object_relation", "reason"}
        if set(entry) != expected_keys:
            raise ValueError("Relation result has an unexpected schema")
        if entry["object1"] != expected_query["object1"] or entry["object2"] != expected_query["object2"]:
            raise ValueError("Relation result does not match its query IDs and geometry")
        relation = entry["object_relation"]
        reason = entry["reason"]
        if not isinstance(relation, str) or relation not in allowed_relations:
            raise ValueError("Relation result contains a disallowed relation")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Relation result must contain a non-empty reason")

    map_path = Path(args.mapfile).resolve()
    map_stat = map_path.stat()
    cache_identity = {
        "schema_version": 1,
        "query": {
            "count": len(relation_queries),
            "sha256": canonical_sha256(relation_queries),
            "prompt_sha256": hashlib.sha256(default_prompt.encode("utf-8")).hexdigest(),
        },
        "model": {
            "base_url": args.openai_base_url,
            "model": args.openai_model,
        },
        "map": {
            "path": str(map_path),
            "size": map_stat.st_size,
            "mtime_ns": map_stat.st_mtime_ns,
            "sha256": sha256_file(map_path),
        },
        "responses": {
            "directory": str(response_dir.resolve()),
            "files_sha256": canonical_sha256(response_file_identity),
            "files": response_file_identity,
        },
    }

    query_path = cachedir / "cfslam_object_relation_queries.json"
    relation_path = cachedir / "cfslam_object_relations.json"
    relation_manifest_path = cachedir / "cfslam_object_relations_manifest.json"
    save_json_atomic(relation_queries, query_path)

    relations = None
    if relation_path.is_file() and relation_manifest_path.is_file():
        try:
            with open(relation_manifest_path, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            with open(relation_path, "r", encoding="utf-8") as f:
                cached_relations = json.load(f)
            if cached_manifest.get("identity") != cache_identity:
                raise ValueError("relation cache identity changed")
            if not isinstance(cached_relations, list) or len(cached_relations) != len(relation_queries):
                raise ValueError("relation cache has the wrong length")
            if cached_manifest.get("relation_count") != len(cached_relations):
                raise ValueError("relation manifest has the wrong result count")
            if cached_manifest.get("relations_sha256") != canonical_sha256(cached_relations):
                raise ValueError("relation cache checksum changed")
            for entry, expected_query in zip(cached_relations, relation_queries):
                validate_relation_entry(entry, expected_query)
            relations = cached_relations
            print(f"Reusing {len(relations)} cached object relations")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Ignoring stale or invalid object-relation cache: {exc}")

    if relations is None:
        relations = []
        client = make_openai_client_from_args(args) if relation_queries else None
        for input_dict in relation_queries:
            object_id1 = input_dict["object1"]["id"]
            object_id2 = input_dict["object2"]["id"]
            print(
                f"Inferring relation for original objects {object_id1} and {object_id2}: "
                f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}"
            )
            content = request_openai_text(
                client,
                [
                    {
                        "role": "user",
                        "content": default_prompt + "\n\n" + json.dumps(input_dict),
                    }
                ],
                args.openai_timeout,
                model=args.openai_model,
            )
            relation_error = None
            try:
                parsed = parse_json_object_text(content)
                if not isinstance(parsed, dict) or set(parsed) != {"object_relation", "reason"}:
                    raise ValueError("response must contain exactly object_relation and reason")
                relation_value = parsed["object_relation"]
                reason_value = parsed["reason"]
                if not isinstance(relation_value, str) or not isinstance(reason_value, str):
                    raise ValueError("relation and reason must both be strings")
                output_dict = {
                    "object1": input_dict["object1"],
                    "object2": input_dict["object2"],
                    "object_relation": relation_value.strip().lower(),
                    "reason": reason_value.strip(),
                }
                validate_relation_entry(output_dict, input_dict)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                relation_error = type(exc).__name__
            if relation_error is not None:
                raise RuntimeError(
                    f"Invalid relation response for original objects {object_id1} and "
                    f"{object_id2} ({relation_error}); response body omitted"
                )
            relations.append(output_dict)

        save_json_atomic(relations, relation_path)
        relation_manifest = {
            "identity": cache_identity,
            "relation_count": len(relations),
            "relations_sha256": canonical_sha256(relations),
        }
        save_json_atomic(relation_manifest, relation_manifest_path)
        print(f"Saved {len(relations)} object relations")

    if len(relations) != len(tree_edges) or len(relation_queries) != len(tree_edges):
        raise ValueError("Relation result count diverged from the minimum spanning tree")

    scenegraph_edges = []
    for (original_id1, original_id2), relation_dict, expected_query in zip(
        tree_edges, relations, relation_queries
    ):
        validate_relation_entry(relation_dict, expected_query)
        if (
            relation_dict["object1"]["id"] != original_id1
            or relation_dict["object2"]["id"] != original_id2
        ):
            raise ValueError("Relation order diverged from the minimum spanning tree")
        if relation_dict["object_relation"] != "none of these":
            scenegraph_edges.append(
                (original_id1, original_id2, relation_dict["object_relation"])
            )
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    save_pickle_atomic(scenegraph_edges, cachedir / "cfslam_scenegraph_edges.pkl")


def generate_scenegraph_json(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    

    # Generate the JSON file summarizing the scene, if it doesn't exist already
    # or if the --recopmute_scenegraph_json flag is set
    scene_desc = []
    print("Generating scene graph JSON file...")

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))
    print(f"Loaded scene map with {len(scene_map)} objects")

    for i, segment in enumerate(scene_map):
        bbox = segment.get("bbox")
        geometry_type = segment.get(
            "geometry_type",
            "colmap_3d" if bbox is not None else "multiview_2d",
        )
        _d = {
            "id": segment["caption_dict"]["id"],
            "geometry_type": geometry_type,
            "point_count": int(
                segment.get(
                    "point_count",
                    len(np.asarray(segment["pcd"].points)),
                )
            ),
            "source_object_ids": list(segment.get("source_object_ids") or []),
            "bbox_extent": (
                np.round(bbox.extent, 1).tolist() if bbox is not None else None
            ),
            "bbox_center": (
                np.round(bbox.center, 1).tolist() if bbox is not None else None
            ),
            "possible_tags": segment["caption_dict"]["response"]["possible_tags"],
            "object_tag": segment["caption_dict"]["response"]["object_tag"],
            "caption": segment["caption_dict"]["response"]["summary"],
        }
        scene_desc.append(_d)
    # Keep a stable detailed node list for the generic formatter. The legacy
    # name is also written here and is replaced by the final sparse dictionary
    # only when scenegraph_output.py format is run explicitly.
    save_json_atomic(scene_desc, Path(args.cachedir) / "scene_graph_nodes.json")
    save_json_atomic(scene_desc, Path(args.cachedir) / "scene_graph.json")
    print(f"Saved {len(scene_desc)} detailed nodes to scene_graph_nodes.json")


def display_images(image_list):
    num_images = len(image_list)
    cols = 2  # Number of columns for the subplots (you can change this as needed)
    rows = (num_images + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = image_list[i]
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def annotate_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))

    annot_inds = None
    if args.annot_inds is not None:
        annot_inds = args.annot_inds
    # If annot_inds is not None, we also need to load the annotation json file and only
    # annotate the objects that are specified in the annot_inds list
    annots = []
    if annot_inds is not None:
        annots = json.load(open(Path(args.cachedir) / "annotated_scenegraph.json", "r"))

    if annot_inds is None:
        annot_inds = list(range(len(scene_map)))

    for idx in annot_inds:
        print(f"Object {idx} out of {len(annot_inds)}...")
        obj = scene_map[idx]

        prev_annot = None
        if len(annots) >= idx + 1:
            prev_annot = annots[idx]

        annot = {}
        annot["id"] = idx

        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        print(obj.keys())

        imgs = []

        for idx_det in idx_most_conf:
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            mask_crop = crop_image_pil(Image.fromarray(mask), x1, y1, x2, y2, padding=padding)
            mask_crop = np.array(mask_crop)[..., None]
            mask_crop[mask_crop == 0] = 0.05
            image_crop = np.array(image_crop) * mask_crop
            imgs.append(image_crop)
            if len(imgs) >= 5:
                break
            # if idx_det >= 5:
            #     break

        # Display the images
        display_images(imgs)
        plt.close("all")

        # Ask the user to annotate the object
        if prev_annot is not None:
            print("Previous annotation:")
            print(prev_annot)
        annot["object_tags"] = input("Enter object tags (comma-separated): ")
        annot["colors"] = input("Enter colors (comma-separated): ")
        annot["materials"] = input("Enter materials (comma-separated): ")

        if prev_annot is not None:
            annots[idx] = annot
        else:
            annots.append(annot)

        go_on = input("Continue? (y/n): ")
        if go_on == "n":
            break

    # Save the annotations
    with open(Path(args.cachedir) / "annotated_scenegraph.json", "w") as f:
        json.dump(annots, f, indent=4)


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    
    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        build_scenegraph(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    elif args.mode == "annotate-scenegraph":
        annotate_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
