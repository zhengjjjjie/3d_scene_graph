"""Convert a validated Video2Mesh project into a ConceptGraphs scene map.

This module deliberately depends on Video2Mesh only through its on-disk
contract.  It does not import or copy Video2Mesh source code, and it never
modifies the source project.  Heavy optional dependencies used by the default
CLIP embedder are imported only when conversion actually needs them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Mapping, Protocol, Sequence


_FRAME_ID_RE = re.compile(r"^[0-9]+$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


class ImageTextEmbedder(Protocol):
    """Minimal injection interface used by :func:`convert_video2mesh_project`."""

    def embed_images(self, images: Sequence[Any]) -> Any:
        """Return one feature row per PIL image."""

    def embed_texts(self, texts: Sequence[str]) -> Any:
        """Return one feature row per string."""


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for the Video2Mesh-to-ConceptGraphs conversion."""

    clip_model_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[3]
            / "models"
            / "huggingface"
            / "clip-vit-base-patch16"
        )
    )
    clip_model_sha256: str | None = None
    clip_device: str = "auto"
    clip_local_files_only: bool = True
    clip_batch_size: int = 16
    embedding_dim: int = 512
    crop_padding_px: int = 20
    min_foreground_views: int = 2
    min_background_views: int = 1
    min_2d_only_views: int = 3
    background_classes: tuple[str, ...] = ("wall", "floor", "ceiling")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.clip_model_path, str)
            or not self.clip_model_path.strip()
        ):
            raise ValueError("clip_model_path must not be empty")
        if self.clip_model_sha256 is not None and (
            not isinstance(self.clip_model_sha256, str)
            or not _SHA256_RE.fullmatch(self.clip_model_sha256.lower())
        ):
            raise ValueError("clip_model_sha256 must be null or a 64-character SHA-256")
        if not isinstance(self.clip_device, str) or not self.clip_device.strip():
            raise ValueError("clip_device must not be empty")
        if self.clip_batch_size <= 0:
            raise ValueError("clip_batch_size must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.crop_padding_px < 0:
            raise ValueError("crop_padding_px must be non-negative")
        if self.min_foreground_views <= 0:
            raise ValueError("min_foreground_views must be positive")
        if self.min_background_views <= 0:
            raise ValueError("min_background_views must be positive")
        if self.min_2d_only_views < 3:
            raise ValueError("min_2d_only_views must be at least 3")
        normalized = tuple(_normalize_label(value) for value in self.background_classes)
        if not normalized or any(not value for value in normalized):
            raise ValueError("background_classes must contain non-empty labels")
        if len(set(normalized)) != len(normalized):
            raise ValueError("background_classes contains duplicate normalized labels")


@dataclass(frozen=True)
class _FrameRecord:
    frame_id: str
    ordinal: int
    source_frame_index: int
    source_time_sec: float | None
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class _MaskView:
    object_id: str
    frame_id: str
    path: Path
    bbox: tuple[int, int, int, int] | None
    pixel_area: int
    caption_valid: bool
    caption_rejection_reason: str | None


@dataclass
class _ProjectData:
    root: Path
    frames: list[_FrameRecord]
    frames_by_id: dict[str, _FrameRecord]
    points: Any
    colors: Any
    labels: dict[str, dict[str, Any]]
    object_summaries: dict[str, dict[str, Any]]
    point_indices: dict[str, Any]
    mask_views: dict[str, list[_MaskView]]
    summary: dict[str, Any]


class _HFClipEmbedder:
    """Lazily loaded Hugging Face CLIP implementation."""

    def __init__(self, config: AdapterConfig) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The default Video2Mesh adapter embedder requires torch and "
                "transformers. Install them in the conversion environment or "
                "inject an object with embed_images() and embed_texts()."
            ) from exc

        model_path = Path(config.clip_model_path).expanduser()
        if config.clip_local_files_only and not model_path.exists():
            raise FileNotFoundError(
                "Local CLIP model not found at "
                f"{model_path}. Set AdapterConfig.clip_model_path to the local "
                "openai/clip-vit-base-patch16 directory or inject an embedder."
            )

        source = (
            str(model_path.resolve()) if model_path.exists() else config.clip_model_path
        )
        weights_sha256: str | None = None
        if config.clip_model_sha256 is not None:
            weights_path = model_path / "pytorch_model.bin"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    "Pinned CLIP verification requires pytorch_model.bin at "
                    f"{weights_path}"
                )
            weights_sha256 = _sha256_file(weights_path)
            expected_sha256 = str(config.clip_model_sha256).lower()
            if weights_sha256 != expected_sha256:
                raise RuntimeError(
                    "Local CLIP weights failed SHA-256 verification: "
                    f"path={weights_path}, expected={expected_sha256}, "
                    f"actual={weights_sha256}"
                )
        try:
            self._processor = CLIPProcessor.from_pretrained(
                source,
                local_files_only=config.clip_local_files_only,
            )
            self._model = CLIPModel.from_pretrained(
                source,
                local_files_only=config.clip_local_files_only,
            )
        except Exception as exc:  # pragma: no cover - model/environment dependent
            raise RuntimeError(
                f"Failed to load Hugging Face CLIP model from {source!r}"
            ) from exc

        if config.clip_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.clip_device
        try:
            self._model = self._model.eval().to(device)
        except Exception as exc:  # pragma: no cover - device dependent
            raise RuntimeError(
                f"Failed to move CLIP model to device {device!r}"
            ) from exc
        self._torch = torch
        self._device = device
        self._batch_size = config.clip_batch_size
        self.metadata = {
            "implementation": "transformers.CLIPModel",
            "model": source,
            "weights_sha256": weights_sha256,
            "device": device,
            "local_files_only": config.clip_local_files_only,
        }

    def embed_images(self, images: Sequence[Any]) -> Any:
        import numpy as np

        rows = []
        with self._torch.inference_mode():
            for start in range(0, len(images), self._batch_size):
                batch = list(images[start : start + self._batch_size])
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {
                    key: value.to(self._device)
                    for key, value in inputs.items()
                    if hasattr(value, "to")
                }
                features = self._model.get_image_features(**inputs)
                rows.append(features.detach().float().cpu().numpy())
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    def embed_texts(self, texts: Sequence[str]) -> Any:
        import numpy as np

        rows = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {
                    key: value.to(self._device)
                    for key, value in inputs.items()
                    if hasattr(value, "to")
                }
                features = self._model.get_text_features(**inputs)
                rows.append(features.detach().float().cpu().numpy())
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(rows, axis=0)


def validate_video2mesh_project(project_root: str | Path) -> dict[str, Any]:
    """Validate all artifacts consumed by the adapter.

    The returned dictionary contains only JSON-compatible values.  Contract
    violations raise ``FileNotFoundError`` or ``ValueError`` with the offending
    artifact and field in the message.
    """

    return _inspect_project(Path(project_root)).summary


def convert_video2mesh_project(
    project_root: str | Path,
    output_pickle: str | Path,
    *,
    config: AdapterConfig | None = None,
    embedder: ImageTextEmbedder | None = None,
    provenance: Mapping[str, Any] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a Video2Mesh project to the current ConceptGraphs pickle contract.

    ``embedder`` is injectable for tests.  It must expose ``embed_images`` and
    ``embed_texts`` and return finite ``N x embedding_dim`` arrays.

    Both output files are created atomically and never overwrite an existing
    path.  The source Video2Mesh project is only read.
    """

    import numpy as np

    cfg = config or AdapterConfig()
    pickle_path = Path(output_pickle).expanduser().resolve()
    if not pickle_path.name.endswith(".pkl.gz"):
        raise ValueError(
            f"output_pickle must use the ConceptGraphs .pkl.gz suffix: {pickle_path}"
        )
    report_path = (
        Path(output_json).expanduser().resolve()
        if output_json is not None
        else _default_report_path(pickle_path)
    )
    if pickle_path == report_path:
        raise ValueError("output_pickle and output_json must be different paths")
    for path in (pickle_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing conversion output: {path}"
            )

    data = _inspect_project(Path(project_root))
    background_classes = {_normalize_label(value) for value in cfg.background_classes}

    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    view_rejections: dict[str, list[dict[str, Any]]] = {}
    object_is_background: dict[str, bool] = {}

    for object_id in sorted(data.object_summaries):
        label = _label_for_object(
            object_id,
            data.labels[object_id],
            data.object_summaries[object_id],
        )
        is_background = bool(
            {
                _normalize_label(label["class_name"]),
                _normalize_label(label["name"]),
                _normalize_label(label["category"]),
            }
            & background_classes
        )
        object_is_background[object_id] = is_background
        indices = data.point_indices[object_id]
        views = data.mask_views[object_id]
        nonempty_views = [view for view in views if view.bbox is not None]
        caption_views = [view for view in views if view.caption_valid]
        minimum = (
            cfg.min_background_views if is_background else cfg.min_foreground_views
        )

        rejected_views = [
            {
                "frame_id": view.frame_id,
                "mask": str(view.path),
                "reason": view.caption_rejection_reason,
            }
            for view in views
            if not view.caption_valid
        ]
        if rejected_views:
            view_rejections[object_id] = rejected_views

        reasons = []
        geometry_type = "colmap_3d" if int(indices.size) else "multiview_2d"
        if (
            geometry_type == "multiview_2d"
            and len(nonempty_views) < cfg.min_2d_only_views
        ):
            reasons.append(
                f"nonempty_2d_views={len(nonempty_views)} below "
                f"2d_only_required={cfg.min_2d_only_views}"
            )
        if len(caption_views) < minimum:
            reasons.append(
                f"caption_valid_views={len(caption_views)} below required={minimum}"
            )
        if not nonempty_views:
            reasons.append("no_nonempty_2d_masks")
        if reasons:
            rejections.append(
                {
                    "object_id": object_id,
                    "name": label["name"],
                    "category": label["category"],
                    "is_background": is_background,
                    "point_count": int(indices.size),
                    "mask_count": len(views),
                    "caption_valid_view_count": len(caption_views),
                    "reasons": reasons,
                }
            )
            continue

        candidates.append(
            {
                "object_id": object_id,
                "label": label,
                "is_background": is_background,
                "indices": indices,
                "views": nonempty_views,
                "caption_views": caption_views,
                "geometry_type": geometry_type,
            }
        )

    class_names = sorted({candidate["label"]["class_name"] for candidate in candidates})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    class_colors = {
        str(index): _stable_color(f"class:{name}").tolist()
        for index, name in enumerate(class_names)
    }
    class_colors["-1"] = [0.0, 0.0, 0.0]

    # mask_idx is the stable index of an object among all masks in that frame.
    mask_index: dict[tuple[str, str], int] = {}
    object_ids_by_frame: dict[str, list[str]] = {}
    for object_id, views in data.mask_views.items():
        for view in views:
            object_ids_by_frame.setdefault(view.frame_id, []).append(object_id)
    for frame_id, object_ids in object_ids_by_frame.items():
        for index, object_id in enumerate(sorted(set(object_ids))):
            mask_index[(frame_id, object_id)] = index

    actual_embedder: ImageTextEmbedder | None = embedder
    if candidates and actual_embedder is None:
        actual_embedder = _HFClipEmbedder(cfg)
    if candidates and (
        actual_embedder is None
        or not callable(getattr(actual_embedder, "embed_images", None))
        or not callable(getattr(actual_embedder, "embed_texts", None))
    ):
        raise TypeError(
            "embedder must provide callable embed_images(images) and "
            "embed_texts(texts) methods"
        )

    text_feature_cache: dict[str, Any] = {}
    foreground_objects: list[dict[str, Any]] = []
    background_objects: list[dict[str, Any]] = []
    source_to_map_index: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        object_id = candidate["object_id"]
        label = candidate["label"]
        class_name = label["class_name"]
        class_id = class_to_id[class_name]

        crops = [
            _load_padded_crop(
                data.frames_by_id[view.frame_id].path,
                view.bbox,
                cfg.crop_padding_px,
            )
            for view in candidate["caption_views"]
        ]
        image_rows = _normalized_feature_rows(
            actual_embedder.embed_images(crops),
            expected_rows=len(crops),
            expected_dim=cfg.embedding_dim,
            source=f"image embeddings for {object_id}",
        )
        clip_ft = _normalize_vector(
            image_rows.mean(axis=0),
            source=f"mean image embedding for {object_id}",
        ).astype(np.float32)

        if class_name not in text_feature_cache:
            text_rows = _normalized_feature_rows(
                actual_embedder.embed_texts([class_name]),
                expected_rows=1,
                expected_dim=cfg.embedding_dim,
                source=f"text embedding for {class_name!r}",
            )
            text_feature_cache[class_name] = text_rows[0].astype(np.float32)
        text_ft = text_feature_cache[class_name].copy()

        indices = candidate["indices"]
        object_points = np.asarray(data.points[indices], dtype=np.float64)
        object_colors = np.asarray(data.colors[indices], dtype=np.float64)
        bbox_points = (
            _axis_aligned_bbox_corners(object_points)
            if candidate["geometry_type"] == "colmap_3d"
            else np.empty((0, 3), dtype=np.float64)
        )

        image_idx = []
        source_frame_index = []
        frame_ids = []
        masks = []
        boxes = []
        confidences = []
        mask_indices = []
        n_points = []
        pixel_areas = []
        color_paths = []
        frame_scores = data.object_summaries[object_id].get("frame_scores", {})

        for view in candidate["views"]:
            frame = data.frames_by_id[view.frame_id]
            mask = _read_binary_mask(view.path, frame.width, frame.height)
            image_idx.append(frame.ordinal)
            source_frame_index.append(frame.source_frame_index)
            frame_ids.append(frame.frame_id)
            masks.append(mask)
            boxes.append(np.asarray(view.bbox, dtype=np.float64))
            confidences.append(1.0)
            mask_indices.append(mask_index[(frame.frame_id, object_id)])
            score = frame_scores.get(frame.frame_id, {})
            n_points.append(int(score.get("hit_points", 0)))
            pixel_areas.append(int(view.pixel_area))
            color_paths.append(str(frame.path))

        object_record = {
            "image_idx": image_idx,
            "source_frame_index": source_frame_index,
            "frame_id": frame_ids,
            "mask_idx": mask_indices,
            "color_path": color_paths,
            "class_name": [class_name] * len(masks),
            "class_id": [class_id] * len(masks),
            "num_detections": len(masks),
            "mask": masks,
            "xyxy": boxes,
            "conf": confidences,
            "n_points": n_points,
            "pixel_area": pixel_areas,
            "contain_number": [None] * len(masks),
            "inst_color": _stable_color(f"instance:{object_id}"),
            "is_background": bool(candidate["is_background"]),
            "pcd_np": object_points,
            "bbox_np": bbox_points,
            "pcd_color_np": object_colors,
            "clip_ft": clip_ft,
            "text_ft": text_ft,
            "v2m_object_id": object_id,
            "v2m_name": label["name"],
            "v2m_category": label["category"],
            "v2m_description": label["description"],
            "coordinate_units": "colmap_arbitrary",
            "point_indices": np.asarray(indices, dtype=np.int64),
            "geometry_type": candidate["geometry_type"],
            "point_count": int(indices.size),
            "source_object_ids": _lineage_list(
                data.labels[object_id].get("source_object_ids"),
                default=(object_id,),
            ),
            "source_detection_ids": _lineage_list(
                data.labels[object_id].get("source_detection_ids")
            ),
            "source_prompt_ids": _lineage_list(
                data.labels[object_id].get("source_prompt_ids")
            ),
            "parent_candidate_ids": _lineage_list(
                data.labels[object_id].get("parent_candidate_ids")
            ),
            "parent_object_ids": _lineage_list(
                data.labels[object_id].get("parent_object_ids")
            ),
        }

        if candidate["is_background"]:
            destination = background_objects
            destination_name = "bg_objects"
        else:
            destination = foreground_objects
            destination_name = "objects"
        destination_index = len(destination)
        destination.append(object_record)
        source_to_map_index[object_id] = {
            "collection": destination_name,
            "index": destination_index,
            "class_id": class_id,
        }

    embedder_metadata = _jsonable(getattr(actual_embedder, "metadata", {}))
    conversion_provenance = {
        "schema_version": 1,
        "adapter": "conceptgraph.integrations.video2mesh.adapter",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": str(data.root),
        "conversion_report": str(report_path),
        "coordinate_units": "colmap_arbitrary",
        "source_validation": data.summary,
        "adapter_config": _jsonable(asdict(cfg)),
        "embedder": embedder_metadata,
        "source_object_to_map_index": source_to_map_index,
        "caller": _jsonable(dict(provenance or {})),
    }
    payload = {
        "objects": foreground_objects,
        "bg_objects": background_objects,
        "cfg": {
            "pipeline_backend": "video2mesh",
            "coordinate_units": "colmap_arbitrary",
            "adapter_config": _jsonable(asdict(cfg)),
        },
        "class_names": class_names,
        "class_colors": class_colors,
        "provenance": conversion_provenance,
    }

    _atomic_pickle_no_clobber(payload, pickle_path)
    pickle_sha256 = _sha256_file(pickle_path)
    report = {
        "schema_version": 1,
        "status": "converted",
        "source_project": str(data.root),
        "coordinate_units": "colmap_arbitrary",
        "validation": data.summary,
        "counts": {
            "source_objects": len(data.object_summaries),
            "accepted_foreground_objects": len(foreground_objects),
            "accepted_background_objects": len(background_objects),
            "rejected_objects": len(rejections),
        },
        "accepted": source_to_map_index,
        "rejections": rejections,
        "caption_view_rejections": view_rejections,
        "provenance": conversion_provenance,
        "outputs": {
            "pickle": str(pickle_path),
            "pickle_sha256": pickle_sha256,
            "conversion_json": str(report_path),
        },
    }
    try:
        _atomic_json_no_clobber(report, report_path)
    except Exception:
        # The pickle remains a valid, atomic result.  Do not delete it: doing so
        # could race with a consumer and would violate the no-destructive-writes
        # contract.  The raised error names the missing report path.
        raise
    return report


def _inspect_project(project_root: Path) -> _ProjectData:
    import numpy as np

    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video2Mesh project directory not found: {root}")

    frames_manifest_path = root / "scene" / "frames_manifest.json"
    camera_info_path = root / "scene" / "cameras" / "camera_info.json"
    point_cloud_path = root / "scene" / "reconstruction" / "point_cloud.ply"
    mask_2d_root = root / "masks" / "2d"
    object_masks_path = root / "masks" / "3d" / "object_masks.json"
    labels_path = root / "masks" / "object_labels.json"
    required = (
        frames_manifest_path,
        camera_info_path,
        point_cloud_path,
        object_masks_path,
        labels_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required Video2Mesh artifact not found: {path}")
    if not mask_2d_root.is_dir():
        raise FileNotFoundError(
            f"Required Video2Mesh 2D mask directory not found: {mask_2d_root}"
        )

    frames_manifest = _read_json_object(frames_manifest_path)
    camera_info = _read_json_object(camera_info_path)
    object_masks = _read_json_object(object_masks_path)
    raw_labels = _read_json_object(labels_path)

    frames = _validate_frames_manifest(root, frames_manifest, frames_manifest_path)
    frames_by_id = {frame.frame_id: frame for frame in frames}
    registered_frames = _validate_camera_info(
        camera_info, frames_by_id, camera_info_path
    )
    points, colors, ply_metadata = _read_ply_points_and_colors(point_cloud_path)

    declared_point_cloud = object_masks.get("point_cloud")
    if not isinstance(declared_point_cloud, str) or not declared_point_cloud.strip():
        raise ValueError(f"{object_masks_path}: point_cloud must be a non-empty path")
    _require_same_file(
        _resolve_declared_path(declared_point_cloud, root),
        point_cloud_path,
        f"{object_masks_path}: point_cloud",
    )
    if "camera_info" in object_masks:
        declared_camera = object_masks["camera_info"]
        if not isinstance(declared_camera, str) or not declared_camera.strip():
            raise ValueError(f"{object_masks_path}: camera_info must be a path")
        _require_same_file(
            _resolve_declared_path(declared_camera, root),
            camera_info_path,
            f"{object_masks_path}: camera_info",
        )
    declared_mask_root = object_masks.get("mask_root")
    if not isinstance(declared_mask_root, str) or not declared_mask_root.strip():
        raise ValueError(f"{object_masks_path}: mask_root must be a non-empty path")
    resolved_mask_root = _resolve_declared_path(declared_mask_root, root)
    if not resolved_mask_root.is_dir():
        raise FileNotFoundError(
            f"{object_masks_path}: declared mask_root not found: {resolved_mask_root}"
        )
    if resolved_mask_root != mask_2d_root.resolve():
        raise ValueError(
            f"{object_masks_path}: declared mask_root {resolved_mask_root} is not "
            f"the canonical directory {mask_2d_root.resolve()}"
        )
    declared_num_points = object_masks.get("num_points")
    if not _is_int(declared_num_points) or int(declared_num_points) != len(points):
        raise ValueError(
            f"{object_masks_path}: num_points={declared_num_points!r} does not "
            f"match PLY vertex count {len(points)}"
        )

    raw_objects = object_masks.get("objects")
    if not isinstance(raw_objects, dict) or not raw_objects:
        raise ValueError(
            f"{object_masks_path}: objects must be a non-empty JSON object"
        )

    labels: dict[str, dict[str, Any]] = {}
    for object_id, value in raw_labels.items():
        _validate_object_id(object_id, labels_path)
        if not isinstance(value, dict):
            raise ValueError(
                f"{labels_path}: label entry {object_id!r} must be a JSON object"
            )
        labels[object_id] = value

    point_indices: dict[str, Any] = {}
    object_summaries: dict[str, dict[str, Any]] = {}
    owner = np.full(len(points), -1, dtype=np.int64)
    sorted_object_ids = sorted(raw_objects)
    object_ordinal = {
        object_id: index for index, object_id in enumerate(sorted_object_ids)
    }

    for object_id in sorted_object_ids:
        _validate_object_id(object_id, object_masks_path)
        if object_id not in labels:
            raise ValueError(
                f"{labels_path}: missing label for fused object {object_id!r}"
            )
        object_summary = raw_objects[object_id]
        if not isinstance(object_summary, dict):
            raise ValueError(
                f"{object_masks_path}: objects[{object_id!r}] must be a JSON object"
            )
        if object_summary.get("object_id", object_id) != object_id:
            raise ValueError(
                f"{object_masks_path}: object_id mismatch for key {object_id!r}"
            )
        canonical_npy = root / "masks" / "3d" / object_id / "point_indices.npy"
        if not canonical_npy.is_file():
            raise FileNotFoundError(
                f"Missing 3D point-index mask for {object_id!r}: {canonical_npy}"
            )
        mask_3d = object_summary.get("mask_3d")
        if not isinstance(mask_3d, dict):
            raise ValueError(
                f"{object_masks_path}: {object_id!r}.mask_3d must be an object"
            )
        declared_npy = mask_3d.get("point_indices_npy")
        if not isinstance(declared_npy, str) or not declared_npy.strip():
            raise ValueError(
                f"{object_masks_path}: {object_id!r}.mask_3d.point_indices_npy is required"
            )
        _require_same_file(
            _resolve_declared_path(declared_npy, root),
            canonical_npy,
            f"{object_masks_path}: {object_id!r} point_indices_npy",
        )
        try:
            indices = np.load(canonical_npy, allow_pickle=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to read integer indices from {canonical_npy}"
            ) from exc
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"{canonical_npy}: expected a one-dimensional integer array, "
                f"got shape={indices.shape}, dtype={indices.dtype}"
            )
        indices = indices.astype(np.int64, copy=False)
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= len(points)
        ):
            raise ValueError(f"{canonical_npy}: point index outside [0, {len(points)})")
        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"{canonical_npy}: duplicate point indices are not allowed"
            )
        overlap = indices[owner[indices] >= 0]
        if overlap.size:
            other_ordinal = int(owner[int(overlap[0])])
            other_id = sorted_object_ids[other_ordinal]
            raise ValueError(
                "Video2Mesh object masks are not mutually exclusive: "
                f"{object_id!r} and {other_id!r} both contain point "
                f"{int(overlap[0])}"
            )
        owner[indices] = object_ordinal[object_id]

        declared_count = object_summary.get("point_count")
        if not _is_int(declared_count) or int(declared_count) != int(indices.size):
            raise ValueError(
                f"{object_masks_path}: {object_id!r}.point_count="
                f"{declared_count!r} does not match {indices.size}"
            )
        _validate_probability_sidecar(root, object_id, mask_3d, indices)
        frame_scores = _validate_frame_scores(
            object_id,
            object_summary.get("frame_scores", {}),
            frames_by_id,
            root,
            object_masks_path,
        )
        object_summary = dict(object_summary)
        object_summary["frame_scores"] = frame_scores
        object_summaries[object_id] = object_summary
        point_indices[object_id] = indices

    mask_views = _validate_2d_masks(
        mask_2d_root,
        sorted_object_ids,
        frames_by_id,
    )
    total_masks = sum(len(views) for views in mask_views.values())
    declared_num_masks = object_masks.get("num_masks")
    if declared_num_masks is not None and (
        not _is_int(declared_num_masks) or int(declared_num_masks) != total_masks
    ):
        raise ValueError(
            f"{object_masks_path}: num_masks={declared_num_masks!r} does not "
            f"match the {total_masks} canonical 2D mask files"
        )

    # A frame score is emitted only for masks that could be projected.  It must
    # still reference a real 2D mask for the same object and frame.
    for object_id, object_summary in object_summaries.items():
        mask_frame_ids = {view.frame_id for view in mask_views[object_id]}
        missing = set(object_summary["frame_scores"]) - mask_frame_ids
        if missing:
            raise ValueError(
                f"{object_masks_path}: {object_id!r}.frame_scores references "
                f"frames without 2D masks: {sorted(missing)}"
            )

    input_hashes = {
        str(path.relative_to(root)): _sha256_file(path) for path in required
    }
    input_hashes["scene/frames"] = _tree_digest([frame.path for frame in frames], root)
    input_hashes["masks/2d"] = _tree_digest(
        [view.path for views in mask_views.values() for view in views], root
    )
    input_hashes["masks/3d/point_indices"] = _tree_digest(
        [
            root / "masks" / "3d" / object_id / "point_indices.npy"
            for object_id in sorted_object_ids
        ],
        root,
    )

    summary = {
        "schema_version": 1,
        "valid": True,
        "project_root": str(root),
        "coordinate_units": "colmap_arbitrary",
        "frames": {
            "count": len(frames),
            "first_frame_id": frames[0].frame_id,
            "last_frame_id": frames[-1].frame_id,
            "source_frame_indices": [frame.source_frame_index for frame in frames],
        },
        "camera": {
            "extrinsic_type": "world_to_camera",
            "registered_frame_count": len(registered_frames),
            "unregistered_frame_ids": sorted(set(frames_by_id) - registered_frames),
        },
        "point_cloud": {
            "path": str(point_cloud_path),
            "point_count": int(len(points)),
            **ply_metadata,
        },
        "objects": {
            "count": len(sorted_object_ids),
            "ids": sorted_object_ids,
            "point_counts": {
                object_id: int(point_indices[object_id].size)
                for object_id in sorted_object_ids
            },
            "mask_counts": {
                object_id: len(mask_views[object_id]) for object_id in sorted_object_ids
            },
            "caption_valid_view_counts": {
                object_id: sum(view.caption_valid for view in mask_views[object_id])
                for object_id in sorted_object_ids
            },
            "mutually_exclusive_3d_masks": True,
        },
        "input_sha256": input_hashes,
    }
    return _ProjectData(
        root=root,
        frames=frames,
        frames_by_id=frames_by_id,
        points=points,
        colors=colors,
        labels=labels,
        object_summaries=object_summaries,
        point_indices=point_indices,
        mask_views=mask_views,
        summary=summary,
    )


def _validate_frames_manifest(
    root: Path,
    manifest: dict[str, Any],
    path: Path,
) -> list[_FrameRecord]:
    from PIL import Image, UnidentifiedImageError

    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"{path}: frames must be a non-empty JSON list")
    declared_count = manifest.get("written_frame_count")
    if not _is_int(declared_count) or int(declared_count) != len(raw_frames):
        raise ValueError(
            f"{path}: written_frame_count={declared_count!r} does not match "
            f"frames length {len(raw_frames)}"
        )
    source_width = manifest.get("source_width")
    source_height = manifest.get("source_height")
    if (
        not _is_int(source_width)
        or not _is_int(source_height)
        or int(source_width) <= 0
        or int(source_height) <= 0
    ):
        raise ValueError(
            f"{path}: source_width/source_height must be positive integers"
        )

    frames = []
    seen_ids: set[str] = set()
    seen_sources: set[int] = set()
    seen_paths: set[Path] = set()
    for ordinal, raw in enumerate(raw_frames):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: frames[{ordinal}] must be a JSON object")
        frame_id = raw.get("frame_id")
        if not isinstance(frame_id, str) or not _FRAME_ID_RE.fullmatch(frame_id):
            raise ValueError(
                f"{path}: frames[{ordinal}].frame_id must be a digit string"
            )
        if frame_id in seen_ids:
            raise ValueError(f"{path}: duplicate frame_id {frame_id!r}")
        source_index = raw.get("source_frame_index")
        if not _is_int(source_index) or int(source_index) < 0:
            raise ValueError(
                f"{path}: frame {frame_id!r} has invalid source_frame_index"
            )
        source_index = int(source_index)
        if source_index in seen_sources:
            raise ValueError(f"{path}: duplicate source_frame_index {source_index}")
        raw_frame_path = raw.get("path")
        if not isinstance(raw_frame_path, str) or not raw_frame_path.strip():
            raise ValueError(f"{path}: frame {frame_id!r} has no path")
        frame_path = _resolve_declared_path(raw_frame_path, root)
        if not frame_path.is_file():
            raise FileNotFoundError(f"Frame declared by {path} not found: {frame_path}")
        canonical_parent = (root / "scene" / "frames").resolve()
        if frame_path.parent != canonical_parent:
            raise ValueError(
                f"{path}: frame {frame_id!r} must be directly under {canonical_parent}"
            )
        if (
            frame_path.stem != frame_id
            or frame_path.suffix.lower() not in _IMAGE_SUFFIXES
        ):
            raise ValueError(
                f"{path}: frame path {frame_path.name!r} does not match "
                f"frame_id {frame_id!r}"
            )
        if frame_path in seen_paths:
            raise ValueError(f"{path}: duplicate frame path {frame_path}")
        try:
            with Image.open(frame_path) as image:
                image.verify()
            with Image.open(frame_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable frame image: {frame_path}") from exc
        if (width, height) != (int(source_width), int(source_height)):
            raise ValueError(
                f"{path}: frame {frame_id!r} size {(width, height)} does not "
                f"match source size {(int(source_width), int(source_height))}"
            )

        raw_time = raw.get("source_time_sec")
        if raw_time is None:
            source_time = None
        elif not _is_finite_number(raw_time) or float(raw_time) < 0:
            raise ValueError(f"{path}: frame {frame_id!r} has invalid source_time_sec")
        else:
            source_time = float(raw_time)
        frames.append(
            _FrameRecord(
                frame_id=frame_id,
                ordinal=ordinal,
                source_frame_index=source_index,
                source_time_sec=source_time,
                path=frame_path,
                width=width,
                height=height,
            )
        )
        seen_ids.add(frame_id)
        seen_sources.add(source_index)
        seen_paths.add(frame_path)

    if any(
        later.source_frame_index <= earlier.source_frame_index
        for earlier, later in zip(frames, frames[1:])
    ):
        raise ValueError(f"{path}: frames must be ordered by source_frame_index")
    return frames


def _validate_camera_info(
    camera_info: dict[str, Any],
    frames_by_id: dict[str, _FrameRecord],
    path: Path,
) -> set[str]:
    import numpy as np

    if camera_info.get("extrinsic_type") != "world_to_camera":
        raise ValueError(
            f"{path}: extrinsic_type must be 'world_to_camera', got "
            f"{camera_info.get('extrinsic_type')!r}"
        )
    default_intrinsic = _validate_intrinsic(
        camera_info.get("intrinsic"), f"{path}: intrinsic"
    )
    raw_intrinsics = camera_info.get("intrinsics", {})
    if raw_intrinsics is None:
        raw_intrinsics = {}
    if not isinstance(raw_intrinsics, dict):
        raise ValueError(f"{path}: intrinsics must be a JSON object")
    intrinsics = {
        str(camera_id): _validate_intrinsic(value, f"{path}: intrinsics[{camera_id!r}]")
        for camera_id, value in raw_intrinsics.items()
    }
    raw_camera_ids = camera_info.get("frame_camera_ids", {})
    if raw_camera_ids is None:
        raw_camera_ids = {}
    if not isinstance(raw_camera_ids, dict):
        raise ValueError(f"{path}: frame_camera_ids must be a JSON object")

    aliases = _frame_aliases(frames_by_id)
    normalized_camera_ids: dict[str, str] = {}
    for raw_frame_id, camera_id in raw_camera_ids.items():
        frame_id = _resolve_frame_alias(str(raw_frame_id), aliases, path)
        camera_key = str(camera_id)
        if intrinsics and camera_key not in intrinsics:
            raise ValueError(
                f"{path}: frame {frame_id!r} references unknown camera {camera_key!r}"
            )
        if frame_id in normalized_camera_ids:
            raise ValueError(
                f"{path}: duplicate frame_camera_ids alias for {frame_id!r}"
            )
        normalized_camera_ids[frame_id] = camera_key

    raw_extrinsics = camera_info.get("extrinsic")
    if not isinstance(raw_extrinsics, dict) or not raw_extrinsics:
        raise ValueError(f"{path}: extrinsic must be a non-empty JSON object")
    registered: set[str] = set()
    for raw_frame_id, raw_matrix in raw_extrinsics.items():
        frame_id = _resolve_frame_alias(str(raw_frame_id), aliases, path)
        if frame_id in registered:
            raise ValueError(
                f"{path}: duplicate extrinsic alias for frame {frame_id!r}"
            )
        matrix = np.asarray(raw_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"{path}: extrinsic for {frame_id!r} must be a finite 4x4 matrix"
            )
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(
                f"{path}: extrinsic for {frame_id!r} has an invalid homogeneous row"
            )
        rotation = matrix[:3, :3]
        if abs(float(np.linalg.det(rotation))) < 1e-8:
            raise ValueError(
                f"{path}: extrinsic for {frame_id!r} has singular rotation"
            )
        registered.add(frame_id)

    for frame_id, frame in frames_by_id.items():
        camera_key = normalized_camera_ids.get(frame_id)
        intrinsic = intrinsics.get(camera_key, default_intrinsic)
        if (intrinsic["w"], intrinsic["h"]) != (frame.width, frame.height):
            raise ValueError(
                f"{path}: intrinsic size for frame {frame_id!r} is "
                f"{(intrinsic['w'], intrinsic['h'])}, image size is "
                f"{(frame.width, frame.height)}"
            )
    return registered


def _validate_intrinsic(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object")
    result = {}
    for key in ("w", "h"):
        raw = value.get(key)
        if not _is_int(raw) or int(raw) <= 0:
            raise ValueError(f"{source}.{key} must be a positive integer")
        result[key] = int(raw)
    for key in ("fx", "fy", "cx", "cy"):
        raw = value.get(key)
        if not _is_finite_number(raw):
            raise ValueError(f"{source}.{key} must be finite")
        result[key] = float(raw)
    if result["fx"] <= 0 or result["fy"] <= 0:
        raise ValueError(f"{source}.fx/fy must be positive")
    return result


def _validate_2d_masks(
    mask_root: Path,
    object_ids: list[str],
    frames_by_id: dict[str, _FrameRecord],
) -> dict[str, list[_MaskView]]:
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    expected = set(object_ids)
    actual = {
        child.name
        for child in mask_root.iterdir()
        if child.is_dir() and not child.name.startswith("_")
    }
    extra = actual - expected
    if extra:
        raise ValueError(
            f"{mask_root}: object directories not present in object_masks.json: "
            f"{sorted(extra)}"
        )

    result: dict[str, list[_MaskView]] = {}
    for object_id in object_ids:
        object_dir = mask_root / object_id
        # Tracking may legitimately yield no mask for a prompted object.  Keep
        # that as an object-level conversion rejection rather than making the
        # whole project unreadable.
        if not object_dir.exists():
            result[object_id] = []
            continue
        if not object_dir.is_dir():
            raise ValueError(f"{object_dir}: expected an object mask directory")
        files = sorted(
            path
            for path in object_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        unsupported = [
            path
            for path in object_dir.iterdir()
            if path.is_file() and path.suffix.lower() != ".png"
        ]
        if unsupported:
            raise ValueError(
                f"{object_dir}: only canonical PNG masks are allowed; found "
                f"{[path.name for path in unsupported]}"
            )
        seen_frames: set[str] = set()
        views = []
        for mask_path in files:
            frame_id = mask_path.stem
            if frame_id not in frames_by_id:
                raise ValueError(
                    f"{mask_path}: mask frame_id is absent from frames_manifest.json"
                )
            if frame_id in seen_frames:
                raise ValueError(
                    f"{object_dir}: duplicate 2D mask for frame {frame_id!r}"
                )
            frame = frames_by_id[frame_id]
            try:
                with Image.open(mask_path) as image:
                    array = np.asarray(image)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Unreadable 2D mask: {mask_path}") from exc
            if array.ndim != 2 or array.shape != (frame.height, frame.width):
                raise ValueError(
                    f"{mask_path}: expected a 2D mask of shape "
                    f"{(frame.height, frame.width)}, got {array.shape}"
                )
            unique = np.unique(array)
            if not set(int(value) for value in unique).issubset({0, 255}):
                raise ValueError(
                    f"{mask_path}: expected a binary 0/255 SAM2 mask, got "
                    f"values {unique[:16].tolist()}"
                )
            binary = array > 0
            bbox = _bbox_from_binary_mask(binary)
            area = int(binary.sum())
            caption_valid, reason = _caption_view_validity(
                bbox, area, frame.width, frame.height
            )
            views.append(
                _MaskView(
                    object_id=object_id,
                    frame_id=frame_id,
                    path=mask_path.resolve(),
                    bbox=bbox,
                    pixel_area=area,
                    caption_valid=caption_valid,
                    caption_rejection_reason=reason,
                )
            )
            seen_frames.add(frame_id)
        views.sort(key=lambda view: frames_by_id[view.frame_id].ordinal)
        result[object_id] = views
    return result


def _validate_frame_scores(
    object_id: str,
    raw_scores: Any,
    frames_by_id: dict[str, _FrameRecord],
    root: Path,
    source_path: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_scores, dict):
        raise ValueError(
            f"{source_path}: {object_id!r}.frame_scores must be a JSON object"
        )
    aliases = _frame_aliases(frames_by_id)
    result = {}
    for raw_frame_id, raw_score in raw_scores.items():
        frame_id = _resolve_frame_alias(str(raw_frame_id), aliases, source_path)
        if frame_id in result:
            raise ValueError(
                f"{source_path}: duplicate frame-score alias for {frame_id!r}"
            )
        if not isinstance(raw_score, dict):
            raise ValueError(
                f"{source_path}: frame score {object_id!r}/{frame_id!r} "
                "must be an object"
            )
        score = dict(raw_score)
        for key in ("mask_area", "projected_points", "visible_points", "hit_points"):
            if key in score and (not _is_int(score[key]) or int(score[key]) < 0):
                raise ValueError(
                    f"{source_path}: {object_id!r}/{frame_id!r}.{key} "
                    "must be a non-negative integer"
                )
        if "mask" in score:
            raw_mask = score["mask"]
            if not isinstance(raw_mask, str) or not raw_mask.strip():
                raise ValueError(
                    f"{source_path}: {object_id!r}/{frame_id!r}.mask must be a path"
                )
            expected = root / "masks" / "2d" / object_id / f"{frame_id}.png"
            _require_same_file(
                _resolve_declared_path(raw_mask, root),
                expected,
                f"{source_path}: {object_id!r}/{frame_id!r}.mask",
            )
        result[frame_id] = score
    return result


def _validate_probability_sidecar(
    root: Path,
    object_id: str,
    mask_3d: dict[str, Any],
    indices: Any,
) -> None:
    import numpy as np

    raw_path = mask_3d.get("point_probabilities_npz")
    if raw_path is None:
        return
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"{object_id!r}.mask_3d.point_probabilities_npz must be a path"
        )
    canonical = root / "masks" / "3d" / object_id / "point_probabilities.npz"
    _require_same_file(
        _resolve_declared_path(raw_path, root),
        canonical,
        f"{object_id!r} point_probabilities_npz",
    )
    try:
        with np.load(canonical, allow_pickle=False) as values:
            required = {
                "point_indices",
                "probability_mean",
                "probability_max",
                "probability_observations",
            }
            missing = required - set(values.files)
            if missing:
                raise ValueError(f"missing arrays {sorted(missing)}")
            sidecar_indices = values["point_indices"]
            means = np.asarray(values["probability_mean"])
            maxima = np.asarray(values["probability_max"])
            observations = np.asarray(values["probability_observations"])
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("missing arrays"):
            raise ValueError(f"{canonical}: {exc}") from exc
        raise ValueError(f"Failed to validate probability sidecar {canonical}") from exc
    if (
        sidecar_indices.ndim != 1
        or sidecar_indices.dtype.kind not in {"i", "u"}
        or not np.array_equal(sidecar_indices.astype(np.int64), indices)
    ):
        raise ValueError(f"{canonical}: point_indices do not match point_indices.npy")
    expected_shape = (indices.size,)
    for name, array in (("probability_mean", means), ("probability_max", maxima)):
        if (
            array.shape != expected_shape
            or not np.isfinite(array).all()
            or (array < 0).any()
            or (array > 1).any()
        ):
            raise ValueError(
                f"{canonical}: {name} must be finite [0,1] with shape {expected_shape}"
            )
    if (
        observations.shape != expected_shape
        or observations.dtype.kind not in {"i", "u"}
        or (observations < 0).any()
    ):
        raise ValueError(
            f"{canonical}: probability_observations must be non-negative "
            f"integers with shape {expected_shape}"
        )


def _read_ply_points_and_colors(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Read the scalar vertex table of an ASCII or binary PLY file."""

    import numpy as np

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise FileNotFoundError(f"Unable to open PLY point cloud: {path}") from exc
    with handle:
        first = handle.readline()
        if first.rstrip(b"\r\n") != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        format_name = None
        elements: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        header_bytes = len(first)
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header has no end_header")
            header_bytes += len(line)
            if header_bytes > 4 * 1024 * 1024:
                raise ValueError(f"{path}: PLY header exceeds 4 MiB")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: PLY header must be ASCII") from exc
            if not text or text.startswith("comment ") or text.startswith("obj_info "):
                continue
            fields = text.split()
            if fields[0] == "format" and len(fields) >= 3:
                format_name = fields[1]
            elif fields[0] == "element" and len(fields) == 3:
                try:
                    count = int(fields[2])
                except ValueError as exc:
                    raise ValueError(f"{path}: invalid PLY element count") from exc
                if count < 0:
                    raise ValueError(f"{path}: negative PLY element count")
                current = {"name": fields[1], "count": count, "properties": []}
                elements.append(current)
            elif fields[0] == "property":
                if current is None:
                    raise ValueError(f"{path}: PLY property before element")
                if len(fields) == 3:
                    if fields[1] not in _PLY_SCALAR_TYPES:
                        raise ValueError(
                            f"{path}: unsupported PLY scalar type {fields[1]!r}"
                        )
                    current["properties"].append(
                        {"kind": "scalar", "type": fields[1], "name": fields[2]}
                    )
                elif len(fields) == 5 and fields[1] == "list":
                    current["properties"].append(
                        {
                            "kind": "list",
                            "count_type": fields[2],
                            "item_type": fields[3],
                            "name": fields[4],
                        }
                    )
                else:
                    raise ValueError(f"{path}: malformed PLY property {text!r}")
            elif fields[0] == "end_header":
                data_offset = handle.tell()
                break

        if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"{path}: unsupported PLY format {format_name!r}")
        vertex_positions = [
            index
            for index, element in enumerate(elements)
            if element["name"] == "vertex"
        ]
        if len(vertex_positions) != 1:
            raise ValueError(f"{path}: expected exactly one vertex element")
        vertex_position = vertex_positions[0]
        if any(element["count"] for element in elements[:vertex_position]):
            raise ValueError(
                f"{path}: non-empty elements before vertex are unsupported"
            )
        vertex = elements[vertex_position]
        if vertex["count"] <= 0:
            raise ValueError(f"{path}: PLY contains no vertices")
        properties = vertex["properties"]
        if any(prop["kind"] != "scalar" for prop in properties):
            raise ValueError(f"{path}: list-valued vertex properties are unsupported")
        names = [prop["name"] for prop in properties]
        required = {"x", "y", "z", "red", "green", "blue"}
        if not required.issubset(names):
            raise ValueError(
                f"{path}: PLY vertices must include x/y/z and red/green/blue; "
                f"found {names}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate PLY vertex property names")

        endian = "<" if format_name == "binary_little_endian" else ">"
        dtype_fields = []
        for prop in properties:
            scalar = _PLY_SCALAR_TYPES[prop["type"]]
            dtype_fields.append(
                (prop["name"], scalar if scalar.endswith("1") else endian + scalar)
            )
        dtype = np.dtype(dtype_fields)

        if format_name == "ascii":
            columns: dict[str, list[Any]] = {name: [] for name in names}
            for row_index in range(vertex["count"]):
                line = handle.readline()
                if not line:
                    raise ValueError(
                        f"{path}: PLY ended at vertex {row_index}/{vertex['count']}"
                    )
                fields = line.split()
                if len(fields) != len(properties):
                    raise ValueError(
                        f"{path}: vertex {row_index} has {len(fields)} values, "
                        f"expected {len(properties)}"
                    )
                for field, prop in zip(fields, properties):
                    try:
                        value = (
                            float(field)
                            if _PLY_SCALAR_TYPES[prop["type"]].startswith("f")
                            else int(field)
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}: invalid {prop['name']} at vertex {row_index}"
                        ) from exc
                    columns[prop["name"]].append(value)
            table = np.empty(vertex["count"], dtype=dtype)
            for name in names:
                table[name] = columns[name]
        else:
            handle.seek(data_offset)
            table = np.fromfile(handle, dtype=dtype, count=vertex["count"])
            if table.size != vertex["count"]:
                raise ValueError(
                    f"{path}: binary PLY ended after {table.size}/"
                    f"{vertex['count']} vertices"
                )

    points = np.column_stack([table["x"], table["y"], table["z"]]).astype(
        np.float64, copy=False
    )
    if points.shape != (vertex["count"], 3) or not np.isfinite(points).all():
        raise ValueError(f"{path}: PLY coordinates must be finite Nx3 values")

    color_props = {
        prop["name"]: prop
        for prop in properties
        if prop["name"] in {"red", "green", "blue"}
    }
    color_types = {prop["type"] for prop in color_props.values()}
    if color_types.issubset({"uchar", "uint8"}):
        colors = (
            np.column_stack([table["red"], table["green"], table["blue"]]).astype(
                np.float64
            )
            / 255.0
        )
    elif all(_PLY_SCALAR_TYPES[value].startswith("f") for value in color_types):
        colors = np.column_stack([table["red"], table["green"], table["blue"]]).astype(
            np.float64
        )
    else:
        raise ValueError(f"{path}: RGB properties must all be uint8 or floating point")
    if (
        colors.shape != points.shape
        or not np.isfinite(colors).all()
        or (colors < 0).any()
        or (colors > 1).any()
    ):
        raise ValueError(f"{path}: PLY colors must be finite RGB values in [0,1]")
    return (
        points,
        colors,
        {
            "ply_format": format_name,
            "has_rgb": True,
        },
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON number {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _read_binary_mask(path: Path, width: int, height: int) -> Any:
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable 2D mask: {path}") from exc
    if array.ndim != 2 or array.shape != (height, width):
        raise ValueError(
            f"{path}: mask changed after validation; expected {(height, width)}, "
            f"got {array.shape}"
        )
    unique = np.unique(array)
    if not set(int(value) for value in unique).issubset({0, 255}):
        raise ValueError(
            f"{path}: mask changed after validation and is no longer binary"
        )
    return np.asarray(array > 0, dtype=bool)


def _load_padded_crop(
    image_path: Path,
    bbox: tuple[int, int, int, int] | None,
    padding: int,
) -> Any:
    from PIL import Image

    if bbox is None:
        raise ValueError(f"Cannot crop an empty mask in {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = bbox
    return image.crop(
        (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding),
        )
    )


def _normalized_feature_rows(
    values: Any,
    *,
    expected_rows: int,
    expected_dim: int,
    source: str,
) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1 and expected_rows == 1:
        array = array.reshape(1, -1)
    if array.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"{source}: expected shape {(expected_rows, expected_dim)}, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{source}: embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError(f"{source}: embeddings contain a zero vector")
    return array / norms


def _normalize_vector(value: Any, *, source: str) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{source}: expected one finite feature vector")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{source}: feature vector has zero norm")
    return array / norm


def _axis_aligned_bbox_corners(points: Any) -> Any:
    import numpy as np

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Cannot construct a bbox for an empty/non-Nx3 point set")
    minimum = np.asarray(points.min(axis=0), dtype=np.float64)
    maximum = np.asarray(points.max(axis=0), dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(points))))
    epsilon = scale * 1e-6
    degenerate = (maximum - minimum) <= epsilon
    minimum[degenerate] -= epsilon / 2.0
    maximum[degenerate] += epsilon / 2.0
    return np.asarray(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
        ],
        dtype=np.float64,
    )


def _bbox_from_binary_mask(mask: Any) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _caption_view_validity(
    bbox: tuple[int, int, int, int] | None,
    mask_area: int,
    image_width: int,
    image_height: int,
) -> tuple[bool, str | None]:
    """Mirror the current ConceptGraphs caption-view acceptance checks."""

    if bbox is None or mask_area <= 0:
        return False, "empty_mask"
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    if bbox_area <= 0:
        return False, "invalid_bbox"
    if mask_area < 100:
        return False, "mask_area_below_100"
    if mask_area / bbox_area < 0.1:
        return False, "mask_fill_ratio_below_0.1"
    padding = int(min(50, max(20, round(0.2 * max(bbox_width, bbox_height)))))
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(image_width, x2 + padding)
    crop_y2 = min(image_height, y2 + padding)
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    if min(crop_width, crop_height) < 48:
        return False, "caption_crop_dimension_below_48"
    if crop_width * crop_height < 70 * 70:
        return False, "caption_crop_area_below_4900"
    return True, None


def _label_for_object(
    object_id: str,
    label: dict[str, Any],
    object_summary: dict[str, Any],
) -> dict[str, str]:
    name = _first_nonempty_string(
        label.get("name"),
        object_summary.get("name"),
        object_id.replace("_", " "),
    )
    category = _first_nonempty_string(
        label.get("category"),
        object_summary.get("category"),
        name,
    )
    class_name = _normalize_label(category)
    if class_name in {"", "unknown", "object"}:
        class_name = _normalize_label(name) or _normalize_label(object_id)
    description = _first_nonempty_string(
        label.get("description"),
        object_summary.get("description"),
        "",
    )
    return {
        "name": name,
        "category": category,
        "class_name": class_name,
        "description": description,
    }


def _first_nonempty_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    return ""


def _lineage_list(
    value: Any,
    *,
    default: Sequence[str] = (),
) -> list[str]:
    if value is None:
        values: Sequence[Any] = default
    elif isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ValueError(f"lineage value must be a string or sequence, got {value!r}")
    result = []
    for item in values:
        normalized = str(item).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().replace("_", " ").split())


def _stable_color(value: str) -> Any:
    import numpy as np

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    # Keep colors away from black so serialized point-cloud bbox colors remain
    # visible in the existing visualizers.
    return np.asarray(
        [0.2 + (component / 255.0) * 0.75 for component in digest[:3]],
        dtype=np.float64,
    )


def _frame_aliases(frames_by_id: dict[str, _FrameRecord]) -> dict[str, str]:
    aliases = {}
    for frame_id in frames_by_id:
        aliases[frame_id] = frame_id
        numeric = str(int(frame_id))
        existing = aliases.get(numeric)
        if existing is not None and existing != frame_id:
            raise ValueError(
                f"Ambiguous numeric frame aliases: {existing!r} and {frame_id!r}"
            )
        aliases[numeric] = frame_id
    return aliases


def _resolve_frame_alias(raw: str, aliases: dict[str, str], source: Path) -> str:
    frame_id = aliases.get(raw)
    if frame_id is None and _FRAME_ID_RE.fullmatch(raw):
        frame_id = aliases.get(str(int(raw)))
    if frame_id is None:
        raise ValueError(f"{source}: references unknown frame_id {raw!r}")
    return frame_id


def _validate_object_id(value: Any, source: Path) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{source}: unsafe object_id {value!r}")


def _resolve_declared_path(raw: str, root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _require_same_file(declared: Path, expected: Path, source: str) -> None:
    if not declared.is_file():
        raise FileNotFoundError(f"{source}: declared file not found: {declared}")
    if not expected.is_file():
        raise FileNotFoundError(f"{source}: canonical file not found: {expected}")
    try:
        same = os.path.samefile(declared, expected)
    except OSError as exc:
        raise ValueError(f"{source}: failed to compare paths") from exc
    if not same:
        raise ValueError(
            f"{source}: declared path {declared} is not canonical file {expected}"
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (value.resolve() for value in paths), key=lambda item: str(item)
    ):
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Provenance contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _default_report_path(pickle_path: Path) -> Path:
    name = pickle_path.name
    if name.endswith(".pkl.gz"):
        name = name[: -len(".pkl.gz")] + ".conversion.json"
    else:
        name = name + ".conversion.json"
    return pickle_path.with_name(name)


def _atomic_pickle_no_clobber(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                pickle.dump(value, compressed, protocol=pickle.HIGHEST_PROTOCOL)
            raw.flush()
            os.fsync(raw.fileno())
        _link_no_clobber(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_no_clobber(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _link_no_clobber(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _link_no_clobber(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(
            f"Refusing to overwrite existing conversion output: {target}"
        ) from None
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
