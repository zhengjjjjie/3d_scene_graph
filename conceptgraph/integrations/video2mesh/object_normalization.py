"""Object-granularity normalization for Video2Mesh detections and SAM2 tracks.

The implementation lives in ConceptGraphs and communicates with Video2Mesh
only through files below a new project root.  Raw GroundingDINO detections and
raw SAM2 masks are immutable inputs; normalized prompts, tracks, lineage, and
fusion-only masks are written to separate paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


BED_LABELS = frozenset(
    {
        "bed",
        "mattress",
        "blanket",
        "quilt",
        "comforter",
        "bedding",
        "bed sheet",
        "bed skirt",
        "headboard",
    }
)
SEED_LABELS = ("pillow", "lamp", "nightstand")
_SAFE_ID = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True)
class TrackMergeConfig:
    min_shared_nonempty_frames: int = 5
    min_shorter_coverage: float = 0.60
    min_median_iou: float = 0.85
    min_frame_iou: float = 0.75
    min_frame_iou_fraction: float = 0.80
    anchor_separation_iou: float = 0.20
    max_anchor_frames_per_class: int = 2
    max_seeds_per_class: int = 8
    pillow_bed_min_containment: float = 0.80
    pillow_bed_max_area_ratio: float = 0.35
    pillow_dilation_px: int = 2

    def __post_init__(self) -> None:
        if self.min_shared_nonempty_frames <= 0:
            raise ValueError("min_shared_nonempty_frames must be positive")
        if self.max_anchor_frames_per_class <= 0:
            raise ValueError("max_anchor_frames_per_class must be positive")
        if self.max_seeds_per_class <= 0:
            raise ValueError("max_seeds_per_class must be positive")
        if self.pillow_dilation_px < 0:
            raise ValueError("pillow_dilation_px must be non-negative")
        for field in (
            "min_shorter_coverage",
            "min_median_iou",
            "min_frame_iou",
            "min_frame_iou_fraction",
            "anchor_separation_iou",
            "pillow_bed_min_containment",
            "pillow_bed_max_area_ratio",
        ):
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [0, 1]")


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def mask_iou(first: Any, second: Any) -> float | None:
    """Return binary IoU, excluding empty/empty frames from the denominator."""

    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} != {b.shape}")
    area_a = int(a.sum())
    area_b = int(b.sum())
    if area_a == 0 or area_b == 0:
        return None
    intersection = int(np.logical_and(a, b).sum())
    union = area_a + area_b - intersection
    return float(intersection / union) if union else None


def track_pair_metrics(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    config: TrackMergeConfig | None = None,
    forbidden: bool = False,
) -> dict[str, Any]:
    """Compute the exact pair predicate used by complete-link clustering."""

    cfg = config or TrackMergeConfig()
    masks_a = first.get("masks") or {}
    masks_b = second.get("masks") or {}
    valid_a = {str(key) for key, value in masks_a.items() if np.asarray(value, dtype=bool).any()}
    valid_b = {str(key) for key, value in masks_b.items() if np.asarray(value, dtype=bool).any()}
    shared = sorted(valid_a & valid_b)
    ious = [
        value
        for frame_id in shared
        if (value := mask_iou(masks_a[frame_id], masks_b[frame_id])) is not None
    ]
    shorter = min(len(valid_a), len(valid_b))
    coverage = len(shared) / shorter if shorter else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    strong_fraction = (
        float(np.mean(np.asarray(ious) >= cfg.min_frame_iou)) if ious else 0.0
    )
    compatible = bool(
        not forbidden
        and len(shared) >= cfg.min_shared_nonempty_frames
        and coverage >= cfg.min_shorter_coverage
        and median_iou >= cfg.min_median_iou
        and strong_fraction >= cfg.min_frame_iou_fraction
    )
    return {
        "shared_nonempty_frames": len(shared),
        "shorter_track_coverage": coverage,
        "median_mask_iou": median_iou,
        "frame_iou_at_least_threshold_fraction": strong_fraction,
        "forbidden": bool(forbidden),
        "compatible": compatible,
    }


def complete_link_track_clusters(
    tracks: Sequence[Mapping[str, Any]],
    *,
    config: TrackMergeConfig | None = None,
    forbidden_pairs: Iterable[tuple[str, str]] = (),
) -> tuple[list[list[int]], dict[str, dict[str, Any]]]:
    """Cluster same-label tracks without allowing transitive chain merges."""

    cfg = config or TrackMergeConfig()
    forbidden = {frozenset((str(a), str(b))) for a, b in forbidden_pairs}
    pair_metrics: dict[str, dict[str, Any]] = {}

    def compatible(left: int, right: int) -> bool:
        first = tracks[left]
        second = tracks[right]
        id_a = str(first["object_id"])
        id_b = str(second["object_id"])
        key = "::".join(sorted((id_a, id_b)))
        if key not in pair_metrics:
            same_label = normalize_label(first.get("label")) == normalize_label(second.get("label"))
            metrics = track_pair_metrics(
                first,
                second,
                config=cfg,
                forbidden=frozenset((id_a, id_b)) in forbidden,
            )
            metrics["same_label"] = same_label
            metrics["compatible"] = bool(same_label and metrics["compatible"])
            pair_metrics[key] = metrics
        return bool(pair_metrics[key]["compatible"])

    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            compatible(left, right)

    clusters: list[list[int]] = []
    # Deterministic greedy complete-link: a track can join a cluster only when
    # it is compatible with every member, which prevents A-B-C chain collapse.
    for index in sorted(range(len(tracks)), key=lambda item: str(tracks[item]["object_id"])):
        eligible = [
            cluster
            for cluster in clusters
            if all(compatible(index, member) for member in cluster)
        ]
        if not eligible:
            clusters.append([index])
            continue
        chosen = max(
            eligible,
            key=lambda cluster: (
                len(cluster),
                tuple(str(tracks[item]["object_id"]) for item in cluster),
            ),
        )
        chosen.append(index)
    return clusters, pair_metrics


def _area_stability(track: Mapping[str, Any]) -> float:
    areas = [
        int(np.asarray(mask, dtype=bool).sum())
        for mask in (track.get("masks") or {}).values()
        if np.asarray(mask, dtype=bool).any()
    ]
    if not areas:
        return math.inf
    mean = float(np.mean(areas))
    return float(np.std(areas) / mean) if mean else math.inf


def canonical_track_index(cluster: Sequence[int], tracks: Sequence[Mapping[str, Any]]) -> int:
    def rank(index: int) -> tuple[Any, ...]:
        track = tracks[index]
        valid_count = sum(
            np.asarray(value, dtype=bool).any()
            for value in (track.get("masks") or {}).values()
        )
        confidence = float(track.get("detection_confidence", 0.0) or 0.0)
        return (
            -int(valid_count),
            _area_stability(track),
            -confidence,
            str(track["object_id"]),
        )

    return min(cluster, key=rank)


def merge_track_cluster(
    cluster: Sequence[int],
    tracks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_index = canonical_track_index(cluster, tracks)
    canonical = dict(tracks[canonical_index])
    merged_masks: dict[str, np.ndarray] = {}
    for index in cluster:
        for frame_id, mask in (tracks[index].get("masks") or {}).items():
            binary = np.asarray(mask, dtype=bool)
            if frame_id in merged_masks:
                if merged_masks[frame_id].shape != binary.shape:
                    raise ValueError(f"mask shape mismatch in merged frame {frame_id}")
                merged_masks[frame_id] = np.logical_or(merged_masks[frame_id], binary)
            else:
                merged_masks[str(frame_id)] = binary.copy()
    source_ids = sorted(str(tracks[index]["object_id"]) for index in cluster)
    canonical["masks"] = merged_masks
    canonical["source_object_ids"] = source_ids
    canonical["canonical_object_id"] = str(tracks[canonical_index]["object_id"])
    canonical["object_id"] = canonical["canonical_object_id"]
    canonical["valid_frame_count"] = int(
        sum(bool(mask.any()) for mask in merged_masks.values())
    )
    canonical["area_cv"] = _area_stability(canonical)
    return canonical


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if pixels <= 0:
        return binary.copy()
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(binary, iterations=pixels)
    except ImportError:  # pragma: no cover - scipy is a core ConceptGraphs dep
        padded = np.pad(binary, pixels)
        result = np.zeros_like(binary)
        for dy in range(2 * pixels + 1):
            for dx in range(2 * pixels + 1):
                result |= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
        return result


def carve_child_masks_from_parent(
    parent_masks: Mapping[str, Any],
    child_masks: Mapping[str, Any],
    *,
    min_median_containment: float = 0.80,
    max_area_ratio: float = 0.35,
    dilation_px: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carve fusion-only parent masks while preserving every child mask."""

    shared: list[str] = []
    containments: list[float] = []
    area_ratios: list[float] = []
    for frame_id in sorted(set(parent_masks) & set(child_masks)):
        parent = np.asarray(parent_masks[frame_id], dtype=bool)
        child = np.asarray(child_masks[frame_id], dtype=bool)
        if parent.shape != child.shape:
            raise ValueError(f"parent/child mask shape mismatch for frame {frame_id}")
        parent_area = int(parent.sum())
        child_area = int(child.sum())
        if not parent_area or not child_area:
            continue
        shared.append(frame_id)
        containments.append(float(np.logical_and(parent, child).sum() / child_area))
        area_ratios.append(float(child_area / parent_area))
    median_containment = float(np.median(containments)) if containments else 0.0
    max_ratio = max(area_ratios, default=math.inf)
    accepted = bool(
        shared
        and median_containment >= min_median_containment
        and max_ratio <= max_area_ratio
    )
    carved = {str(key): np.asarray(value, dtype=bool).copy() for key, value in parent_masks.items()}
    removed_pixels: dict[str, int] = {}
    if accepted:
        for frame_id in shared:
            before = carved[frame_id]
            after = np.logical_and(
                before,
                np.logical_not(_dilate(np.asarray(child_masks[frame_id], dtype=bool), dilation_px)),
            )
            removed_pixels[frame_id] = int(before.sum() - after.sum())
            carved[frame_id] = after
    return carved, {
        "accepted": accepted,
        "shared_nonempty_frames": len(shared),
        "median_containment": median_containment,
        "max_area_ratio": None if not math.isfinite(max_ratio) else max_ratio,
        "dilation_px": dilation_px,
        "removed_pixels_by_frame": removed_pixels,
        "child_masks_modified": False,
    }


def _slug(value: str) -> str:
    normalized = _SAFE_ID.sub("-", normalize_label(value)).strip("-")
    return normalized or "object"


def _raw_index(detection: Mapping[str, Any], fallback: int) -> int:
    value = detection.get("raw_index", fallback)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else fallback


def _source_detection_id(detection: Mapping[str, Any], fallback: int) -> str:
    return f"{str(detection.get('frame_id', 'unknown'))}:{_raw_index(detection, fallback):06d}"


def _prompt_from_detection(
    detection: Mapping[str, Any],
    index: int,
    *,
    label: str,
    parent_candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    frame_id = str(detection["frame_id"])
    raw_index = _raw_index(detection, index)
    object_id = f"cg_{_slug(label)}_f{frame_id}_d{raw_index:06d}"
    score = float(detection.get("score", 0.0) or 0.0)
    source_id = _source_detection_id(detection, index)
    return {
        "id": object_id,
        "object_id": object_id,
        "name": label,
        "category": label,
        "description": f"ConceptGraphs provisional {label} seed from raw GroundingDINO detection.",
        "frame_id": frame_id,
        "bbox": [int(round(float(value))) for value in detection["bbox"]],
        "bbox_format": "xyxy",
        "score": score,
        "detection_count": 1,
        "source": "conceptgraphs_object_normalization",
        "source_detection_ids": [source_id],
        "source_prompt_ids": [],
        "parent_candidate_ids": list(parent_candidate_ids),
        "open_vocab": {
            "provider": "groundingdino",
            "label": label,
            "source_labels": [normalize_label(detection.get("label"))],
            "anchor_frames": [frame_id],
        },
    }


def _select_seed_detections(
    detections: Sequence[Mapping[str, Any]],
    label: str,
    config: TrackMergeConfig,
) -> list[tuple[int, Mapping[str, Any]]]:
    accepted_labels = (
        {"nightstand", "bedside table"} if label == "nightstand" else {label}
    )
    matching = [
        (index, item)
        for index, item in enumerate(detections)
        if normalize_label(item.get("label") or item.get("category"))
        in accepted_labels
    ]
    by_frame: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for item in matching:
        by_frame.setdefault(str(item[1].get("frame_id")), []).append(item)
    ranked_frames = sorted(
        by_frame,
        key=lambda frame: (
            -max(float(item.get("score", 0.0) or 0.0) for _, item in by_frame[frame]),
            frame,
        ),
    )[: config.max_anchor_frames_per_class]
    selected = [item for frame in ranked_frames for item in by_frame[frame]]
    selected.sort(
        key=lambda pair: (
            -float(pair[1].get("score", 0.0) or 0.0),
            str(pair[1].get("frame_id")),
            pair[0],
        )
    )
    return selected[: config.max_seeds_per_class]


def normalize_detection_manifest(
    manifest: Mapping[str, Any],
    *,
    config: TrackMergeConfig | None = None,
) -> dict[str, Any]:
    """Build tracking prompts from retained raw GroundingDINO detections."""

    cfg = config or TrackMergeConfig()
    raw = manifest.get("raw_detections")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "GroundingDINO manifest has no raw_detections; run discovery with "
            "--keep-raw-detections"
        )
    raw_detections = [item for item in raw if isinstance(item, Mapping)]
    existing = [item for item in manifest.get("objects", []) if isinstance(item, Mapping)]

    bed_candidates = [
        (index, item)
        for index, item in enumerate(raw_detections)
        if normalize_label(item.get("label") or item.get("category")) in BED_LABELS
    ]
    bed_prompt: dict[str, Any] | None = None
    if bed_candidates:
        index, best = max(
            bed_candidates,
            key=lambda pair: (float(pair[1].get("score", 0.0) or 0.0), -pair[0]),
        )
        bed_prompt = _prompt_from_detection(best, index, label="bed")
        bed_prompt["object_id"] = bed_prompt["id"] = "cg_bed"
        bed_prompt["source_detection_ids"] = [
            _source_detection_id(item, raw_index)
            for raw_index, item in bed_candidates
        ]
        bed_prompt["description"] = (
            "ConceptGraphs whole-bed prompt absorbing mattress, blanket, quilt, "
            "comforter, bedding, bed sheet, bed skirt, and headboard detections."
        )

    prompts: list[dict[str, Any]] = []
    if bed_prompt is not None:
        prompts.append(bed_prompt)
    parent_candidates = [bed_prompt["object_id"]] if bed_prompt is not None else []
    for label in SEED_LABELS:
        for raw_index, detection in _select_seed_detections(raw_detections, label, cfg):
            prompts.append(
                _prompt_from_detection(
                    detection,
                    raw_index,
                    label=label,
                    parent_candidate_ids=parent_candidates if label == "pillow" else (),
                )
            )

    special_or_bed = set(SEED_LABELS) | BED_LABELS | {"bedside table"}
    used_ids = {prompt["object_id"] for prompt in prompts}
    for item in existing:
        label = normalize_label(item.get("name") or item.get("category"))
        if label in special_or_bed:
            continue
        prompt = dict(item)
        object_id = str(prompt.get("object_id") or prompt.get("id") or f"cg_{_slug(label)}")
        if object_id in used_ids:
            suffix = 2
            while f"{object_id}_{suffix}" in used_ids:
                suffix += 1
            object_id = f"{object_id}_{suffix}"
        prompt["id"] = prompt["object_id"] = object_id
        prompt["source_prompt_ids"] = [str(item.get("object_id") or item.get("id"))]
        prompt.setdefault("source_detection_ids", [])
        prompt.setdefault("parent_candidate_ids", [])
        prompts.append(prompt)
        used_ids.add(object_id)

    forbidden_pairs = []
    by_label_frame: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for prompt in prompts:
        key = (normalize_label(prompt.get("name")), str(prompt.get("frame_id")))
        by_label_frame.setdefault(key, []).append(prompt)
    for (_label, _frame), items in by_label_frame.items():
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if bbox_iou(left["bbox"], right["bbox"]) < cfg.anchor_separation_iou:
                    forbidden_pairs.append(
                        sorted((str(left["object_id"]), str(right["object_id"])))
                    )

    output = dict(manifest)
    output.update(
        {
            "schema_version": 2,
            "method": "conceptgraphs_object_granularity_normalization",
            "source_method": manifest.get("method"),
            "source_manifest_sha256": canonical_sha256(manifest),
            "object_count": len(prompts),
            "objects": prompts,
            "normalization": {
                "config": asdict(cfg),
                "bed_absorbed_labels": sorted(BED_LABELS - {"bed"}),
                "independent_seed_labels": list(SEED_LABELS),
                "forbidden_merge_pairs": forbidden_pairs,
            },
        }
    )
    return output


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return array > 0


def _write_mask(path: Path, mask: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.png")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(temporary, format="PNG")
    os.replace(temporary, path)


def normalize_prompts_project(
    project_root: str | Path,
    *,
    config: TrackMergeConfig | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    source = root / "masks" / "object_prompts_groundingdino.json"
    destination = root / "masks" / "object_prompts_normalized.json"
    if destination.exists():
        raise FileExistsError(f"normalized prompt manifest already exists: {destination}")
    normalized = normalize_detection_manifest(_read_json(source), config=config)
    _write_json_atomic(destination, normalized)
    labels_path = root / "masks" / "object_labels.json"
    source_labels = _read_json(labels_path) if labels_path.is_file() else {}
    labels: dict[str, Any] = {}
    for prompt in normalized["objects"]:
        object_id = str(prompt["object_id"])
        source_prompt_ids = prompt.get("source_prompt_ids") or []
        source_label = next(
            (source_labels[item] for item in source_prompt_ids if item in source_labels),
            {},
        )
        labels[object_id] = {
            **dict(source_label),
            "object_id": object_id,
            "name": prompt["name"],
            "category": prompt["category"],
            "description": prompt["description"],
            "confidence": prompt.get("score"),
            "source": "conceptgraphs_object_normalization",
            "source_detection_ids": prompt.get("source_detection_ids", []),
            "source_prompt_ids": source_prompt_ids,
            "parent_candidate_ids": prompt.get("parent_candidate_ids", []),
        }
    _write_json_atomic(labels_path, labels)
    return normalized


def _tracks_from_raw(
    raw_root: Path,
    prompts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tracks = []
    for object_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        object_id = object_dir.name
        prompt = prompts.get(object_id, {})
        masks = {
            path.stem: _load_mask(path)
            for path in sorted(object_dir.glob("*.png"))
        }
        tracks.append(
            {
                "object_id": object_id,
                "label": normalize_label(
                    prompt.get("name") or prompt.get("category") or object_id
                ),
                "detection_confidence": float(prompt.get("score", 0.0) or 0.0),
                "prompt": dict(prompt),
                "masks": masks,
            }
        )
    return tracks


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _output_path_is_occupied(path: Path) -> bool:
    """Treat Video2Mesh's empty init directories as available outputs."""

    if path.is_symlink():
        return True
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def normalize_tracks_project(
    project_root: str | Path,
    *,
    config: TrackMergeConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TrackMergeConfig()
    root = Path(project_root).expanduser().resolve()
    raw_root = root / "masks" / "2d_raw"
    output_root = root / "masks" / "2d"
    fusion_root = root / "masks" / "2d_fusion"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw SAM2 mask root not found: {raw_root}")
    occupied_outputs = [
        path
        for path in (output_root, fusion_root)
        if _output_path_is_occupied(path)
    ]
    if occupied_outputs:
        raise FileExistsError(
            "normalized or fusion mask output already exists: "
            + ", ".join(str(path) for path in occupied_outputs)
        )

    prompt_manifest = _read_json(root / "masks" / "object_prompts_normalized.json")
    prompts = {
        str(item["object_id"]): item
        for item in prompt_manifest.get("objects", [])
        if isinstance(item, Mapping) and "object_id" in item
    }
    tracks = _tracks_from_raw(raw_root, prompts)
    if not tracks:
        raise ValueError(f"raw SAM2 mask root contains no object tracks: {raw_root}")
    forbidden_pairs = prompt_manifest.get("normalization", {}).get(
        "forbidden_merge_pairs", []
    )
    clusters, pair_metrics = complete_link_track_clusters(
        tracks,
        config=cfg,
        forbidden_pairs=(tuple(item) for item in forbidden_pairs),
    )
    merged = [merge_track_cluster(cluster, tracks) for cluster in clusters]

    labels: dict[str, Any] = {}
    normalized_objects: dict[str, Any] = {}
    for track in merged:
        object_id = str(track["object_id"])
        prompt_sources = [tracks[index]["prompt"] for index in range(len(tracks)) if tracks[index]["object_id"] in track["source_object_ids"]]
        prompt = next(
            (item for item in prompt_sources if str(item.get("object_id")) == object_id),
            prompt_sources[0] if prompt_sources else {},
        )
        frame_records = []
        for frame_id, mask in sorted(track["masks"].items()):
            if not mask.any():
                continue
            mask_path = output_root / object_id / f"{frame_id}.png"
            _write_mask(mask_path, mask)
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "mask": str(mask_path),
                    "bbox": _bbox_from_mask(mask),
                    "mask_area": int(mask.sum()),
                    "track_score": 1.0,
                    "tracking_mode": "sam2_video_normalized",
                }
            )
        normalized_objects[object_id] = {
            "object_id": object_id,
            "canonical_object_id": object_id,
            "source_object_ids": track["source_object_ids"],
            "source_detection_ids": sorted(
                {
                    source_id
                    for item in prompt_sources
                    for source_id in item.get("source_detection_ids", [])
                }
            ),
            "source_prompt_ids": sorted(
                {
                    source_id
                    for item in prompt_sources
                    for source_id in item.get("source_prompt_ids", [])
                }
            ),
            "parent_candidate_ids": sorted(
                {
                    parent_id
                    for item in prompt_sources
                    for parent_id in item.get("parent_candidate_ids", [])
                }
            ),
            "name": prompt.get("name", track["label"]),
            "category": prompt.get("category", track["label"]),
            "detection_confidence": track["detection_confidence"],
            "valid_frame_count": track["valid_frame_count"],
            "area_cv": track["area_cv"],
            "frames_written": len(frame_records),
            "frames": frame_records,
        }
        labels[object_id] = {
            "object_id": object_id,
            "name": normalized_objects[object_id]["name"],
            "category": normalized_objects[object_id]["category"],
            "description": prompt.get("description", ""),
            "confidence": track["detection_confidence"],
            "source": "conceptgraphs_track_normalization",
            "source_object_ids": track["source_object_ids"],
            "source_detection_ids": normalized_objects[object_id]["source_detection_ids"],
            "source_prompt_ids": normalized_objects[object_id]["source_prompt_ids"],
            "parent_candidate_ids": normalized_objects[object_id]["parent_candidate_ids"],
        }

    # Fusion starts from exact copies of normalized masks. Pillow pixels are
    # removed only from qualifying bed masks in this separate tree.
    for object_id, record in normalized_objects.items():
        for frame in record["frames"]:
            source = Path(frame["mask"])
            target = fusion_root / object_id / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    bed_tracks = [track for track in merged if normalize_label(track.get("label")) == "bed"]
    pillow_tracks = [
        track for track in merged if normalize_label(track.get("label")) == "pillow"
    ]
    carve_reports = []
    for bed in bed_tracks:
        fusion_masks = {
            frame_id: _load_mask(fusion_root / bed["object_id"] / f"{frame_id}.png")
            for frame_id in bed["masks"]
            if (fusion_root / bed["object_id"] / f"{frame_id}.png").is_file()
        }
        for pillow in pillow_tracks:
            carved, report = carve_child_masks_from_parent(
                fusion_masks,
                pillow["masks"],
                min_median_containment=cfg.pillow_bed_min_containment,
                max_area_ratio=cfg.pillow_bed_max_area_ratio,
                dilation_px=cfg.pillow_dilation_px,
            )
            report.update(
                {
                    "parent_object_id": bed["object_id"],
                    "child_object_id": pillow["object_id"],
                }
            )
            carve_reports.append(report)
            if report["accepted"]:
                fusion_masks = carved
                normalized_objects[pillow["object_id"]]["parent_object_ids"] = [
                    bed["object_id"]
                ]
                labels[pillow["object_id"]]["parent_object_ids"] = [bed["object_id"]]
        for frame_id, mask in fusion_masks.items():
            _write_mask(fusion_root / bed["object_id"] / f"{frame_id}.png", mask)

    raw_manifest_path = raw_root / "tracking_manifest.json"
    raw_manifest = _read_json(raw_manifest_path) if raw_manifest_path.is_file() else {}
    manifest = {
        "schema_version": 2,
        "method": "conceptgraphs_complete_link_track_normalization",
        "source_mask_root": str(raw_root),
        "source_tracking_manifest_sha256": (
            hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()
            if raw_manifest_path.is_file()
            else None
        ),
        "mask_root": str(output_root),
        "fusion_mask_root": str(fusion_root),
        "config": asdict(cfg),
        "objects": normalized_objects,
        "clusters": [
            {
                "canonical_object_id": merged[index]["object_id"],
                "source_object_ids": merged[index]["source_object_ids"],
            }
            for index in range(len(merged))
        ],
        "pair_metrics": pair_metrics,
        "pillow_bed_carve": carve_reports,
        "raw_tracking": {
            "method": raw_manifest.get("method"),
            "mask_backend": raw_manifest.get("mask_backend"),
        },
    }
    _write_json_atomic(output_root / "tracking_manifest.json", manifest)
    fusion_manifest = {
        **manifest,
        "mask_root": str(fusion_root),
        "observation_mask_root": str(output_root),
    }
    _write_json_atomic(fusion_root / "tracking_manifest.json", fusion_manifest)
    _write_json_atomic(root / "masks" / "object_labels.json", labels)
    project_manifest_path = root / "manifest.json"
    if project_manifest_path.is_file():
        project_manifest = _read_json(project_manifest_path)
        project_manifest.setdefault("artifacts", {})["object_masks_2d_raw"] = str(
            raw_root
        )
        project_manifest["artifacts"]["object_masks_2d"] = str(output_root)
        project_manifest["artifacts"]["mask_tracking_manifest"] = str(
            output_root / "tracking_manifest.json"
        )
        project_manifest.setdefault("external_stages", {})[
            "conceptgraphs_object_normalization"
        ] = {
            "status": "normalized",
            "raw_mask_root": str(raw_root),
            "mask_root": str(output_root),
            "fusion_mask_root": str(fusion_root),
        }
        _write_json_atomic(project_manifest_path, project_manifest)
    return manifest


def finalize_fusion_manifest(project_root: str | Path) -> dict[str, Any]:
    """Point downstream observations at original normalized masks after fusion."""

    root = Path(project_root).expanduser().resolve()
    path = root / "masks" / "3d" / "object_masks.json"
    value = _read_json(path)
    fusion_root = (root / "masks" / "2d_fusion").resolve()
    observation_root = (root / "masks" / "2d").resolve()
    declared = Path(str(value.get("mask_root", ""))).expanduser().resolve()
    if declared != fusion_root:
        raise ValueError(
            f"fusion manifest mask_root is {declared}, expected {fusion_root}"
        )
    value["fusion_input_mask_root"] = str(fusion_root)
    value["mask_root"] = str(observation_root)
    for object_id, summary in (value.get("objects") or {}).items():
        for frame_id, score in (summary.get("frame_scores") or {}).items():
            if isinstance(score, dict) and "mask" in score:
                expected = observation_root / str(object_id) / f"{frame_id}.png"
                if not expected.is_file():
                    raise FileNotFoundError(
                        f"normalized observation mask missing after fusion: {expected}"
                    )
                score["fusion_input_mask"] = score["mask"]
                if "mask_area" in score:
                    score["fusion_input_mask_area"] = score["mask_area"]
                score["mask"] = str(expected)
                score["mask_area"] = int(_load_mask(expected).sum())
    _write_json_atomic(path, value)
    return value


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("normalize-prompts", "normalize-tracks", "finalize-fusion"),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()
    try:
        config_value = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--config-json must contain a JSON object") from exc
    if not isinstance(config_value, dict):
        raise ValueError("--config-json must contain a JSON object")
    config = TrackMergeConfig(**config_value)
    if args.stage == "normalize-prompts":
        result = normalize_prompts_project(args.project_root, config=config)
    elif args.stage == "normalize-tracks":
        result = normalize_tracks_project(args.project_root, config=config)
    else:
        result = finalize_fusion_manifest(args.project_root)
    print(json.dumps({"ok": True, "stage": args.stage, "sha256": canonical_sha256(result)}))


if __name__ == "__main__":
    _main()
