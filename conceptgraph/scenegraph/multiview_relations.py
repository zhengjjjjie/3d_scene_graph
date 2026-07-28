"""Multi-view 2D and scale-independent 3D relation candidate evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RelationThresholds:
    pose_angle_degrees: float = 5.0
    pose_translation_scene_fraction: float = 0.03
    min_pose_clusters: int = 3
    min_above_vote: float = 0.75
    min_support_vote: float = 0.50
    min_median_x_support: float = 0.25
    min_x_support: float = 0.25
    min_gap: float = -0.25
    max_gap: float = 0.10
    containment_bbox: float = 0.90
    containment_hull: float = 0.80
    containment_max_area_ratio: float = 0.50
    containment_cluster_support: float = 0.60
    background_min_pose_clusters: int = 5
    background_min_support_vote: float = 0.60
    neighbor_radius_spacing_multiplier: float = 2.5
    min_directional_neighbor_fraction: float = 0.02
    max_points_per_object: int = 5000

    def __post_init__(self) -> None:
        if self.min_pose_clusters < 1 or self.background_min_pose_clusters < 1:
            raise ValueError("pose cluster thresholds must be positive")
        if self.max_points_per_object < 2:
            raise ValueError("max_points_per_object must be at least 2")
        if self.pose_angle_degrees <= 0:
            raise ValueError("pose_angle_degrees must be positive")
        if self.pose_translation_scene_fraction < 0:
            raise ValueError("pose translation fraction must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bbox(mask: Any) -> tuple[int, int, int, int] | None:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"relation mask must be two-dimensional, got {binary.shape}")
    ys, xs = np.where(binary)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_intersection_over_first(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    return float(width * height / area) if area else 0.0


def frame_on_evidence(
    source_mask: Any,
    target_mask: Any,
    *,
    thresholds: RelationThresholds | None = None,
) -> dict[str, Any] | None:
    cfg = thresholds or RelationThresholds()
    source = _bbox(source_mask)
    target = _bbox(target_mask)
    if source is None or target is None:
        return None
    width = source[2] - source[0]
    height = source[3] - source[1]
    if width <= 0 or height <= 0:
        return None
    horizontal_intersection = max(
        0, min(source[2], target[2]) - max(source[0], target[0])
    )
    x_support = float(horizontal_intersection / width)
    gap = float((target[1] - source[3]) / height)
    source_centroid_y = (source[1] + source[3]) * 0.5
    target_centroid_y = (target[1] + target[3]) * 0.5
    above = source_centroid_y < target_centroid_y
    support = bool(
        x_support >= cfg.min_x_support and cfg.min_gap <= gap <= cfg.max_gap
    )
    return {
        "above": bool(above),
        "support": support,
        "vote": bool(above and support),
        "x_support": x_support,
        "gap": gap,
    }


def _filled_hull(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary)
    try:
        import cv2

        points = np.column_stack(np.where(binary)[::-1]).astype(np.int32)
        hull = cv2.convexHull(points)
        filled = np.zeros(binary.shape, dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull, 1)
        return filled.astype(bool)
    except ImportError:  # pragma: no cover - opencv is a core project dependency
        return binary.copy()


def frame_containment_evidence(
    small_mask: Any,
    large_mask: Any,
    *,
    thresholds: RelationThresholds | None = None,
) -> dict[str, Any] | None:
    cfg = thresholds or RelationThresholds()
    small = np.asarray(small_mask, dtype=bool)
    large = np.asarray(large_mask, dtype=bool)
    if small.shape != large.shape:
        raise ValueError("containment masks must have the same shape")
    small_bbox = _bbox(small)
    large_bbox = _bbox(large)
    small_area = int(small.sum())
    large_area = int(large.sum())
    if small_bbox is None or large_bbox is None or not small_area or not large_area:
        return None
    bbox_containment = _bbox_intersection_over_first(small_bbox, large_bbox)
    hull = _filled_hull(large)
    hull_containment = float(np.logical_and(small, hull).sum() / small_area)
    ys, xs = np.where(small)
    center_x = int(round(float(xs.mean())))
    center_y = int(round(float(ys.mean())))
    center_x = min(max(center_x, 0), hull.shape[1] - 1)
    center_y = min(max(center_y, 0), hull.shape[0] - 1)
    center_inside = bool(hull[center_y, center_x])
    area_ratio = float(small_area / large_area)
    vote = bool(
        bbox_containment >= cfg.containment_bbox
        and hull_containment >= cfg.containment_hull
        and center_inside
        and area_ratio <= cfg.containment_max_area_ratio
    )
    return {
        "vote": vote,
        "bbox_containment": bbox_containment,
        "filled_mask_hull_containment": hull_containment,
        "small_center_inside_large_hull": center_inside,
        "area_ratio": area_ratio,
    }


def _camera_pose(camera_info: Mapping[str, Any], frame_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    raw = (camera_info.get("extrinsic") or {}).get(str(frame_id))
    if raw is None:
        return None
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid camera extrinsic for frame {frame_id}")
    extrinsic_type = str(camera_info.get("extrinsic_type", "world_to_camera"))
    if extrinsic_type == "world_to_camera":
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        center = -rotation.T @ translation
        forward = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    elif extrinsic_type == "camera_to_world":
        center = matrix[:3, 3]
        forward = matrix[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"unsupported extrinsic_type: {extrinsic_type!r}")
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError(f"degenerate camera view direction for frame {frame_id}")
    return center, forward / norm


def scene_diagonal(objects: Sequence[Mapping[str, Any]]) -> float:
    points = []
    for obj in objects:
        array = _object_points(obj)
        if array.ndim == 2 and array.shape[1:] == (3,) and len(array):
            points.append(array)
    if not points:
        return 1.0
    combined = np.concatenate(points, axis=0)
    diagonal = float(np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)))
    return diagonal if math.isfinite(diagonal) and diagonal > 1e-12 else 1.0


def cluster_pose_frames(
    frame_ids: Sequence[str],
    camera_info: Mapping[str, Any],
    *,
    scene_diagonal_value: float,
    thresholds: RelationThresholds | None = None,
) -> tuple[list[list[str]], list[str]]:
    cfg = thresholds or RelationThresholds()
    clusters: list[list[str]] = []
    representatives: list[tuple[np.ndarray, np.ndarray]] = []
    missing = []
    translation_threshold = (
        scene_diagonal_value * cfg.pose_translation_scene_fraction
    )
    for frame_id in sorted(set(str(value) for value in frame_ids)):
        pose = _camera_pose(camera_info, frame_id)
        if pose is None:
            missing.append(frame_id)
            continue
        center, forward = pose
        selected = None
        for index, (rep_center, rep_forward) in enumerate(representatives):
            dot = float(np.clip(np.dot(forward, rep_forward), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            translation = float(np.linalg.norm(center - rep_center))
            if (
                angle <= cfg.pose_angle_degrees
                and translation <= translation_threshold
            ):
                selected = index
                break
        if selected is None:
            representatives.append((center, forward))
            clusters.append([frame_id])
        else:
            clusters[selected].append(frame_id)
    return clusters, missing


def _cluster_boolean(values: Sequence[bool]) -> bool:
    return bool(np.mean(np.asarray(values, dtype=float)) >= 0.5)


def summarize_directional_2d(
    source_masks: Mapping[str, Any],
    target_masks: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    *,
    scene_diagonal_value: float,
    thresholds: RelationThresholds | None = None,
    background_pair: bool = False,
) -> dict[str, Any]:
    cfg = thresholds or RelationThresholds()
    shared = sorted(set(source_masks) & set(target_masks))
    clusters, missing_poses = cluster_pose_frames(
        shared,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
    )
    frame_values = {
        frame_id: frame_on_evidence(
            source_masks[frame_id],
            target_masks[frame_id],
            thresholds=cfg,
        )
        for frame_id in shared
    }
    cluster_values = []
    for cluster in clusters:
        values = [frame_values[frame] for frame in cluster if frame_values[frame] is not None]
        if not values:
            continue
        cluster_values.append(
            {
                "frame_ids": cluster,
                "above": _cluster_boolean([value["above"] for value in values]),
                "support": _cluster_boolean([value["support"] for value in values]),
                "vote": _cluster_boolean([value["vote"] for value in values]),
                "x_support": float(np.median([value["x_support"] for value in values])),
                "gap": float(np.median([value["gap"] for value in values])),
            }
        )
    count = len(cluster_values)
    above_vote = (
        float(np.mean([value["above"] for value in cluster_values])) if count else 0.0
    )
    support_vote = (
        float(np.mean([value["support"] for value in cluster_values])) if count else 0.0
    )
    median_x_support = (
        float(np.median([value["x_support"] for value in cluster_values]))
        if count
        else 0.0
    )
    minimum_clusters = (
        cfg.background_min_pose_clusters if background_pair else cfg.min_pose_clusters
    )
    minimum_support = (
        cfg.background_min_support_vote if background_pair else cfg.min_support_vote
    )
    candidate = bool(
        count >= minimum_clusters
        and above_vote >= cfg.min_above_vote
        and support_vote >= minimum_support
        and median_x_support >= cfg.min_median_x_support
    )
    return {
        "shared_frame_count": len(shared),
        "pose_cluster_count": count,
        "missing_pose_frame_ids": missing_poses,
        "above_vote": above_vote,
        "support_vote": support_vote,
        "median_x_support": median_x_support,
        "candidate": candidate,
        "clusters": cluster_values,
    }


def summarize_containment_2d(
    small_masks: Mapping[str, Any],
    large_masks: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    *,
    scene_diagonal_value: float,
    thresholds: RelationThresholds | None = None,
    background_pair: bool = False,
) -> dict[str, Any]:
    cfg = thresholds or RelationThresholds()
    shared = sorted(set(small_masks) & set(large_masks))
    clusters, missing_poses = cluster_pose_frames(
        shared,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
    )
    frame_values = {
        frame_id: frame_containment_evidence(
            small_masks[frame_id],
            large_masks[frame_id],
            thresholds=cfg,
        )
        for frame_id in shared
    }
    cluster_values = []
    for cluster in clusters:
        values = [frame_values[frame] for frame in cluster if frame_values[frame] is not None]
        if not values:
            continue
        cluster_values.append(
            {
                "frame_ids": cluster,
                "vote": _cluster_boolean([value["vote"] for value in values]),
                "bbox_containment": float(
                    np.median([value["bbox_containment"] for value in values])
                ),
                "filled_mask_hull_containment": float(
                    np.median(
                        [value["filled_mask_hull_containment"] for value in values]
                    )
                ),
                "area_ratio": float(
                    np.median([value["area_ratio"] for value in values])
                ),
            }
        )
    count = len(cluster_values)
    support = (
        float(np.mean([value["vote"] for value in cluster_values])) if count else 0.0
    )
    min_clusters = (
        cfg.background_min_pose_clusters if background_pair else cfg.min_pose_clusters
    )
    candidate = bool(
        count >= min_clusters and support >= cfg.containment_cluster_support
    )
    return {
        "shared_frame_count": len(shared),
        "pose_cluster_count": count,
        "missing_pose_frame_ids": missing_poses,
        "cluster_support": support,
        "candidate": candidate,
        "clusters": cluster_values,
    }


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _internal_spacing(points: np.ndarray) -> float | None:
    if len(points) < 2:
        return None
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(points).query(points, k=2)
        positive = distances[:, 1]
    except ImportError:  # pragma: no cover
        delta = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        distances[distances == 0] = np.inf
        positive = distances.min(axis=1)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return float(np.median(positive)) if len(positive) else None


def scale_independent_3d_evidence(
    first_points: Any,
    second_points: Any,
    *,
    thresholds: RelationThresholds | None = None,
) -> dict[str, Any]:
    cfg = thresholds or RelationThresholds()
    first = np.asarray(first_points, dtype=float)
    second = np.asarray(second_points, dtype=float)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1:] != (3,)
        or second.shape[1:] != (3,)
        or not len(first)
        or not len(second)
    ):
        return {
            "available": False,
            "candidate": False,
            "radius": None,
            "first_to_second_fraction": 0.0,
            "second_to_first_fraction": 0.0,
        }
    first = _sample_points(first, cfg.max_points_per_object)
    second = _sample_points(second, cfg.max_points_per_object)
    spacing_a = _internal_spacing(first)
    spacing_b = _internal_spacing(second)
    spacings = [value for value in (spacing_a, spacing_b) if value is not None]
    if not spacings:
        return {
            "available": False,
            "candidate": False,
            "radius": None,
            "first_to_second_fraction": 0.0,
            "second_to_first_fraction": 0.0,
        }
    radius = max(spacings) * cfg.neighbor_radius_spacing_multiplier
    from scipy.spatial import cKDTree

    distance_a, _ = cKDTree(second).query(first, k=1)
    distance_b, _ = cKDTree(first).query(second, k=1)
    fraction_a = float(np.mean(distance_a <= radius))
    fraction_b = float(np.mean(distance_b <= radius))
    candidate = bool(
        max(fraction_a, fraction_b) >= cfg.min_directional_neighbor_fraction
    )
    return {
        "available": True,
        "candidate": candidate,
        "spacing_first": spacing_a,
        "spacing_second": spacing_b,
        "radius": radius,
        "first_to_second_fraction": fraction_a,
        "second_to_first_fraction": fraction_b,
    }


def _object_masks(obj: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frame_ids = obj.get("frame_id") or []
    masks = obj.get("mask") or []
    if len(frame_ids) != len(masks):
        raise ValueError("object frame_id and mask lengths differ")
    result: dict[str, np.ndarray] = {}
    for frame_id, mask in zip(frame_ids, masks):
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2:
            raise ValueError(f"object mask for {frame_id} is not 2D")
        if binary.any():
            result[str(frame_id)] = binary
    return result


def _object_points(obj: Mapping[str, Any]) -> np.ndarray:
    if "pcd_np" in obj:
        return np.asarray(obj["pcd_np"], dtype=float)
    pcd = obj.get("pcd")
    if pcd is None:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pcd.points, dtype=float)


def _label(obj: Mapping[str, Any]) -> str:
    values = obj.get("class_name") or []
    if values:
        return " ".join(str(values[0]).lower().split())
    return " ".join(str(obj.get("v2m_name", "")).lower().split())


def pair_relation_evidence(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    *,
    scene_diagonal_value: float,
    thresholds: RelationThresholds | None = None,
) -> dict[str, Any]:
    cfg = thresholds or RelationThresholds()
    first_masks = _object_masks(first)
    second_masks = _object_masks(second)
    first_background = bool(first.get("is_background", False))
    second_background = bool(second.get("is_background", False))
    background_pair = first_background or second_background
    forbidden_background_pair = first_background and second_background
    first_on_second = summarize_directional_2d(
        first_masks,
        second_masks,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
        background_pair=background_pair,
    )
    second_on_first = summarize_directional_2d(
        second_masks,
        first_masks,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
        background_pair=background_pair,
    )
    first_in_second = summarize_containment_2d(
        first_masks,
        second_masks,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
        background_pair=background_pair,
    )
    second_in_first = summarize_containment_2d(
        second_masks,
        first_masks,
        camera_info,
        scene_diagonal_value=scene_diagonal_value,
        thresholds=cfg,
        background_pair=background_pair,
    )
    evidence_3d = scale_independent_3d_evidence(
        _object_points(first),
        _object_points(second),
        thresholds=cfg,
    )
    parent_ids_first = {
        str(value)
        for value in (
            list(first.get("parent_object_ids") or [])
            + list(first.get("parent_candidate_ids") or [])
        )
    }
    parent_ids_second = {
        str(value)
        for value in (
            list(second.get("parent_object_ids") or [])
            + list(second.get("parent_candidate_ids") or [])
        )
    }
    first_id = str(first.get("v2m_object_id", first.get("id", "")))
    second_id = str(second.get("v2m_object_id", second.get("id", "")))
    parent_hint = second_id in parent_ids_first or first_id in parent_ids_second
    reasons = []
    if not forbidden_background_pair:
        if first_on_second["candidate"]:
            reasons.append("multiview_first_on_second")
        if second_on_first["candidate"]:
            reasons.append("multiview_second_on_first")
        if first_in_second["candidate"]:
            reasons.append("multiview_first_in_second")
        if second_in_first["candidate"]:
            reasons.append("multiview_second_in_first")
        if evidence_3d["candidate"]:
            reasons.append("scale_independent_3d_neighbors")
        if parent_hint:
            reasons.append("parent_hint_recall_only")
    return {
        "labels": [_label(first), _label(second)],
        "geometry_types": [
            first.get("geometry_type", "colmap_3d" if len(_object_points(first)) else "multiview_2d"),
            second.get("geometry_type", "colmap_3d" if len(_object_points(second)) else "multiview_2d"),
        ],
        "background": [first_background, second_background],
        "background_background_forbidden": forbidden_background_pair,
        "first_on_second_2d": first_on_second,
        "second_on_first_2d": second_on_first,
        "first_in_second_2d": first_in_second,
        "second_in_first_2d": second_in_first,
        "scale_independent_3d": evidence_3d,
        "parent_hint": parent_hint,
        "candidate_reasons": reasons,
        "candidate": bool(reasons),
    }


def load_camera_info_from_objects(
    objects: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path | None]:
    paths = [
        Path(str(path))
        for obj in objects
        for path in (obj.get("color_path") or [])
        if str(path)
    ]
    if not paths:
        return {"extrinsic_type": "world_to_camera", "extrinsic": {}}, None
    project_roots = {
        path.expanduser().resolve().parents[2]
        for path in paths
        if len(path.expanduser().resolve().parents) >= 3
    }
    if len(project_roots) != 1:
        raise ValueError("map object frame paths do not identify one Video2Mesh project")
    camera_path = next(iter(project_roots)) / "scene" / "cameras" / "camera_info.json"
    if not camera_path.is_file():
        return {"extrinsic_type": "world_to_camera", "extrinsic": {}}, camera_path
    value = json.loads(camera_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"camera info must be a JSON object: {camera_path}")
    return value, camera_path


def mask_digest(obj: Mapping[str, Any]) -> str:
    rows = []
    for frame_id, mask in sorted(_object_masks(obj).items()):
        rows.append(
            {
                "frame_id": frame_id,
                "shape": list(mask.shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
                ).hexdigest(),
            }
        )
    return canonical_sha256(rows)
