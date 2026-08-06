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
    allow_cross_label_duplicates: bool = True
    duplicate_min_median_iou: float = 0.90
    duplicate_min_frame_iou: float = 0.85
    duplicate_min_frame_iou_fraction: float = 0.80
    fragment_min_median_containment: float = 0.95
    fragment_min_frame_containment: float = 0.90
    fragment_min_frame_containment_fraction: float = 0.75
    fragment_max_median_area_ratio: float = 0.75
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
        if not isinstance(self.allow_cross_label_duplicates, bool):
            raise ValueError("allow_cross_label_duplicates must be a boolean")
        for field in (
            "min_shorter_coverage",
            "min_median_iou",
            "min_frame_iou",
            "min_frame_iou_fraction",
            "duplicate_min_median_iou",
            "duplicate_min_frame_iou",
            "duplicate_min_frame_iou_fraction",
            "fragment_min_median_containment",
            "fragment_min_frame_containment",
            "fragment_min_frame_containment_fraction",
            "fragment_max_median_area_ratio",
            "anchor_separation_iou",
            "pillow_bed_min_containment",
            "pillow_bed_max_area_ratio",
        ):
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [0, 1]")


@dataclass(frozen=True)
class IdentityQualityConfig:
    min_track_frames: int = 3
    min_coverage_ratio: float = 0.70
    max_area_cv: float = 1.0
    max_area_step_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.min_track_frames <= 0:
            raise ValueError("min_track_frames must be positive")
        if not 0.0 <= float(self.min_coverage_ratio) <= 1.0:
            raise ValueError("min_coverage_ratio must lie in [0, 1]")
        if float(self.max_area_cv) <= 0:
            raise ValueError("max_area_cv must be positive")
        if float(self.max_area_step_ratio) < 1:
            raise ValueError("max_area_step_ratio must be at least 1")


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
    first_in_second: list[float] = []
    second_in_first: list[float] = []
    first_to_second_area_ratios: list[float] = []
    first_areas: list[int] = []
    second_areas: list[int] = []
    for frame_id in shared:
        first_mask = np.asarray(masks_a[frame_id], dtype=bool)
        second_mask = np.asarray(masks_b[frame_id], dtype=bool)
        first_area = int(first_mask.sum())
        second_area = int(second_mask.sum())
        if not first_area or not second_area:
            continue
        intersection = int(np.logical_and(first_mask, second_mask).sum())
        first_in_second.append(float(intersection / first_area))
        second_in_first.append(float(intersection / second_area))
        first_to_second_area_ratios.append(float(first_area / second_area))
        first_areas.append(first_area)
        second_areas.append(second_area)
    shorter = min(len(valid_a), len(valid_b))
    coverage = len(shared) / shorter if shorter else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    strong_fraction = (
        float(np.mean(np.asarray(ious) >= cfg.min_frame_iou)) if ious else 0.0
    )
    duplicate_strong_fraction = (
        float(np.mean(np.asarray(ious) >= cfg.duplicate_min_frame_iou))
        if ious
        else 0.0
    )
    first_containment_fraction = (
        float(
            np.mean(
                np.asarray(first_in_second)
                >= cfg.fragment_min_frame_containment
            )
        )
        if first_in_second
        else 0.0
    )
    second_containment_fraction = (
        float(
            np.mean(
                np.asarray(second_in_first)
                >= cfg.fragment_min_frame_containment
            )
        )
        if second_in_first
        else 0.0
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
        "duplicate_frame_iou_at_least_threshold_fraction": duplicate_strong_fraction,
        "median_first_in_second": (
            float(np.median(first_in_second)) if first_in_second else 0.0
        ),
        "median_second_in_first": (
            float(np.median(second_in_first)) if second_in_first else 0.0
        ),
        "first_in_second_at_least_threshold_fraction": first_containment_fraction,
        "second_in_first_at_least_threshold_fraction": second_containment_fraction,
        "median_first_to_second_area_ratio": (
            float(np.median(first_to_second_area_ratios))
            if first_to_second_area_ratios
            else 0.0
        ),
        "median_first_area": float(np.median(first_areas)) if first_areas else 0.0,
        "median_second_area": float(np.median(second_areas)) if second_areas else 0.0,
        "forbidden": bool(forbidden),
        "compatible": compatible,
    }


def complete_link_track_clusters(
    tracks: Sequence[Mapping[str, Any]],
    *,
    config: TrackMergeConfig | None = None,
    forbidden_pairs: Iterable[tuple[str, str]] = (),
) -> tuple[list[list[int]], dict[str, dict[str, Any]]]:
    """Cluster duplicate tracks without allowing transitive chain merges."""

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
            same_label = normalize_label(first.get("label")) == normalize_label(
                second.get("label")
            )
            metrics = track_pair_metrics(
                first,
                second,
                config=cfg,
                forbidden=frozenset((id_a, id_b)) in forbidden,
            )
            metrics["object_ids"] = [id_a, id_b]
            metrics["same_label"] = same_label
            same_label_match = bool(same_label and metrics["compatible"])
            cross_label_duplicate = bool(
                cfg.allow_cross_label_duplicates
                and not same_label
                and not metrics["forbidden"]
                and metrics["shared_nonempty_frames"]
                >= cfg.min_shared_nonempty_frames
                and metrics["shorter_track_coverage"]
                >= cfg.min_shorter_coverage
                and metrics["median_mask_iou"]
                >= cfg.duplicate_min_median_iou
                and metrics["duplicate_frame_iou_at_least_threshold_fraction"]
                >= cfg.duplicate_min_frame_iou_fraction
            )
            forbidden_overlap_conflict = bool(
                metrics["forbidden"]
                and same_label
                and metrics["shared_nonempty_frames"]
                >= cfg.min_shared_nonempty_frames
                and metrics["median_mask_iou"]
                >= cfg.duplicate_min_median_iou
            )
            metrics["same_label_match"] = same_label_match
            metrics["cross_label_duplicate"] = cross_label_duplicate
            metrics["forbidden_overlap_conflict"] = forbidden_overlap_conflict
            metrics["merge_candidate"] = bool(
                same_label_match or cross_label_duplicate
            )
            metrics["merge_kind"] = (
                "same_label_duplicate"
                if same_label_match
                else (
                    "cross_label_duplicate"
                    if cross_label_duplicate
                    else None
                )
            )
            metrics["compatible"] = metrics["merge_candidate"]
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


def _median_track_area(track: Mapping[str, Any]) -> float:
    areas = [
        int(np.asarray(mask, dtype=bool).sum())
        for mask in (track.get("masks") or {}).values()
        if np.asarray(mask, dtype=bool).any()
    ]
    return float(np.median(areas)) if areas else 0.0


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
    *,
    canonical_index: int | None = None,
    preserve_canonical_masks: bool = False,
) -> dict[str, Any]:
    if not cluster:
        raise ValueError("cannot merge an empty track cluster")
    if canonical_index is None:
        canonical_index = canonical_track_index(cluster, tracks)
    if canonical_index not in cluster:
        raise ValueError("canonical_index must be a member of cluster")
    canonical = dict(tracks[canonical_index])
    merged_masks: dict[str, np.ndarray] = {}
    ordered_indices = [canonical_index] + [
        index for index in cluster if index != canonical_index
    ]
    for index in ordered_indices:
        for frame_id, mask in (tracks[index].get("masks") or {}).items():
            binary = np.asarray(mask, dtype=bool)
            if frame_id in merged_masks:
                if merged_masks[frame_id].shape != binary.shape:
                    raise ValueError(f"mask shape mismatch in merged frame {frame_id}")
                if not preserve_canonical_masks or not merged_masks[frame_id].any():
                    merged_masks[frame_id] = np.logical_or(
                        merged_masks[frame_id], binary
                    )
            else:
                merged_masks[str(frame_id)] = binary.copy()
    source_ids = sorted(
        {
            str(source_id)
            for index in cluster
            for source_id in (
                tracks[index].get("source_object_ids")
                or [tracks[index]["object_id"]]
            )
        }
    )
    source_labels = sorted(
        {
            normalize_label(source_label)
            for index in cluster
            for source_label in (
                tracks[index].get("source_labels")
                or [tracks[index].get("label")]
            )
            if normalize_label(source_label)
        }
    )
    canonical["masks"] = merged_masks
    canonical["source_object_ids"] = source_ids
    canonical["source_labels"] = source_labels
    canonical["canonical_object_id"] = str(tracks[canonical_index]["object_id"])
    canonical["object_id"] = canonical["canonical_object_id"]
    canonical["valid_frame_count"] = int(
        sum(bool(mask.any()) for mask in merged_masks.values())
    )
    canonical["area_cv"] = _area_stability(canonical)
    return canonical


def _track_source_ids(track: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(value)
        for value in (track.get("source_object_ids") or [track["object_id"]])
    )


def _fragment_candidate(
    metrics: Mapping[str, Any],
    *,
    first_is_child: bool,
    config: TrackMergeConfig,
) -> dict[str, Any] | None:
    if metrics.get("forbidden"):
        return None
    if (
        int(metrics.get("shared_nonempty_frames", 0))
        < config.min_shared_nonempty_frames
    ):
        return None
    if float(metrics.get("shorter_track_coverage", 0.0)) < config.min_shorter_coverage:
        return None
    if first_is_child:
        containment = float(metrics.get("median_first_in_second", 0.0))
        containment_fraction = float(
            metrics.get("first_in_second_at_least_threshold_fraction", 0.0)
        )
        area_ratio = float(metrics.get("median_first_to_second_area_ratio", 0.0))
        child_area = float(metrics.get("median_first_area", 0.0))
        parent_area = float(metrics.get("median_second_area", 0.0))
    else:
        containment = float(metrics.get("median_second_in_first", 0.0))
        containment_fraction = float(
            metrics.get("second_in_first_at_least_threshold_fraction", 0.0)
        )
        raw_ratio = float(metrics.get("median_first_to_second_area_ratio", 0.0))
        area_ratio = float(1.0 / raw_ratio) if raw_ratio > 0 else math.inf
        child_area = float(metrics.get("median_second_area", 0.0))
        parent_area = float(metrics.get("median_first_area", 0.0))
    if not (
        child_area > 0
        and parent_area > child_area
        and containment >= config.fragment_min_median_containment
        and containment_fraction
        >= config.fragment_min_frame_containment_fraction
        and area_ratio <= config.fragment_max_median_area_ratio
    ):
        return None
    return {
        "median_containment": containment,
        "containment_at_least_threshold_fraction": containment_fraction,
        "median_child_to_parent_area_ratio": area_ratio,
        "median_child_area": child_area,
        "median_parent_area": parent_area,
        "shared_nonempty_frames": int(metrics["shared_nonempty_frames"]),
        "shorter_track_coverage": float(metrics["shorter_track_coverage"]),
    }


def resolve_track_identities(
    tracks: Sequence[Mapping[str, Any]],
    *,
    config: TrackMergeConfig | None = None,
    forbidden_pairs: Iterable[tuple[str, str]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve exact duplicates first, then attach stable same-label fragments.

    Exact duplicate clustering remains complete-link to prevent transitive
    identity collapse. Multiple disjoint fragments may attach to a larger
    same-label track. Every attachment is directional (strictly smaller to
    larger), and attachment to independent parents remains unresolved.
    """

    cfg = config or TrackMergeConfig()
    forbidden = {
        frozenset((str(first), str(second)))
        for first, second in forbidden_pairs
    }
    base_clusters, pair_metrics = complete_link_track_clusters(
        tracks,
        config=cfg,
        forbidden_pairs=(tuple(pair) for pair in forbidden),
    )
    base_tracks = [
        merge_track_cluster(cluster, tracks) for cluster in base_clusters
    ]

    def base_pair_is_forbidden(left: int, right: int) -> bool:
        return any(
            frozenset((first, second)) in forbidden
            for first in _track_source_ids(base_tracks[left])
            for second in _track_source_ids(base_tracks[right])
        )

    fragment_candidates: list[dict[str, Any]] = []
    by_child: dict[int, list[dict[str, Any]]] = {}
    for left in range(len(base_tracks)):
        for right in range(left + 1, len(base_tracks)):
            first = base_tracks[left]
            second = base_tracks[right]
            if normalize_label(first.get("label")) != normalize_label(
                second.get("label")
            ):
                continue
            metrics = track_pair_metrics(
                first,
                second,
                config=cfg,
                forbidden=base_pair_is_forbidden(left, right),
            )
            for child, parent, first_is_child in (
                (left, right, True),
                (right, left, False),
            ):
                fragment_metrics = _fragment_candidate(
                    metrics,
                    first_is_child=first_is_child,
                    config=cfg,
                )
                if fragment_metrics is None:
                    continue
                record = {
                    "child_base_index": child,
                    "parent_base_index": parent,
                    "child_object_id": str(base_tracks[child]["object_id"]),
                    "parent_object_id": str(base_tracks[parent]["object_id"]),
                    "child_source_object_ids": _track_source_ids(base_tracks[child]),
                    "parent_source_object_ids": _track_source_ids(base_tracks[parent]),
                    "label": normalize_label(base_tracks[child].get("label")),
                    **fragment_metrics,
                }
                fragment_candidates.append(record)
                by_child.setdefault(child, []).append(record)

    attachments: dict[int, int] = {}
    selected_fragment_records: list[dict[str, Any]] = []
    for child, candidates in sorted(
        by_child.items(),
        key=lambda item: (
            _median_track_area(base_tracks[item[0]]),
            str(base_tracks[item[0]]["object_id"]),
        ),
    ):
        selected = max(
            candidates,
            key=lambda item: (
                item["containment_at_least_threshold_fraction"],
                item["median_containment"],
                item["median_parent_area"],
                -item["median_child_to_parent_area_ratio"],
                item["shared_nonempty_frames"],
                item["parent_object_id"],
            ),
        )
        attachments[child] = int(selected["parent_base_index"])
        selected_fragment_records.append(dict(selected))

    def root_for(index: int) -> int:
        visited: set[int] = set()
        current = index
        while current in attachments:
            if current in visited:
                raise ValueError("fragment identity attachments contain a cycle")
            visited.add(current)
            current = attachments[current]
        return current

    groups: dict[int, list[int]] = {}
    for index in range(len(base_tracks)):
        groups.setdefault(root_for(index), []).append(index)

    final_tracks: list[dict[str, Any]] = []
    final_root_by_base: dict[int, int] = {}
    for root, members in sorted(
        groups.items(), key=lambda item: str(base_tracks[item[0]]["object_id"])
    ):
        for member in members:
            final_root_by_base[member] = root
        if len(members) == 1:
            merged = dict(base_tracks[root])
        else:
            merged = merge_track_cluster(
                members,
                base_tracks,
                canonical_index=root,
                preserve_canonical_masks=True,
            )
        merged["identity_resolution"] = {
            "canonical_object_id": str(merged["object_id"]),
            "source_object_ids": _track_source_ids(merged),
            "source_labels": list(merged.get("source_labels") or []),
            "fragment_source_object_ids": sorted(
                {
                    source_id
                    for member in members
                    if member != root
                    for source_id in _track_source_ids(base_tracks[member])
                }
            ),
        }
        final_tracks.append(merged)

    source_to_base = {
        source_id: base_index
        for base_index, track in enumerate(base_tracks)
        for source_id in _track_source_ids(track)
    }
    source_to_canonical = {
        source_id: str(base_tracks[final_root_by_base[base_index]]["object_id"])
        for source_id, base_index in source_to_base.items()
    }

    duplicate_clusters = []
    for cluster, base_track in zip(base_clusters, base_tracks):
        if len(cluster) <= 1:
            continue
        source_ids = sorted(str(tracks[index]["object_id"]) for index in cluster)
        labels = sorted(
            {
                normalize_label(tracks[index].get("label"))
                for index in cluster
            }
        )
        kinds = sorted(
            {
                str(metrics["merge_kind"])
                for metrics in pair_metrics.values()
                if metrics.get("merge_kind")
                and set(metrics.get("object_ids") or []).issubset(source_ids)
            }
        )
        duplicate_clusters.append(
            {
                "canonical_object_id": str(base_track["object_id"]),
                "source_object_ids": source_ids,
                "source_labels": labels,
                "merge_kinds": kinds,
            }
        )

    unresolved_candidates = []
    forbidden_overlap_conflicts = []
    for metrics in pair_metrics.values():
        first_id, second_id = [str(value) for value in metrics["object_ids"]]
        if metrics.get("forbidden_overlap_conflict"):
            forbidden_overlap_conflicts.append(
                {
                    "object_ids": [first_id, second_id],
                    "median_mask_iou": metrics["median_mask_iou"],
                    "shared_nonempty_frames": metrics["shared_nonempty_frames"],
                }
            )
        if (
            metrics.get("merge_candidate")
            and source_to_canonical.get(first_id)
            != source_to_canonical.get(second_id)
        ):
            unresolved_candidates.append(
                {
                    "kind": metrics.get("merge_kind"),
                    "object_ids": [first_id, second_id],
                    "median_mask_iou": metrics["median_mask_iou"],
                    "shared_nonempty_frames": metrics["shared_nonempty_frames"],
                }
            )

    for record in fragment_candidates:
        child_root = final_root_by_base[int(record["child_base_index"])]
        parent_root = final_root_by_base[int(record["parent_base_index"])]
        record["resolved"] = child_root == parent_root
        record["canonical_object_id"] = str(
            base_tracks[child_root]["object_id"]
        )
        if not record["resolved"]:
            unresolved_candidates.append(
                {
                    "kind": "same_label_fragment",
                    "object_ids": [
                        record["child_object_id"],
                        record["parent_object_id"],
                    ],
                    "median_containment": record["median_containment"],
                    "shared_nonempty_frames": record["shared_nonempty_frames"],
                }
            )

    resolution = {
        "schema_version": 1,
        "method": "complete_link_duplicates_then_fragment_attachment",
        "config": asdict(cfg),
        "raw_track_count": len(tracks),
        "base_track_count": len(base_tracks),
        "final_track_count": len(final_tracks),
        "duplicate_clusters": duplicate_clusters,
        "fragment_candidates": fragment_candidates,
        "fragment_attachments": selected_fragment_records,
        "source_to_canonical_object_id": source_to_canonical,
        "unresolved_candidates": unresolved_candidates,
        "forbidden_overlap_conflicts": forbidden_overlap_conflicts,
        "ok": not unresolved_candidates and not forbidden_overlap_conflicts,
    }
    return final_tracks, {
        "pair_metrics": pair_metrics,
        "base_clusters": base_clusters,
        "resolution": resolution,
    }


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
    merged, identity = resolve_track_identities(
        tracks,
        config=cfg,
        forbidden_pairs=(tuple(item) for item in forbidden_pairs),
    )
    pair_metrics = identity["pair_metrics"]
    identity_resolution = identity["resolution"]

    labels: dict[str, Any] = {}
    normalized_objects: dict[str, Any] = {}
    for track in merged:
        object_id = str(track["object_id"])
        prompt_sources = [
            source_track["prompt"]
            for source_track in tracks
            if source_track["object_id"] in track["source_object_ids"]
        ]
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
            "source_labels": list(track.get("source_labels") or []),
            "identity_resolution": dict(
                track.get("identity_resolution") or {}
            ),
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
            "source_labels": normalized_objects[object_id]["source_labels"],
            "identity_resolution": normalized_objects[object_id][
                "identity_resolution"
            ],
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
        "schema_version": 3,
        "method": "conceptgraphs_identity_track_normalization_v2",
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
                "canonical_object_id": track["object_id"],
                "source_object_ids": track["source_object_ids"],
                "source_labels": track.get("source_labels", []),
            }
            for track in merged
        ],
        "pair_metrics": pair_metrics,
        "identity_resolution": identity_resolution,
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


def _expected_frame_ids(root: Path) -> list[str]:
    manifest_path = root / "scene" / "frames_manifest.json"
    if manifest_path.is_file():
        payload = _read_json(manifest_path)
        frame_ids = [
            str(record["frame_id"])
            for record in payload.get("frames", [])
            if isinstance(record, Mapping) and "frame_id" in record
        ]
        if frame_ids:
            return sorted(set(frame_ids))
    frames_root = root / "scene" / "frames"
    return sorted(
        path.stem
        for path in frames_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if frames_root.is_dir() else []


def build_identity_quality_report(
    project_root: str | Path,
    *,
    config: IdentityQualityConfig | None = None,
) -> dict[str, Any]:
    """Inspect normalized identities without changing project artifacts."""

    cfg = config or IdentityQualityConfig()
    root = Path(project_root).expanduser().resolve()
    tracking_path = root / "masks" / "2d" / "tracking_manifest.json"
    if not tracking_path.is_file():
        raise FileNotFoundError(
            f"normalized tracking manifest not found: {tracking_path}"
        )
    manifest = _read_json(tracking_path)
    mask_root = Path(str(manifest.get("mask_root", ""))).expanduser().resolve()
    expected_mask_root = (root / "masks" / "2d").resolve()
    if mask_root != expected_mask_root:
        raise ValueError(
            f"normalized mask_root is {mask_root}, expected {expected_mask_root}"
        )

    expected_frames = _expected_frame_ids(root)
    expected_frame_set = set(expected_frames)
    resolution = manifest.get("identity_resolution")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(resolution, Mapping):
        errors.append(
            {
                "name": "missing_identity_resolution",
                "detail": "tracking manifest has no identity_resolution record",
            }
        )
        resolution = {}

    unresolved = list(resolution.get("unresolved_candidates") or [])
    forbidden_conflicts = list(
        resolution.get("forbidden_overlap_conflicts") or []
    )
    for item in unresolved:
        errors.append(
            {
                "name": "unresolved_identity_candidate",
                "detail": item,
            }
        )
    for item in forbidden_conflicts:
        errors.append(
            {
                "name": "forbidden_overlap_conflict",
                "detail": item,
            }
        )

    clusters = [
        item for item in manifest.get("clusters", []) if isinstance(item, Mapping)
    ]
    source_assignments: dict[str, list[str]] = {}
    for cluster in clusters:
        canonical = str(cluster.get("canonical_object_id", ""))
        for source_id in cluster.get("source_object_ids") or []:
            source_assignments.setdefault(str(source_id), []).append(canonical)
    for source_id, canonical_ids in sorted(source_assignments.items()):
        unique = sorted(set(canonical_ids))
        if len(unique) > 1:
            errors.append(
                {
                    "name": "source_assigned_to_multiple_instances",
                    "detail": {
                        "source_object_id": source_id,
                        "canonical_object_ids": unique,
                    },
                }
            )

    object_reports = []
    for object_id, record in sorted((manifest.get("objects") or {}).items()):
        if not isinstance(record, Mapping):
            continue
        frame_records = [
            item
            for item in record.get("frames", [])
            if isinstance(item, Mapping) and item.get("frame_id")
        ]
        areas: list[int] = []
        readable_frames: list[str] = []
        missing_files: list[str] = []
        for frame in sorted(frame_records, key=lambda item: str(item["frame_id"])):
            frame_id = str(frame["frame_id"])
            mask_path = Path(
                str(frame.get("mask") or mask_root / object_id / f"{frame_id}.png")
            ).expanduser()
            if not mask_path.is_file():
                missing_files.append(str(mask_path))
                continue
            mask = _load_mask(mask_path)
            area = int(mask.sum())
            if area <= 0:
                continue
            areas.append(area)
            readable_frames.append(frame_id)

        covered = set(readable_frames)
        coverage_ratio = (
            len(covered & expected_frame_set) / len(expected_frame_set)
            if expected_frame_set
            else (1.0 if covered else 0.0)
        )
        area_mean = float(np.mean(areas)) if areas else 0.0
        area_cv = (
            float(np.std(areas) / area_mean) if areas and area_mean > 0 else None
        )
        area_step_ratios = [
            float(max(first, second) / min(first, second))
            for first, second in zip(areas, areas[1:])
            if first > 0 and second > 0
        ]
        max_area_step_ratio = max(area_step_ratios, default=1.0)
        issues: list[dict[str, Any]] = []
        if missing_files:
            issues.append(
                {
                    "name": "missing_mask_files",
                    "severity": "error",
                    "detail": missing_files[:20],
                }
            )
        if not areas:
            issues.append(
                {
                    "name": "empty_track",
                    "severity": "error",
                    "detail": "no readable non-empty normalized masks",
                }
            )
        if len(covered) < cfg.min_track_frames:
            issues.append(
                {
                    "name": "short_track",
                    "severity": "warning",
                    "detail": (
                        f"frames={len(covered)}, minimum={cfg.min_track_frames}"
                    ),
                }
            )
        if coverage_ratio < cfg.min_coverage_ratio:
            issues.append(
                {
                    "name": "low_track_coverage",
                    "severity": "warning",
                    "detail": (
                        f"coverage={coverage_ratio:.3f}, "
                        f"minimum={cfg.min_coverage_ratio:.3f}"
                    ),
                }
            )
        if area_cv is not None and area_cv > cfg.max_area_cv:
            issues.append(
                {
                    "name": "unstable_mask_area",
                    "severity": "warning",
                    "detail": (
                        f"area_cv={area_cv:.3f}, maximum={cfg.max_area_cv:.3f}"
                    ),
                }
            )
        if max_area_step_ratio > cfg.max_area_step_ratio:
            issues.append(
                {
                    "name": "abrupt_mask_area_change",
                    "severity": "warning",
                    "detail": (
                        f"max_area_step_ratio={max_area_step_ratio:.3f}, "
                        f"maximum={cfg.max_area_step_ratio:.3f}"
                    ),
                }
            )
        source_labels = sorted(
            {
                normalize_label(value)
                for value in record.get("source_labels") or []
                if normalize_label(value)
            }
        )
        if len(source_labels) > 1:
            issues.append(
                {
                    "name": "cross_label_identity_merge",
                    "severity": "warning",
                    "detail": {
                        "canonical_label": normalize_label(
                            record.get("name") or record.get("category")
                        ),
                        "source_labels": source_labels,
                    },
                }
            )
        for issue in issues:
            target = errors if issue["severity"] == "error" else warnings
            target.append(
                {
                    "name": issue["name"],
                    "object_id": str(object_id),
                    "detail": issue["detail"],
                }
            )
        object_reports.append(
            {
                "object_id": str(object_id),
                "source_object_ids": sorted(
                    str(value) for value in record.get("source_object_ids") or []
                ),
                "source_labels": source_labels,
                "covered_frame_count": len(covered),
                "expected_frame_count": len(expected_frames),
                "coverage_ratio": coverage_ratio,
                "mask_area_mean": area_mean,
                "mask_area_cv": area_cv,
                "max_area_step_ratio": max_area_step_ratio,
                "issues": issues,
            }
        )

    ok = not errors
    return {
        "schema_version": 1,
        "project_root": str(root),
        "tracking_manifest": str(tracking_path),
        "tracking_manifest_sha256": hashlib.sha256(
            tracking_path.read_bytes()
        ).hexdigest(),
        "status": "identity_ready" if ok else "unresolved_identity_conflicts",
        "ok": ok,
        "quality_clean": ok and not warnings,
        "config": asdict(cfg),
        "summary": {
            "raw_track_count": int(resolution.get("raw_track_count", 0) or 0),
            "canonical_track_count": len(object_reports),
            "duplicate_cluster_count": len(
                resolution.get("duplicate_clusters") or []
            ),
            "fragment_attachment_count": len(
                resolution.get("fragment_attachments") or []
            ),
            "cross_label_merge_count": sum(
                len(item.get("source_labels") or []) > 1
                for item in object_reports
            ),
            "unresolved_candidate_count": len(unresolved),
            "forbidden_overlap_conflict_count": len(forbidden_conflicts),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "identity_resolution": dict(resolution),
        "objects": object_reports,
        "errors": errors,
        "warnings": warnings,
    }


def inspect_identity_quality_project(
    project_root: str | Path,
    *,
    config: IdentityQualityConfig | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    report = build_identity_quality_report(root, config=config)
    output = root / "simulator_assets" / "identity_quality_report.json"
    _write_json_atomic(output, report)
    project_manifest_path = root / "manifest.json"
    if project_manifest_path.is_file():
        project_manifest = _read_json(project_manifest_path)
        project_manifest.setdefault("artifacts", {})[
            "identity_quality_report"
        ] = str(output)
        project_manifest.setdefault("external_stages", {})[
            "conceptgraphs_identity_quality"
        ] = {
            "status": report["status"],
            "ok": report["ok"],
            "report": str(output),
        }
        _write_json_atomic(project_manifest_path, project_manifest)
    return report


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
        choices=(
            "normalize-prompts",
            "normalize-tracks",
            "inspect-identities",
            "finalize-fusion",
        ),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--quality-config-json", default="{}")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    args = parser.parse_args()
    try:
        config_value = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--config-json must contain a JSON object") from exc
    if not isinstance(config_value, dict):
        raise ValueError("--config-json must contain a JSON object")
    try:
        quality_config_value = json.loads(args.quality_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--quality-config-json must contain a JSON object"
        ) from exc
    if not isinstance(quality_config_value, dict):
        raise ValueError("--quality-config-json must contain a JSON object")
    config = TrackMergeConfig(**config_value)
    quality_config = IdentityQualityConfig(**quality_config_value)
    if args.stage == "normalize-prompts":
        result = normalize_prompts_project(args.project_root, config=config)
    elif args.stage == "normalize-tracks":
        result = normalize_tracks_project(args.project_root, config=config)
    elif args.stage == "inspect-identities":
        result = inspect_identity_quality_project(
            args.project_root,
            config=quality_config,
        )
    else:
        result = finalize_fusion_manifest(args.project_root)
    print(
        json.dumps(
            {
                "ok": bool(result.get("ok", True)),
                "stage": args.stage,
                "sha256": canonical_sha256(result),
            }
        )
    )
    if (
        args.stage == "inspect-identities"
        and args.fail_on_unresolved
        and not result["ok"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
