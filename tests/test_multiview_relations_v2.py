from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from conceptgraph.integrations.video2mesh.object_normalization import (
    TrackMergeConfig,
    carve_child_masks_from_parent,
    complete_link_track_clusters,
    mask_iou,
    merge_track_cluster,
    normalize_detection_manifest,
    normalize_tracks_project,
)
from conceptgraph.scenegraph.build_scenegraph_cfslam import (
    _compute_local_overlap_matrix,
    _refinement_request_is_reusable,
)
from conceptgraph.scenegraph.multiview_relations import (
    RelationThresholds,
    pair_relation_evidence,
    scale_independent_3d_evidence,
)


def _rect(top: int, left: int, bottom: int, right: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _camera_info(frame_count: int, translation_step: float = 1.0) -> dict:
    extrinsic = {}
    for index in range(frame_count):
        matrix = np.eye(4)
        matrix[0, 3] = -index * translation_step
        extrinsic[f"{index:06d}"] = matrix.tolist()
    return {
        "extrinsic_type": "world_to_camera",
        "extrinsic": extrinsic,
    }


def _track(object_id: str, label: str, masks: dict[str, np.ndarray], score=0.5):
    return {
        "object_id": object_id,
        "label": label,
        "masks": masks,
        "detection_confidence": score,
    }


def test_detection_normalization_preserves_pillows_and_absorbs_bed_parts() -> None:
    raw = [
        {"frame_id": "000000", "label": "bed", "bbox": [5, 30, 95, 95], "score": 0.9},
        {"frame_id": "000000", "label": "mattress", "bbox": [8, 35, 92, 90], "score": 0.8},
        {"frame_id": "000000", "label": "pillow", "bbox": [15, 40, 35, 55], "score": 0.8},
        {"frame_id": "000000", "label": "pillow", "bbox": [60, 40, 80, 55], "score": 0.7},
        {"frame_id": "000001", "label": "pillow", "bbox": [16, 40, 36, 55], "score": 0.6},
    ]
    result = normalize_detection_manifest(
        {
            "method": "groundingdino_object_level_discovery",
            "raw_detections": raw,
            "objects": [],
        }
    )
    beds = [item for item in result["objects"] if item["name"] == "bed"]
    pillows = [item for item in result["objects"] if item["name"] == "pillow"]
    assert len(beds) == 1
    assert len(pillows) == 3
    assert all(item["object_id"] != beds[0]["object_id"] for item in pillows)
    assert all(item["parent_candidate_ids"] == ["cg_bed"] for item in pillows)
    forbidden = result["normalization"]["forbidden_merge_pairs"]
    same_frame_ids = sorted(item["object_id"] for item in pillows if item["frame_id"] == "000000")
    assert same_frame_ids in forbidden
    assert len(beds[0]["source_detection_ids"]) == 2


def test_complete_link_merges_four_lamp_tracks_into_two_without_chaining() -> None:
    frames = [f"{index:06d}" for index in range(6)]
    left = {frame: _rect(20, 10, 50, 30) for frame in frames}
    left_jitter = {frame: _rect(20, 11, 50, 31) for frame in frames}
    right = {frame: _rect(20, 70, 50, 90) for frame in frames}
    right_jitter = {frame: _rect(20, 69, 50, 89) for frame in frames}
    tracks = [
        _track("lamp_left_a", "lamp", left, 0.7),
        _track("lamp_left_b", "lamp", left_jitter, 0.8),
        _track("lamp_right_a", "lamp", right, 0.6),
        _track("lamp_right_b", "lamp", right_jitter, 0.9),
    ]
    clusters, _ = complete_link_track_clusters(tracks)
    assert sorted(len(cluster) for cluster in clusters) == [2, 2]
    merged = [merge_track_cluster(cluster, tracks) for cluster in clusters]
    assert {item["object_id"] for item in merged} == {"lamp_left_b", "lamp_right_b"}

    # Complete-link rejects a chain when A~B and B~C but A!~C.
    config = TrackMergeConfig(
        min_shared_nonempty_frames=1,
        min_shorter_coverage=1.0,
        min_median_iou=0.3,
        min_frame_iou=0.3,
        min_frame_iou_fraction=1.0,
    )
    chain = [
        _track("a", "lamp", {"0": _rect(10, 10, 30, 30)}),
        _track("b", "lamp", {"0": _rect(10, 18, 30, 38)}),
        _track("c", "lamp", {"0": _rect(10, 26, 30, 46)}),
    ]
    clusters, _ = complete_link_track_clusters(chain, config=config)
    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_empty_masks_do_not_enter_iou_and_pillow_carve_is_fusion_only() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    assert mask_iou(empty, empty) is None
    bed = {"0": _rect(20, 10, 90, 90), "1": _rect(20, 10, 90, 90)}
    pillow = {"0": _rect(30, 25, 45, 45), "1": _rect(30, 25, 45, 45)}
    pillow_before = {key: value.copy() for key, value in pillow.items()}
    carved, report = carve_child_masks_from_parent(bed, pillow, dilation_px=2)
    assert report["accepted"] is True
    assert report["child_masks_modified"] is False
    assert all(carved[key].sum() < bed[key].sum() for key in bed)
    for key in pillow:
        np.testing.assert_array_equal(pillow[key], pillow_before[key])


def test_track_normalization_accepts_init_directory_and_writes_native_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    masks_root = root / "masks"
    raw_track = masks_root / "2d_raw" / "lamp"
    raw_track.mkdir(parents=True)
    # Video2Mesh init creates this empty output directory before tracking.
    (masks_root / "2d").mkdir()
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:10, 4:12] = 255
    Image.fromarray(mask, mode="L").save(raw_track / "000000.png")
    (masks_root / "object_prompts_normalized.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "lamp",
                        "name": "lamp",
                        "category": "lamp",
                        "description": "test lamp",
                        "score": 0.9,
                    }
                ],
                "normalization": {"forbidden_merge_pairs": []},
            }
        ),
        encoding="utf-8",
    )

    result = normalize_tracks_project(root)

    manifest = json.loads(
        (masks_root / "2d" / "tracking_manifest.json").read_text(encoding="utf-8")
    )
    assert result["objects"]["lamp"]["valid_frame_count"] == 1
    assert type(manifest["objects"]["lamp"]["valid_frame_count"]) is int
    assert (masks_root / "2d_fusion" / "tracking_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="output already exists"):
        normalize_tracks_project(root)


def test_legacy_overlap_skips_objects_without_3d_bbox() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]])
    cube = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    bbox = SimpleNamespace(get_box_points=lambda: cube)
    objects = [
        {"pcd": SimpleNamespace(points=points), "bbox": bbox},
        {"pcd": SimpleNamespace(points=points.copy()), "bbox": bbox},
        {
            "pcd": SimpleNamespace(points=np.empty((0, 3))),
            "bbox": None,
            "geometry_type": "multiview_2d",
        },
        {},
    ]

    overlaps = _compute_local_overlap_matrix(objects, 0.025)

    assert overlaps.shape == (4, 4)
    assert overlaps[0, 1] == pytest.approx(1.0)
    assert overlaps[1, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(overlaps[2:, :], 0.0)
    np.testing.assert_array_equal(overlaps[:, 2:], 0.0)


def test_refinement_cache_survives_unrelated_appended_caption_entries() -> None:
    settings = {
        "schema_version": 2,
        "provider": "openai-compatible-responses",
        "base_url": "https://example.test/v1",
        "model": "test-model",
        "prompt_sha256": "prompt",
        "caption_file": "/tmp/captions.json",
        "caption_file_sha256": "old-complete-file",
        "mapfile": "/tmp/map.pkl.gz",
        "mapfile_sha256": "map",
    }
    cached = {
        "settings": settings,
        "input": {"id": 0, "captions": ["same object caption"]},
    }
    current = {
        "settings": {
            **settings,
            "caption_file_sha256": "new-file-with-appended-backgrounds",
        },
        "input": {"id": 0, "captions": ["same object caption"]},
    }

    assert _refinement_request_is_reusable(cached, current) is True
    current["input"]["captions"] = ["changed object caption"]
    assert _refinement_request_is_reusable(cached, current) is False
    current["input"] = cached["input"]
    current["settings"]["model"] = "different-model"
    assert _refinement_request_is_reusable(cached, current) is False


def _object(
    label: str,
    masks: dict[str, np.ndarray],
    *,
    points: np.ndarray | None = None,
    background: bool = False,
) -> dict:
    point_array = (
        np.empty((0, 3), dtype=float) if points is None else np.asarray(points, dtype=float)
    )
    return {
        "v2m_object_id": label,
        "class_name": [label] * len(masks),
        "frame_id": list(masks),
        "mask": list(masks.values()),
        "pcd_np": point_array,
        "geometry_type": "colmap_3d" if len(point_array) else "multiview_2d",
        "is_background": background,
    }


def test_multiview_on_reverse_containment_and_background_thresholds() -> None:
    frames = [f"{index:06d}" for index in range(5)]
    lamp_masks = {frame: _rect(20, 35, 45, 55) for frame in frames}
    table_masks = {frame: _rect(47, 20, 80, 80) for frame in frames}
    lamp = _object("lamp", lamp_masks)
    table = _object(
        "nightstand",
        table_masks,
        points=np.asarray([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]]),
    )
    evidence = pair_relation_evidence(
        lamp,
        table,
        _camera_info(5),
        scene_diagonal_value=10.0,
    )
    assert evidence["first_on_second_2d"]["candidate"] is True
    assert evidence["second_on_first_2d"]["candidate"] is False
    assert "multiview_first_on_second" in evidence["candidate_reasons"]

    pillow_masks = {frame: _rect(35, 35, 50, 55) for frame in frames}
    bed_masks = {frame: _rect(20, 10, 90, 90) for frame in frames}
    containment = pair_relation_evidence(
        _object("pillow", pillow_masks),
        _object("bed", bed_masks),
        _camera_info(5),
        scene_diagonal_value=10.0,
    )
    assert containment["first_in_second_2d"]["candidate"] is True

    # Background relations need all five independent pose clusters.
    floor = _object("floor", table_masks, background=True)
    background = pair_relation_evidence(
        lamp,
        floor,
        _camera_info(5),
        scene_diagonal_value=10.0,
    )
    assert background["first_on_second_2d"]["candidate"] is True
    floor_short = copy.deepcopy(floor)
    floor_short["frame_id"] = floor_short["frame_id"][:4]
    floor_short["mask"] = floor_short["mask"][:4]
    background_short = pair_relation_evidence(
        lamp,
        floor_short,
        _camera_info(5),
        scene_diagonal_value=10.0,
    )
    assert background_short["first_on_second_2d"]["candidate"] is False

    other_background = _object("wall", bed_masks, background=True)
    forbidden = pair_relation_evidence(
        floor,
        other_background,
        _camera_info(5),
        scene_diagonal_value=10.0,
    )
    assert forbidden["background_background_forbidden"] is True
    assert forbidden["candidate"] is False


def test_no_shared_views_and_scale_independent_3d_behavior() -> None:
    first = _object("lamp", {"000000": _rect(10, 10, 20, 20)})
    second = _object("nightstand", {"000001": _rect(30, 10, 50, 30)})
    evidence = pair_relation_evidence(
        first,
        second,
        _camera_info(2),
        scene_diagonal_value=10.0,
    )
    assert evidence["candidate"] is False
    assert evidence["first_on_second_2d"]["shared_frame_count"] == 0

    base = np.asarray([[0.00, 0, 0], [0.01, 0, 0], [0.02, 0, 0]])
    near = np.asarray([[0.025, 0, 0], [0.035, 0, 0], [0.045, 0, 0]])
    far = near + np.asarray([10.0, 0, 0])
    assert scale_independent_3d_evidence(base, near)["candidate"] is True
    assert scale_independent_3d_evidence(base, far)["candidate"] is False
