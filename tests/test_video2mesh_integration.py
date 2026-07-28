from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

import conceptgraph.integrations.video2mesh.runner as runner_module
from conceptgraph.integrations.video2mesh.adapter import (
    AdapterConfig,
    convert_video2mesh_project,
    validate_video2mesh_project,
)
from conceptgraph.integrations.video2mesh.colmap_compat import (
    EXTRACTION_OPTION_ENV,
    MATCHING_OPTION_ENV,
    MODERN_EXTRACTION_OPTION,
    MODERN_MATCHING_OPTION,
    REAL_BINARY_ENV,
    REAL_SHA256_ENV,
    WRAPPER_SHA256_ENV,
)
from conceptgraph.integrations.video2mesh.runner import (
    SAM2_COMMIT,
    StageCommand,
    UnsafeOutputPathError,
    Video2MeshRunnerError,
    _check_import,
    _colmap_compat_wrapper_path,
    _command_digest,
    _probe_colmap_cli_compatibility,
    _stage_artifact_hashes,
    _stage_marker_is_valid,
    bootstrap_sam2,
    build_stage_commands,
    compute_frame_window,
    load_pipeline_config,
)
from conceptgraph.scripts.run_video2mesh_pipeline import (
    _run_overrides,
    _validate_conversion_pair,
    _validate_map_pickle,
)


class _FakeEmbedder:
    metadata = {"implementation": "test-fake", "dimension": 512}

    @staticmethod
    def _rows(count: int, offset: int) -> np.ndarray:
        rows = np.zeros((count, 512), dtype=np.float32)
        for index in range(count):
            rows[index, (offset + index) % 512] = 1.0
            rows[index, (offset + index + 7) % 512] = 0.5
        return rows

    def embed_images(self, images):
        return self._rows(len(images), 3)

    def embed_texts(self, texts):
        return self._rows(len(texts), 101)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_fake_modern_colmap(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

arguments = sys.argv[1:]
if arguments == ["version"]:
    print("COLMAP 4.1.0 (fake without CUDA)")
    raise SystemExit(0)
if len(arguments) == 2 and arguments[1] in {{"-h", "--help"}}:
    if arguments[0] == "feature_extractor":
        print("  --FeatureExtraction.use_gpu arg (=1)")
    elif arguments[0] in {{"exhaustive_matcher", "sequential_matcher"}}:
        print("  --FeatureMatching.use_gpu arg (=1)")
    else:
        raise SystemExit(2)
    raise SystemExit(0)
print(json.dumps(arguments))
print("fake-colmap-stderr", file=sys.stderr)
raise SystemExit(int(os.environ.get("FAKE_COLMAP_EXIT", "0")))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_ascii_ply(path: Path) -> None:
    rows = [
        (0.0, 0.0, 1.0, 255, 0, 0),
        (1.0, 0.0, 1.0, 255, 64, 0),
        (0.0, 1.0, 1.0, 255, 128, 0),
        (0.0, 0.0, 2.0, 255, 192, 0),
        (3.0, 0.0, 1.0, 0, 0, 255),
        (3.0, 1.0, 1.0, 0, 64, 255),
        (3.0, 0.0, 2.0, 0, 128, 255),
        (3.0, 1.0, 2.0, 0, 192, 255),
    ]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = [" ".join(str(value) for value in row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + body) + "\n", encoding="ascii")


def _make_v2m_project(tmp_path: Path) -> Path:
    root = tmp_path / "v2m_project"
    frames_dir = root / "scene" / "frames"
    frames_dir.mkdir(parents=True)
    frame_records = []
    for ordinal, source_index in enumerate((2709, 2733)):
        frame_id = f"{ordinal:06d}"
        image_path = frames_dir / f"{frame_id}.png"
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[..., 0] = 30 + ordinal * 20
        image[..., 1] = np.arange(128, dtype=np.uint8)[None, :]
        Image.fromarray(image).save(image_path)
        frame_records.append(
            {
                "frame_id": frame_id,
                "source_frame_index": source_index,
                "source_time_sec": source_index / 60.0,
                "path": str(image_path),
            }
        )

    _write_json(
        root / "scene" / "frames_manifest.json",
        {
            "schema_version": 1,
            "source_video": str(tmp_path / "source.mp4"),
            "source_width": 128,
            "source_height": 128,
            "source_fps": 60.0,
            "written_frame_count": 2,
            "frames": frame_records,
        },
    )
    _write_json(
        root / "scene" / "cameras" / "camera_info.json",
        {
            "intrinsic": {
                "w": 128,
                "h": 128,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 64.0,
                "cy": 64.0,
                "model": "PINHOLE",
                "params": [100.0, 100.0, 64.0, 64.0],
            },
            "extrinsic_type": "world_to_camera",
            "extrinsic": {
                "000000": np.eye(4).tolist(),
                "000001": [
                    [1.0, 0.0, 0.0, -0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )
    point_cloud = root / "scene" / "reconstruction" / "point_cloud.ply"
    _write_ascii_ply(point_cloud)

    labels = {
        "object_chair_0": {
            "object_id": "object_chair_0",
            "name": "chair",
            "category": "chair",
            "description": "a chair",
        },
        "object_wall_0": {
            "object_id": "object_wall_0",
            "name": "wall",
            "category": "wall",
            "description": "a wall",
        },
    }
    _write_json(root / "masks" / "object_labels.json", labels)

    object_indices = {
        "object_chair_0": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "object_wall_0": np.asarray([4, 5, 6, 7], dtype=np.int64),
    }
    objects = {}
    for object_id, indices in object_indices.items():
        mask_3d_dir = root / "masks" / "3d" / object_id
        mask_3d_dir.mkdir(parents=True)
        indices_path = mask_3d_dir / "point_indices.npy"
        np.save(indices_path, indices)

        frame_scores = {}
        mask_dir = root / "masks" / "2d" / object_id
        mask_dir.mkdir(parents=True)
        for frame_id in ("000000", "000001"):
            mask = np.zeros((128, 128), dtype=np.uint8)
            if object_id == "object_chair_0":
                mask[20:90, 20:90] = 255
            else:
                mask[10:118, 8:120] = 255
            mask_path = mask_dir / f"{frame_id}.png"
            Image.fromarray(mask).save(mask_path)
            frame_scores[frame_id] = {
                "mask": str(mask_path),
                "mask_area": int((mask > 0).sum()),
                "projected_points": 8,
                "visible_points": 8,
                "hit_points": len(indices),
            }

        label = labels[object_id]
        objects[object_id] = {
            "object_id": object_id,
            "name": label["name"],
            "category": label["category"],
            "description": label["description"],
            "point_count": len(indices),
            "mask_3d": {
                "point_indices_npy": str(indices_path),
                "fusion_mode": "probability",
                "min_votes": 1,
            },
            "frame_scores": frame_scores,
        }

    _write_json(
        root / "masks" / "3d" / "object_masks.json",
        {
            "schema_version": 1,
            "point_cloud": str(point_cloud),
            "camera_info": str(root / "scene" / "cameras" / "camera_info.json"),
            "mask_root": str(root / "masks" / "2d"),
            "num_points": 8,
            "num_masks": 4,
            "objects": objects,
            "skipped": [],
            "fusion": {
                "mode": "probability",
                "min_probability": 0.5,
                "min_votes": 1,
                "occlusion_filter": True,
                "depth_tolerance": 0.05,
                "relative_depth_tolerance": 0.03,
                "exclusive_objects": True,
            },
        },
    )
    return root


def test_adapter_validates_converts_and_round_trips(tmp_path: Path) -> None:
    project = _make_v2m_project(tmp_path)
    validation = validate_video2mesh_project(project)
    assert validation["valid"] is True
    assert validation["frames"]["source_frame_indices"] == [2709, 2733]
    assert validation["objects"]["mutually_exclusive_3d_masks"] is True

    output = tmp_path / "converted" / "full_pcd_video2mesh.pkl.gz"
    report = convert_video2mesh_project(
        project,
        output,
        config=AdapterConfig(clip_model_path="/unused/in/injected-test"),
        embedder=_FakeEmbedder(),
        provenance={"test": True},
    )
    assert report["counts"] == {
        "source_objects": 2,
        "accepted_foreground_objects": 1,
        "accepted_background_objects": 1,
        "rejected_objects": 0,
    }
    map_validation = _validate_map_pickle(output)
    pair_validation = _validate_conversion_pair(output, map_validation)
    assert pair_validation["pickle_sha256"] == map_validation["pickle_sha256"]

    with gzip.open(output, "rb") as handle:
        payload = pickle.load(handle)
    assert payload["class_names"] == ["chair", "wall"]
    assert len(payload["objects"]) == 1
    assert len(payload["bg_objects"]) == 1
    chair = payload["objects"][0]
    wall = payload["bg_objects"][0]
    assert chair["v2m_object_id"] == "object_chair_0"
    assert wall["is_background"] is True
    assert chair["source_frame_index"] == [2709, 2733]
    assert all(mask.dtype == np.bool_ for mask in chair["mask"])
    np.testing.assert_array_equal(chair["point_indices"], [0, 1, 2, 3])
    np.testing.assert_allclose(np.linalg.norm(chair["clip_ft"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(chair["text_ft"]), 1.0, atol=1e-6)

    from conceptgraph.slam.slam_classes import MapObjectList

    loaded = MapObjectList()
    loaded.load_serializable(payload["objects"])
    assert len(loaded) == 1
    np.testing.assert_allclose(np.asarray(loaded[0]["pcd"].points), chair["pcd_np"])
    assert len(loaded[0]["color_path"]) == 2


def test_adapter_rejects_overlapping_3d_indices(tmp_path: Path) -> None:
    project = _make_v2m_project(tmp_path)
    overlap_path = project / "masks" / "3d" / "object_wall_0" / "point_indices.npy"
    np.save(overlap_path, np.asarray([3, 5, 6, 7], dtype=np.int64))
    with pytest.raises(ValueError, match="not mutually exclusive"):
        validate_video2mesh_project(project)


def test_adapter_rejects_noncanonical_declared_mask_root(tmp_path: Path) -> None:
    project = _make_v2m_project(tmp_path)
    other_mask_root = project / "masks" / "other"
    other_mask_root.mkdir()
    summary_path = project / "masks" / "3d" / "object_masks.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mask_root"] = str(other_mask_root)
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="canonical directory"):
        validate_video2mesh_project(project)


def test_adapter_reports_an_object_with_no_2d_masks(tmp_path: Path) -> None:
    project = _make_v2m_project(tmp_path)
    wall_mask_dir = project / "masks" / "2d" / "object_wall_0"
    for mask_path in wall_mask_dir.glob("*.png"):
        mask_path.unlink()

    summary_path = project / "masks" / "3d" / "object_masks.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_masks"] = 2
    summary["objects"]["object_wall_0"]["frame_scores"] = {}
    _write_json(summary_path, summary)

    validation = validate_video2mesh_project(project)
    assert validation["objects"]["mask_counts"]["object_wall_0"] == 0

    output = tmp_path / "converted" / "map.pkl.gz"
    report = convert_video2mesh_project(
        project,
        output,
        config=AdapterConfig(clip_model_path="/unused/in/injected-test"),
        embedder=_FakeEmbedder(),
    )
    assert report["counts"]["accepted_foreground_objects"] == 1
    assert report["counts"]["accepted_background_objects"] == 0
    assert report["rejections"][0]["object_id"] == "object_wall_0"
    assert "no_nonempty_2d_masks" in report["rejections"][0]["reasons"]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_video2mesh_project(
            project,
            output,
            config=AdapterConfig(clip_model_path="/unused/in/injected-test"),
            embedder=_FakeEmbedder(),
        )


def test_adapter_converts_stable_zero_point_object_as_multiview_2d(
    tmp_path: Path,
) -> None:
    project = _make_v2m_project(tmp_path)
    frames_manifest_path = project / "scene" / "frames_manifest.json"
    frames_manifest = json.loads(frames_manifest_path.read_text(encoding="utf-8"))
    image_path = project / "scene" / "frames" / "000002.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    frames_manifest["frames"].append(
        {
            "frame_id": "000002",
            "source_frame_index": 2757,
            "source_time_sec": 2757 / 60.0,
            "path": str(image_path),
        }
    )
    frames_manifest["written_frame_count"] = 3
    _write_json(frames_manifest_path, frames_manifest)

    camera_path = project / "scene" / "cameras" / "camera_info.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    camera["extrinsic"]["000002"] = np.eye(4).tolist()
    _write_json(camera_path, camera)

    object_masks_path = project / "masks" / "3d" / "object_masks.json"
    object_masks = json.loads(object_masks_path.read_text(encoding="utf-8"))
    for object_id in ("object_chair_0", "object_wall_0"):
        mask = np.zeros((128, 128), dtype=np.uint8)
        if object_id == "object_chair_0":
            mask[20:90, 20:90] = 255
        else:
            mask[10:118, 8:120] = 255
        mask_path = project / "masks" / "2d" / object_id / "000002.png"
        Image.fromarray(mask).save(mask_path)
        object_masks["objects"][object_id]["frame_scores"]["000002"] = {
            "mask": str(mask_path),
            "mask_area": int((mask > 0).sum()),
            "projected_points": 8,
            "visible_points": 8,
            "hit_points": 0 if object_id == "object_chair_0" else 4,
        }
    empty_indices = np.empty(0, dtype=np.int64)
    np.save(
        project / "masks" / "3d" / "object_chair_0" / "point_indices.npy",
        empty_indices,
    )
    object_masks["objects"]["object_chair_0"]["point_count"] = 0
    object_masks["num_masks"] = 6
    _write_json(object_masks_path, object_masks)

    output = tmp_path / "converted" / "map.pkl.gz"
    convert_video2mesh_project(
        project,
        output,
        config=AdapterConfig(clip_model_path="/unused/in/injected-test"),
        embedder=_FakeEmbedder(),
    )
    with gzip.open(output, "rb") as handle:
        payload = pickle.load(handle)
    chair = payload["objects"][0]
    assert chair["geometry_type"] == "multiview_2d"
    assert chair["point_count"] == 0
    assert chair["pcd_np"].shape == (0, 3)
    assert chair["pcd_color_np"].shape == (0, 3)
    assert chair["bbox_np"].shape == (0, 3)
    assert chair["point_indices"].shape == (0,)

    from conceptgraph.slam.slam_classes import MapObjectList

    loaded = MapObjectList()
    loaded.load_serializable(payload["objects"])
    assert loaded[0]["bbox"] is None
    round_trip = loaded.to_serializable()[0]
    assert round_trip["geometry_type"] == "multiview_2d"
    assert round_trip["bbox_np"].shape == (0, 3)


def test_map_object_geometry_marker_mismatch_fails() -> None:
    from conceptgraph.slam.slam_classes import MapObjectList

    record = {
        "geometry_type": "multiview_2d",
        "point_count": 1,
        "point_indices": np.asarray([0], dtype=np.int64),
        "pcd_np": np.asarray([[0.0, 0.0, 0.0]]),
        "pcd_color_np": np.asarray([[1.0, 0.0, 0.0]]),
        "bbox_np": np.zeros((8, 3)),
        "clip_ft": np.ones(512, dtype=np.float32),
        "text_ft": np.ones(512, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="multiview_2d"):
        MapObjectList().load_serializable([record])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            [
                "feature_extractor",
                "--database_path",
                "/path with spaces/database.db",
                "--SiftExtraction.use_gpu",
                "0",
                "--SiftExtraction.max_num_features",
                "4096",
            ],
            [
                "feature_extractor",
                "--database_path",
                "/path with spaces/database.db",
                "--FeatureExtraction.use_gpu",
                "0",
                "--SiftExtraction.max_num_features",
                "4096",
            ],
        ),
        (
            [
                "exhaustive_matcher",
                "--SiftMatching.use_gpu",
                "0",
                "--SiftMatching.max_ratio",
                "0.8",
            ],
            [
                "exhaustive_matcher",
                "--FeatureMatching.use_gpu",
                "0",
                "--SiftMatching.max_ratio",
                "0.8",
            ],
        ),
        (
            ["sequential_matcher", "--SiftMatching.use_gpu", "1"],
            ["sequential_matcher", "--FeatureMatching.use_gpu", "1"],
        ),
        (
            ["mapper", "--Mapper.ba_refine_focal_length", "1"],
            ["mapper", "--Mapper.ba_refine_focal_length", "1"],
        ),
        (
            ["model_converter", "--output_type", "TXT"],
            ["model_converter", "--output_type", "TXT"],
        ),
    ],
)
def test_colmap_compat_wrapper_rewrites_exact_options_and_delegates(
    tmp_path: Path,
    arguments: list[str],
    expected: list[str],
) -> None:
    fake_colmap = tmp_path / "fake colmap"
    _write_fake_modern_colmap(fake_colmap)
    wrapper = _colmap_compat_wrapper_path()
    environment = os.environ.copy()
    environment.update(
        {
            REAL_BINARY_ENV: str(fake_colmap.resolve()),
            REAL_SHA256_ENV: hashlib.sha256(fake_colmap.read_bytes()).hexdigest(),
            WRAPPER_SHA256_ENV: hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            EXTRACTION_OPTION_ENV: MODERN_EXTRACTION_OPTION,
            MATCHING_OPTION_ENV: MODERN_MATCHING_OPTION,
            "FAKE_COLMAP_EXIT": "17",
        }
    )

    result = subprocess.run(
        [str(wrapper), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 17
    assert json.loads(result.stdout) == expected
    assert "fake-colmap-stderr" in result.stderr
    assert wrapper.stat().st_mode & 0o111


def test_runner_builds_exact_safe_stage_sequence(tmp_path: Path) -> None:
    source_config = load_pipeline_config(
        Path("conceptgraph/configs/video2mesh_pipeline.yaml")
    )
    output_base = tmp_path / "runs"
    source_config["paths"]["output_base"] = str(output_base)
    fake_colmap = tmp_path / "fake-colmap"
    _write_fake_modern_colmap(fake_colmap)
    source_config["tools"]["colmap_binary"] = str(fake_colmap)
    video = tmp_path / "video.mp4"
    args = type(
        "Args",
        (),
        {
            "profile": "bedroom_validation",
            "start_frame": None,
            "end_frame": None,
            "stride": None,
            "max_frames": None,
            "queries_file": None,
        },
    )()
    overrides = _run_overrides(source_config, args)
    overrides["frames"]["source_fps"] = 60.0

    frame_window = compute_frame_window(source_config, overrides)
    assert frame_window["expected_source_indices"] == [
        2709 + 24 * index for index in range(31)
    ]

    commands = build_stage_commands(
        source_config,
        video,
        "bedroom",
        output_base / "bedroom" / "run-1" / "v2m_project",
        overrides,
    )
    assert [command.name for command in commands] == [
        "init",
        "extract_frames",
        "run_colmap",
        "reconstruction_readiness",
        "discover_object_prompts",
        "normalize_object_prompts",
        "track_masks",
        "normalize_mask_tracks",
        "mask_track_quality_report",
        "fuse_masks",
        "finalize_fusion_manifest",
    ]
    joined = "\n".join(" ".join(command.argv) for command in commands)
    assert "run_video2mesh_quick.sh" not in joined
    assert "--overwrite" not in joined
    assert "--clear-" not in joined
    assert "--no-dense-reconstruction" in joined
    assert "--no-use-gpu" in joined
    assert "--depth-tolerance 0.05" in joined
    assert "--relative-depth-tolerance 0.03" in joined
    assert "--min-votes 1" in joined
    assert "--keep-raw-detections" in joined
    assert "--no-merge-bed-parts" in joined
    assert "object_prompts_normalized.json" in joined
    assert "masks/2d_raw" in joined
    assert "masks/2d_fusion" in joined
    run_colmap = next(command for command in commands if command.name == "run_colmap")
    wrapper_index = run_colmap.argv.index("--colmap-binary") + 1
    assert Path(run_colmap.argv[wrapper_index]) == _colmap_compat_wrapper_path()
    assert run_colmap.env[REAL_BINARY_ENV] == str(fake_colmap.resolve())
    assert run_colmap.env[EXTRACTION_OPTION_ENV] == MODERN_EXTRACTION_OPTION
    assert run_colmap.env[MATCHING_OPTION_ENV] == MODERN_MATCHING_OPTION
    compat_ok, _, compat_profile = _probe_colmap_cli_compatibility(str(fake_colmap))
    assert compat_ok
    assert compat_profile is not None
    assert compat_profile["version"] == "COLMAP 4.1.0 (fake without CUDA)"

    with pytest.raises(UnsafeOutputPathError):
        build_stage_commands(
            source_config,
            video,
            "unsafe",
            tmp_path / "outside" / "v2m_project",
        )


def test_sam2_bootstrap_dry_run_is_write_free(tmp_path: Path) -> None:
    config = load_pipeline_config(Path("conceptgraph/configs/video2mesh_pipeline.yaml"))
    dependency_root = tmp_path / "dependencies"
    config["paths"]["dependency_root"] = str(dependency_root)
    config["paths"]["sam2_source"] = str(dependency_root / "sam2")
    config["paths"]["sam2_checkpoint"] = str(
        dependency_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    )
    config["bootstrap"]["prefix"] = str(dependency_root / "sam2-env")
    config["bootstrap"]["source_dir"] = str(dependency_root / "sam2")
    config["bootstrap"]["checkpoint"] = str(
        dependency_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    )
    config["bootstrap"]["conda_binary"] = "/bin/true"
    config["tools"]["git"] = "/bin/true"

    report = bootstrap_sam2(config, dry_run=True)

    assert report["status"] == "dry_run"
    assert not dependency_root.exists()
    commands = {item["name"]: item for item in report["commands"]}
    assert commands["install_sam2"]["env"] == {"SAM2_BUILD_CUDA": "0"}
    assert (
        "opencv-python-headless==4.11.0.86" in commands["install_video_runtime"]["argv"]
    )


def test_sam2_bootstrap_rejects_dirty_existing_source_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config(Path("conceptgraph/configs/video2mesh_pipeline.yaml"))
    dependency_root = tmp_path / "dependencies"
    source = dependency_root / "sam2"
    source.mkdir(parents=True)
    config["paths"]["dependency_root"] = str(dependency_root)
    config["paths"]["sam2_source"] = str(source)
    config["paths"]["sam2_checkpoint"] = str(
        dependency_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    )
    config["bootstrap"]["prefix"] = str(dependency_root / "sam2-env")
    config["bootstrap"]["source_dir"] = str(source)
    config["bootstrap"]["checkpoint"] = str(
        dependency_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    )
    config["bootstrap"]["conda_binary"] = "/bin/true"
    config["tools"]["git"] = "/bin/true"
    monkeypatch.setattr(
        runner_module,
        "_git_head",
        lambda *_args: (SAM2_COMMIT, None),
    )
    monkeypatch.setattr(
        runner_module,
        "_git_tracked_worktree_clean",
        lambda *_args: (False, "tracked worktree has modifications: M sam2.py"),
    )

    with pytest.raises(Video2MeshRunnerError, match="modified SAM2"):
        bootstrap_sam2(config, dry_run=True)


def test_conversion_pair_rejects_a_tampered_report(tmp_path: Path) -> None:
    project = _make_v2m_project(tmp_path)
    output = tmp_path / "converted" / "map.pkl.gz"
    convert_video2mesh_project(
        project,
        output,
        config=AdapterConfig(clip_model_path="/unused/in/injected-test"),
        embedder=_FakeEmbedder(),
        provenance={"runner_fingerprint": "runner-test"},
    )
    map_validation = _validate_map_pickle(output)
    report_path = output.with_name("map.conversion.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["pickle_sha256"] = "0" * 64
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="SHA-256"):
        _validate_conversion_pair(output, map_validation)


def test_import_probe_does_not_create_pycache(tmp_path: Path) -> None:
    module_root = tmp_path / "modules"
    module_root.mkdir()
    module_path = module_root / "probe_module.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    ok, detail = _check_import(
        str(Path("/proc/self/exe").resolve()),
        "probe_module",
        {"PYTHONPATH": str(module_root)},
    )

    assert ok, detail
    assert not (module_root / "__pycache__").exists()


def test_extract_marker_allows_colmap_to_prune_frames(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    frames_dir = project_root / "scene" / "frames"
    frames_dir.mkdir(parents=True)
    kept_frame = frames_dir / "000000.png"
    pruned_frame = frames_dir / "000001.png"
    kept_frame.write_bytes(b"frame-kept")
    pruned_frame.write_bytes(b"frame-pruned")
    manifest = project_root / "scene" / "frames_manifest.json"
    _write_json(
        manifest,
        {"frames": [{"frame_id": "000000"}, {"frame_id": "000001"}]},
    )
    command = StageCommand(
        name="extract_frames",
        argv=("python", "-B", "-m", "video2mesh.cli", "extract-frames"),
        python="python",
        cwd=str(tmp_path),
        expected_outputs=(str(manifest),),
    )
    marker_path = tmp_path / "marker.json"
    _write_json(
        marker_path,
        {
            "status": "completed",
            "fingerprint": "run-fingerprint",
            "command_sha256": _command_digest(command),
            "artifact_sha256": _stage_artifact_hashes(project_root, command.name),
        },
    )

    valid, _ = _stage_marker_is_valid(
        project_root,
        marker_path,
        command,
        "run-fingerprint",
    )
    assert valid

    pruned_frame.unlink()
    valid, detail = _stage_marker_is_valid(
        project_root,
        marker_path,
        command,
        "run-fingerprint",
    )
    assert valid, detail

    _write_json(manifest, {"frames": [{"frame_id": "000000"}]})
    valid, detail = _stage_marker_is_valid(
        project_root,
        marker_path,
        command,
        "run-fingerprint",
    )
    assert not valid
    assert "SHA-256 differs" in detail
