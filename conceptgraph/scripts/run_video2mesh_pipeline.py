"""Run Video2Mesh's mask pipeline and convert it to a ConceptGraphs map.

The orchestration deliberately uses an external process/file boundary: no
Video2Mesh source is vendored, and every run is written below a new run root.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conceptgraph.integrations.video2mesh.adapter import (
    AdapterConfig,
    convert_video2mesh_project,
    validate_video2mesh_project,
)
from conceptgraph.integrations.video2mesh.runner import (
    bootstrap_sam2,
    build_stage_commands,
    load_pipeline_config,
    preflight_environment,
    run_video2mesh_stages,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "video2mesh_pipeline.yaml"
)
DEFAULT_OUTPUT_NAME = "full_pcd_video2mesh_colmap_sam2.pkl.gz"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _print_json(payload: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        file=stream,
    )


def _safe_identifier(value: str, argument: str) -> str:
    if value in {".", ".."} or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{argument} must contain only letters, numbers, '.', '_' or '-', "
            "must start with a letter or number, and cannot be '.' or '..'"
        )
    return value


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _frame_overrides(args: argparse.Namespace) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for argument, key in (
        ("start_frame", "start_frame"),
        ("end_frame", "end_frame_inclusive"),
        ("stride", "every"),
        ("max_frames", "max_frames"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            frames[key] = value

    overrides: dict[str, Any] = {}
    if frames:
        overrides["frames"] = frames
    if getattr(args, "queries_file", None) is not None:
        queries_path = Path(args.queries_file).expanduser().resolve()
        if not queries_path.is_file():
            raise FileNotFoundError(f"Query file does not exist: {queries_path}")
        overrides["detection"] = {"queries": str(queries_path)}
    return overrides


def _run_overrides(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    overrides = _frame_overrides(args)
    profile_name = getattr(args, "profile", None)
    if profile_name is None:
        return overrides
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping) or profile_name not in profiles:
        available = sorted(profiles) if isinstance(profiles, Mapping) else []
        raise ValueError(
            f"Unknown --profile {profile_name!r}; available profiles: {available}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise ValueError(f"Profile {profile_name!r} must be a mapping")
    allowed = {
        "every",
        "max_frames",
        "start_frame",
        "end_frame_inclusive",
        "start_sec",
        "end_sec",
        "duration_sec",
        "source_fps",
    }
    unknown = sorted(set(profile) - allowed)
    if unknown:
        raise ValueError(
            f"Profile {profile_name!r} has unsupported frame keys: {unknown}"
        )
    profile_frames = dict(profile)
    profile_frames.update(overrides.get("frames") or {})
    overrides["frames"] = profile_frames
    return overrides


def _adapter_config(config: Mapping[str, Any]) -> AdapterConfig:
    conversion = dict(config.get("conversion") or {})
    models = dict(config.get("models") or {})
    background_classes = conversion.get(
        "background_classes", ("wall", "floor", "ceiling")
    )
    return AdapterConfig(
        clip_model_path=models.get("clip_model_path"),
        clip_model_sha256=models.get("clip_model_sha256"),
        clip_device=str(conversion.get("clip_device", "auto")),
        clip_local_files_only=bool(
            conversion.get(
                "clip_local_files_only",
                conversion.get("local_files_only", True),
            )
        ),
        clip_batch_size=int(conversion.get("clip_batch_size", 16)),
        embedding_dim=int(conversion.get("embedding_dim", 512)),
        crop_padding_px=int(conversion.get("crop_padding_px", 20)),
        min_foreground_views=int(conversion.get("min_foreground_views", 2)),
        min_background_views=int(conversion.get("min_background_views", 1)),
        min_2d_only_views=int(conversion.get("min_2d_only_views", 3)),
        background_classes=tuple(str(item) for item in background_classes),
    )


def _open_pickle(path: Path) -> Any:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conversion_report_path(map_path: Path) -> Path:
    name = map_path.name
    if not name.endswith(".pkl.gz"):
        raise ValueError(
            f"ConceptGraphs integration maps must end in .pkl.gz: {map_path}"
        )
    return map_path.with_name(name[: -len(".pkl.gz")] + ".conversion.json")


def _validate_map_pickle(path: str | Path) -> dict[str, Any]:
    """Perform a dependency-light structural check of a trusted map pickle."""

    import numpy as np

    map_path = Path(path).expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(f"ConceptGraphs map does not exist: {map_path}")

    payload = _open_pickle(map_path)
    if not isinstance(payload, Mapping):
        raise ValueError("ConceptGraphs map payload must be a mapping")

    required_top_level = {
        "objects",
        "bg_objects",
        "cfg",
        "class_names",
        "class_colors",
    }
    missing_top_level = sorted(required_top_level.difference(payload))
    if missing_top_level:
        raise ValueError(
            "ConceptGraphs map is missing top-level keys: "
            + ", ".join(missing_top_level)
        )

    required_object = {
        "pcd_np",
        "bbox_np",
        "pcd_color_np",
        "clip_ft",
        "text_ft",
        "color_path",
        "mask",
        "xyxy",
        "conf",
        "class_name",
        "class_id",
        "num_detections",
    }
    adapter_config = (
        (payload.get("cfg") or {}).get("adapter_config")
        if isinstance(payload.get("cfg"), Mapping)
        else {}
    )
    embedding_dim = int(
        adapter_config.get("embedding_dim", 512)
        if isinstance(adapter_config, Mapping)
        else 512
    )
    if embedding_dim != 512:
        raise ValueError(
            f"Map embedding_dim must be 512 for current ConceptGraphs, got "
            f"{embedding_dim}"
        )
    object_counts: dict[str, int] = {}
    for group_name in ("objects", "bg_objects"):
        group = payload.get(group_name)
        if group is None:
            group = []
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise ValueError(f"{group_name} must be a sequence or null")
        object_counts[group_name] = len(group)
        for index, obj in enumerate(group):
            if not isinstance(obj, Mapping):
                raise ValueError(f"{group_name}[{index}] must be a mapping")
            missing = sorted(required_object.difference(obj))
            if missing:
                raise ValueError(
                    f"{group_name}[{index}] is missing keys: {', '.join(missing)}"
                )
            prefix = f"{group_name}[{index}]"
            pcd = np.asarray(obj["pcd_np"], dtype=np.float64)
            colors = np.asarray(obj["pcd_color_np"], dtype=np.float64)
            bbox = np.asarray(obj["bbox_np"], dtype=np.float64)
            if (
                pcd.ndim != 2
                or pcd.shape[1:] != (3,)
                or not np.isfinite(pcd).all()
            ):
                raise ValueError(f"{prefix}.pcd_np must be a finite Nx3")
            geometry_type = obj.get("geometry_type")
            if geometry_type is None:
                geometry_type = "colmap_3d" if len(pcd) else "multiview_2d"
            if geometry_type not in {"colmap_3d", "multiview_2d"}:
                raise ValueError(f"{prefix}.geometry_type is invalid")
            if geometry_type == "colmap_3d" and len(pcd) == 0:
                raise ValueError(
                    f"{prefix}.geometry_type=colmap_3d requires non-empty pcd_np"
                )
            if geometry_type == "multiview_2d" and len(pcd) != 0:
                raise ValueError(
                    f"{prefix}.geometry_type=multiview_2d requires empty pcd_np"
                )
            if (
                colors.shape != pcd.shape
                or not np.isfinite(colors).all()
                or (colors < 0).any()
                or (colors > 1).any()
            ):
                raise ValueError(
                    f"{prefix}.pcd_color_np must match pcd_np and lie in [0,1]"
                )
            expected_bbox_shape = (
                (8, 3) if geometry_type == "colmap_3d" else (0, 3)
            )
            if bbox.shape != expected_bbox_shape or not np.isfinite(bbox).all():
                raise ValueError(
                    f"{prefix}.bbox_np must be finite with shape "
                    f"{expected_bbox_shape}"
                )
            for field in ("clip_ft", "text_ft"):
                feature = np.asarray(obj[field], dtype=np.float32)
                if (
                    feature.shape != (embedding_dim,)
                    or not np.isfinite(feature).all()
                    or float(np.linalg.norm(feature)) <= 1e-12
                ):
                    raise ValueError(
                        f"{prefix}.{field} must be a finite non-zero "
                        f"{embedding_dim}-vector"
                    )

            num_detections = obj["num_detections"]
            if (
                not isinstance(num_detections, int)
                or isinstance(num_detections, bool)
                or num_detections <= 0
            ):
                raise ValueError(f"{prefix}.num_detections must be a positive integer")
            detection_fields = (
                "color_path",
                "mask",
                "xyxy",
                "conf",
                "class_name",
                "class_id",
            )
            for field in detection_fields:
                value = obj[field]
                if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
                    raise ValueError(f"{prefix}.{field} must be a sequence")
                if len(value) != num_detections:
                    raise ValueError(
                        f"{prefix}.{field} has length {len(value)}, expected "
                        f"{num_detections}"
                    )
            for detection_index in range(num_detections):
                color_path = Path(str(obj["color_path"][detection_index]))
                if not color_path.is_file():
                    raise ValueError(
                        f"{prefix}.color_path[{detection_index}] is missing: "
                        f"{color_path}"
                    )
                mask = np.asarray(obj["mask"][detection_index])
                if mask.ndim != 2 or mask.size == 0:
                    raise ValueError(
                        f"{prefix}.mask[{detection_index}] must be a non-empty 2D mask"
                    )
                box = np.asarray(obj["xyxy"][detection_index], dtype=np.float64)
                if (
                    box.shape != (4,)
                    or not np.isfinite(box).all()
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    raise ValueError(f"{prefix}.xyxy[{detection_index}] is invalid")
                confidence = obj["conf"][detection_index]
                if (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not math.isfinite(float(confidence))
                ):
                    raise ValueError(f"{prefix}.conf[{detection_index}] must be finite")
                if not str(obj["class_name"][detection_index]).strip():
                    raise ValueError(f"{prefix}.class_name[{detection_index}] is empty")

    provenance = payload.get("provenance")
    caller = provenance.get("caller") if isinstance(provenance, Mapping) else None
    runner_fingerprint = (
        caller.get("runner_fingerprint") if isinstance(caller, Mapping) else None
    )
    source_validation = (
        provenance.get("source_validation") if isinstance(provenance, Mapping) else None
    )
    source_input_sha256 = (
        source_validation.get("input_sha256")
        if isinstance(source_validation, Mapping)
        else None
    )
    conversion_report = (
        provenance.get("conversion_report") if isinstance(provenance, Mapping) else None
    )
    return {
        "ok": True,
        "map": str(map_path),
        "size_bytes": map_path.stat().st_size,
        "pickle_sha256": _sha256_file(map_path),
        "objects": object_counts["objects"],
        "background_objects": object_counts["bg_objects"],
        "class_count": len(payload.get("class_names") or []),
        "has_provenance": isinstance(provenance, Mapping),
        "runner_fingerprint": runner_fingerprint,
        "source_input_sha256": source_input_sha256,
        "conversion_report": conversion_report,
    }


def _validate_conversion_pair(
    map_path: str | Path,
    map_validation: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_map = Path(map_path).expanduser().resolve()
    declared_report = map_validation.get("conversion_report")
    report_path = (
        Path(str(declared_report)).expanduser().resolve()
        if declared_report
        else _conversion_report_path(resolved_map)
    )
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Completed conversion report is missing: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid conversion report {report_path}: {exc}") from exc
    if not isinstance(report, Mapping) or report.get("status") != "converted":
        raise ValueError(
            f"Conversion report is not a completed conversion: {report_path}"
        )
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Conversion report has no outputs mapping: {report_path}")
    reported_map = Path(str(outputs.get("pickle", ""))).expanduser().resolve()
    try:
        same_map = os.path.samefile(reported_map, resolved_map)
    except OSError:
        same_map = False
    if not same_map:
        raise ValueError(f"Conversion report points to a different map: {reported_map}")
    reported_report = (
        Path(str(outputs.get("conversion_json", ""))).expanduser().resolve()
    )
    try:
        same_report = os.path.samefile(reported_report, report_path)
    except OSError:
        same_report = False
    if not same_report:
        raise ValueError(
            "Conversion report outputs.conversion_json does not identify itself: "
            f"{reported_report}"
        )
    actual_sha256 = str(map_validation.get("pickle_sha256") or "")
    if outputs.get("pickle_sha256") != actual_sha256:
        raise ValueError(
            f"Map SHA-256 does not match conversion report: {resolved_map}"
        )
    report_provenance = report.get("provenance")
    report_caller = (
        report_provenance.get("caller")
        if isinstance(report_provenance, Mapping)
        else None
    )
    report_fingerprint = (
        report_caller.get("runner_fingerprint")
        if isinstance(report_caller, Mapping)
        else None
    )
    if report_fingerprint != map_validation.get("runner_fingerprint"):
        raise ValueError("Map and conversion report have different runner fingerprints")
    report_validation = report.get("validation")
    report_input_sha256 = (
        report_validation.get("input_sha256")
        if isinstance(report_validation, Mapping)
        else None
    )
    if report_input_sha256 != map_validation.get("source_input_sha256"):
        raise ValueError(
            "Map and conversion report have different source artifact hashes"
        )
    return {
        "ok": True,
        "conversion_json": str(report_path),
        "pickle_sha256": actual_sha256,
        "runner_fingerprint": report_fingerprint,
    }


def _cmd_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    return bootstrap_sam2(config, dry_run=args.dry_run)


def _cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    report = preflight_environment(
        config,
        video=args.video,
        project_root=args.project_root,
        raise_on_error=False,
    )
    return report


def _cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    scene_id = _safe_identifier(args.scene_id, "--scene-id")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    run_id = _safe_identifier(args.run_id or _new_run_id(), "--run-id")

    output_base = Path(args.output_base).expanduser().resolve()
    run_root = output_base / scene_id / run_id
    project_root = run_root / "v2m_project"
    overrides = _run_overrides(config, args)
    overrides["paths"] = {"output_base": str(output_base)}
    if not args.dry_run:
        if args.resume and not project_root.is_dir():
            raise FileNotFoundError(
                f"Cannot resume because the project does not exist: {project_root}"
            )
        if not args.resume and run_root.exists():
            raise FileExistsError(
                f"Run directory already exists; choose a new --run-id: {run_root}"
            )

    # Build once up front so dry-run reports always expose the exact commands,
    # even if a future runner changes the shape of its aggregate report.
    planned_commands = build_stage_commands(
        config,
        video=args.video,
        scene_id=scene_id,
        project_root=project_root,
        overrides=overrides,
    )
    runner_report = run_video2mesh_stages(
        config,
        video=args.video,
        scene_id=scene_id,
        project_root=project_root,
        overrides=overrides,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    report: dict[str, Any] = {
        "ok": True,
        "scene_id": scene_id,
        "run_id": run_id,
        "profile": args.profile,
        "run_root": str(run_root),
        "project_root": str(project_root),
        "runner": runner_report,
    }
    if args.dry_run:
        report["ready_to_run"] = bool((runner_report.get("preflight") or {}).get("ok"))
        report["planned_stages"] = [command.to_dict() for command in planned_commands]
        report["conversion"] = {
            "status": "skipped",
            "reason": "dry-run does not write a ConceptGraphs map",
        }
        return report

    map_output = run_root / "conceptgraphs" / DEFAULT_OUTPUT_NAME
    provenance = {
        "integration": "conceptgraphs_video2mesh",
        "scene_id": scene_id,
        "run_id": run_id,
        "profile": args.profile,
        "input_video": str(Path(args.video).expanduser().resolve()),
        "versions": dict(config.get("versions") or {}),
        "runner_fingerprint": runner_report.get("fingerprint"),
        "runner_manifest": runner_report.get("manifest_path"),
        "model_sha256": runner_report.get("model_sha256"),
        "coordinate_units": "colmap_arbitrary",
    }
    map_validation: dict[str, Any] | None = None
    project_validation = validate_video2mesh_project(project_root)
    conversion_pair: dict[str, Any] | None = None
    if args.resume and map_output.is_file():
        map_validation = _validate_map_pickle(map_output)
        if map_validation.get("runner_fingerprint") != runner_report.get("fingerprint"):
            raise ValueError(
                "Existing resumed map has a different or missing runner "
                f"fingerprint: {map_output}"
            )
        if map_validation.get("source_input_sha256") != project_validation.get(
            "input_sha256"
        ):
            raise ValueError(
                "Existing resumed map was converted from different or modified "
                f"Video2Mesh artifacts: {map_output}"
            )
        conversion_pair = _validate_conversion_pair(map_output, map_validation)
        conversion_report = {
            "status": "reused",
            "reason": (
                "validated existing map, conversion report, runner fingerprint, "
                "and current Video2Mesh artifact hashes"
            ),
            "output_pickle": str(map_output),
        }
    else:
        conversion_report = convert_video2mesh_project(
            project_root,
            map_output,
            config=_adapter_config(config),
            provenance=provenance,
        )
    if map_validation is None:
        map_validation = _validate_map_pickle(map_output)
    if map_validation.get("source_input_sha256") != project_validation.get(
        "input_sha256"
    ):
        raise ValueError(
            "Converted map source hashes do not match the current Video2Mesh "
            f"project: {map_output}"
        )
    if conversion_pair is None:
        conversion_pair = _validate_conversion_pair(map_output, map_validation)
    report["map_output"] = str(map_output)
    report["conversion"] = conversion_report
    report["validation"] = {
        "project": project_validation,
        "map": map_validation,
        "conversion_pair": conversion_pair,
    }
    accepted_foreground = int(
        (conversion_report.get("counts") or {}).get(
            "accepted_foreground_objects",
            report["validation"]["map"]["objects"],
        )
    )
    report["acceptance"] = {
        "accepted_foreground_objects": accepted_foreground,
        "minimum_foreground_objects": 1,
    }
    if accepted_foreground < 1:
        report["ok"] = False
        report["acceptance"]["error"] = (
            "Conversion produced no downstream-usable foreground object"
        )
    return report


def _cmd_convert(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    output = Path(args.output).expanduser().resolve()
    _conversion_report_path(output)
    if output.exists():
        raise FileExistsError(
            f"Conversion output already exists; choose a new --output: {output}"
        )
    project_report = validate_video2mesh_project(args.project_root)
    conversion_report = convert_video2mesh_project(
        args.project_root,
        output,
        config=_adapter_config(config),
        provenance={
            "integration": "conceptgraphs_video2mesh",
            "versions": dict(config.get("versions") or {}),
            "coordinate_units": "colmap_arbitrary",
        },
    )
    map_validation = _validate_map_pickle(output)
    return {
        "ok": True,
        "project": project_report,
        "conversion": conversion_report,
        "map": map_validation,
        "conversion_pair": _validate_conversion_pair(output, map_validation),
    }


def _cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    # Load the config here as well so a typo/missing YAML fails consistently
    # across all entry points, even though validation itself is read-only.
    load_pipeline_config(args.config)
    if args.project_root is None and args.map_path is None:
        raise ValueError("validate requires --project-root, --map, or both")
    report: dict[str, Any] = {"ok": True}
    if args.project_root is not None:
        report["project"] = validate_video2mesh_project(args.project_root)
    if args.map_path is not None:
        map_validation = _validate_map_pickle(args.map_path)
        report["map"] = map_validation
        report["conversion_pair"] = _validate_conversion_pair(
            args.map_path, map_validation
        )
        if args.project_root is not None and map_validation.get(
            "source_input_sha256"
        ) != report["project"].get("input_sha256"):
            raise ValueError(
                "Map source hashes do not match the supplied Video2Mesh project"
            )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Pipeline YAML (default: {DEFAULT_CONFIG})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Video2Mesh multi-view 2D/3D masks and a compatible "
            "ConceptGraphs MapObjectList without modifying existing data."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Prepare the isolated, pinned SAM2 environment."
    )
    _add_config_argument(bootstrap)
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.set_defaults(handler=_cmd_bootstrap)

    preflight = subparsers.add_parser(
        "preflight", help="Check repositories, tools, models and optional input."
    )
    _add_config_argument(preflight)
    preflight.add_argument("--video", type=Path)
    preflight.add_argument("--project-root", type=Path)
    preflight.set_defaults(handler=_cmd_preflight)

    run = subparsers.add_parser(
        "run", help="Run all Video2Mesh mask stages and convert the result."
    )
    _add_config_argument(run)
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--scene-id", required=True)
    run.add_argument("--output-base", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument(
        "--profile",
        help="Named frame-selection profile from the pipeline config.",
    )
    run.add_argument("--start-frame", type=_nonnegative_int)
    run.add_argument("--end-frame", type=_nonnegative_int)
    run.add_argument("--stride", type=_positive_int)
    run.add_argument(
        "--max-frames",
        type=_nonnegative_int,
        help="Maximum selected frames; 0 disables the cap.",
    )
    run.add_argument("--queries-file", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=_cmd_run)

    convert = subparsers.add_parser(
        "convert", help="Convert an existing complete Video2Mesh project."
    )
    _add_config_argument(convert)
    convert.add_argument("--project-root", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.set_defaults(handler=_cmd_convert)

    validate = subparsers.add_parser(
        "validate", help="Validate a Video2Mesh project and/or converted map."
    )
    _add_config_argument(validate)
    validate.add_argument("--project-root", type=Path)
    validate.add_argument("--map", dest="map_path", type=Path)
    validate.set_defaults(handler=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = args.handler(args)
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _print_json(
            {
                "ok": False,
                "command": args.command,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(report)
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
