"""Safe orchestration for the external Video2Mesh mask pipeline.

This module deliberately treats Video2Mesh as an external program.  It does
not import or copy Video2Mesh implementation details, and it never invokes the
project's destructive quick-run shell script.  Commands are executed without a
shell and all writable paths are constrained to a caller-provided output base.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

from conceptgraph.integrations.video2mesh.colmap_compat import (
    EXTRACTION_OPTION_ENV,
    LEGACY_EXTRACTION_OPTION,
    LEGACY_MATCHING_OPTION,
    MATCHING_OPTION_ENV,
    MODERN_EXTRACTION_OPTION,
    MODERN_MATCHING_OPTION,
    REAL_BINARY_ENV,
    REAL_SHA256_ENV,
    WRAPPER_SHA256_ENV,
)


VIDEO2MESH_COMMIT = "3ed5ece2974594c26498676e1276f168e6db8962"
GROUNDINGDINO_COMMIT = "856dde20aee659246248e20734ef9ba5214f5e44"
GROUNDINGDINO_CONFIG_SHA256 = (
    "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1"
)
GROUNDINGDINO_CHECKPOINT_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
CLIP_CHECKPOINT_SHA256 = (
    "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
)
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_REPOSITORY = "https://github.com/facebookresearch/sam2.git"
SAM2_TINY_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
SAM2_TINY_CHECKPOINT_SIZE = 156_008_466
SAM2_TINY_CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)
SAM2_TINY_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_VIDEO_RUNTIME_PACKAGES = ("opencv-python-headless==4.11.0.86",)
SAM2_PYTHON_VERSION = "3.10"
SAM2_TORCH_VERSION = "2.5.1"
SAM2_TORCHVISION_VERSION = "0.20.1"
SAM2_OPENCV_VERSION = "4.11.0"
SAM2_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

_RUNNER_DIR = Path("logs") / "conceptgraphs_video2mesh"
_RUN_MANIFEST_NAME = "run_manifest.json"
_MISSING = object()


class Video2MeshRunnerError(RuntimeError):
    """Base class for runner failures."""


class PreflightError(Video2MeshRunnerError):
    """Raised when required repositories, models, or tools are unavailable."""


class UnsafeOutputPathError(Video2MeshRunnerError):
    """Raised when a writable path is not isolated below the output base."""


class StageExecutionError(Video2MeshRunnerError):
    """Raised after an external Video2Mesh stage exits unsuccessfully."""


@dataclass(frozen=True)
class StageCommand:
    """One shell-free external stage invocation."""

    name: str
    argv: tuple[str, ...]
    python: str
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)
    expected_outputs: tuple[str, ...] = ()
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "python": self.python,
            "cwd": self.cwd,
            "env": dict(self.env),
            "expected_outputs": list(self.expected_outputs),
            "optional": self.optional,
        }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get(
    config: Mapping[str, Any],
    *keys: str,
    default: Any = _MISSING,
) -> Any:
    """Get the first present dotted key from a mapping."""

    for dotted_key in keys:
        value: Any = config
        found = True
        for part in dotted_key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value is not None:
            return value
    if default is _MISSING:
        raise KeyError(f"Missing configuration value; tried: {', '.join(keys)}")
    return default


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _config_dir(config: Mapping[str, Any]) -> Path:
    return Path(str(config.get("__config_dir__", Path.cwd()))).expanduser().resolve()


def _portable_config_variables(config_path: Path) -> dict[str, str]:
    """Return stable path variables used by the checked-in pipeline config.

    The defaults assume that external repositories are siblings of this
    checkout, model files live below ``models/``, run artifacts live below
    ``runs/``, and the caller is running from the ``svpp`` Conda environment.
    Every root can be overridden without editing the tracked YAML.
    """

    repo_root = Path(__file__).resolve().parents[3]
    # sys.prefix belongs to the interpreter actually running the pipeline.
    # An ambient CONDA_PREFIX can point at a different shell environment when
    # an absolute Python executable is used, so it must not drive resolution.
    runtime_prefix = Path(sys.prefix).expanduser().resolve()
    default_envs_root = runtime_prefix.parent
    default_conda_root = (
        default_envs_root.parent
        if default_envs_root.name == "envs"
        else runtime_prefix.parent
    )

    def configured_path(name: str, fallback: Path) -> Path:
        value = os.environ.get(name)
        return (
            Path(value).expanduser().resolve()
            if value
            else fallback.expanduser().resolve()
        )

    workspace_root = configured_path("CG_WORKSPACE_ROOT", repo_root.parent)
    model_root = configured_path("CG_MODEL_ROOT", repo_root / "models")
    output_base = configured_path("CG_OUTPUT_BASE", repo_root / "runs")
    dependency_root = configured_path(
        "CG_DEPENDENCY_ROOT",
        repo_root / ".cache" / "video2mesh_dependencies",
    )
    conda_envs_root = configured_path("CG_CONDA_ENVS_ROOT", default_envs_root)
    conda_root = configured_path("CG_CONDA_ROOT", default_conda_root)

    variables = {str(key): str(value) for key, value in os.environ.items()}
    variables.update(
        {
            "CONFIG_DIR": str(config_path.parent),
            "REPO_ROOT": str(repo_root),
            "WORKSPACE_ROOT": str(workspace_root),
            "MODEL_ROOT": str(model_root),
            "OUTPUT_BASE": str(output_base),
            "DEPENDENCY_ROOT": str(dependency_root),
            "CONDA_ENVS_ROOT": str(conda_envs_root),
            "CONDA_ROOT": str(conda_root),
        }
    )
    return variables


def _expand_config_strings(
    value: Any,
    variables: Mapping[str, str],
    *,
    location: str = "<root>",
) -> Any:
    """Recursively expand ``${NAME}`` placeholders with actionable errors."""

    if isinstance(value, Mapping):
        return {
            key: _expand_config_strings(
                item,
                variables,
                location=f"{location}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_config_strings(
                item,
                variables,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and "$" in value:
        try:
            return Template(value).substitute(variables)
        except KeyError as exc:
            missing = str(exc.args[0])
            raise ValueError(
                f"Undefined configuration variable {missing!r} at {location}"
            ) from None
        except ValueError as exc:
            raise ValueError(
                f"Invalid configuration variable syntax at {location}: {exc}"
            ) from None
    return value


def _as_path(
    config: Mapping[str, Any],
    *keys: str,
    default: Any = _MISSING,
) -> Path:
    value = _get(config, *keys, default=default)
    if value is _MISSING:
        raise KeyError(keys[0])
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = _config_dir(config) / path
    return path.resolve()


def _as_command(
    config: Mapping[str, Any],
    *keys: str,
    default: str,
) -> str:
    value = str(_get(config, *keys, default=default))
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute() or "/" in value:
        if not expanded.is_absolute():
            expanded = _config_dir(config) / expanded
        return str(expanded.resolve())
    return value


def _without_internal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _jsonable(value)
        for key, value in config.items()
        if not str(key).startswith("__")
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _path_is_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def _safe_project_root(
    config: Mapping[str, Any],
    project_root: os.PathLike[str] | str,
) -> tuple[Path, Path]:
    output_base = _as_path(
        config,
        "paths.output_base",
        "runtime.output_base",
        "output_base",
    )
    root = Path(project_root).expanduser().resolve()
    if root == output_base or not _is_relative_to(root, output_base):
        raise UnsafeOutputPathError(
            f"Project root must be a child of output_base ({output_base}), got {root}"
        )
    if root in {Path("/"), Path.home().resolve()} or len(root.parts) < 4:
        raise UnsafeOutputPathError(f"Refusing unsafe project root: {root}")
    return root, output_base


def _prepend_pythonpath(*entries: Path) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    # Do not inherit an ambient PYTHONPATH: it could shadow either pinned
    # checkout with an unrelated package.
    for entry in entries:
        text = str(entry.expanduser().resolve())
        if text not in seen:
            ordered.append(text)
            seen.add(text)
    return os.pathsep.join(ordered)


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _probe_video_fps(
    config: Mapping[str, Any],
    video: os.PathLike[str] | str,
) -> float:
    """Read the first video stream's FPS with ffprobe."""

    video_path = Path(video).expanduser().resolve()
    ffprobe = _as_command(
        config,
        "tools.ffprobe",
        "runtime.ffprobe",
        default="ffprobe",
    )
    executable = _resolve_executable(ffprobe)
    if executable is None:
        colmap = _as_command(
            config,
            "tools.colmap_binary",
            "reconstruction.colmap_binary",
            default="colmap",
        )
        colmap_executable = _resolve_executable(colmap)
        if colmap_executable is not None:
            sibling_ffprobe = colmap_executable.parent / "ffprobe"
            if sibling_ffprobe.is_file() and os.access(sibling_ffprobe, os.X_OK):
                executable = sibling_ffprobe.resolve()
    if executable is None:
        video_python = _as_command(
            config,
            "tools.video2mesh_python",
            "runtime.video2mesh_python",
            default=sys.executable,
        )
        python_executable = _resolve_executable(video_python)
        if python_executable is None:
            raise PreflightError(
                "Source-frame bounds require frames.source_fps, ffprobe, or a "
                "Video2Mesh Python with OpenCV"
            )
        probe = subprocess.run(
            [
                str(python_executable),
                "-B",
                "-c",
                (
                    "import cv2,sys;"
                    "cap=cv2.VideoCapture(sys.argv[1]);"
                    "fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0);"
                    "cap.release();"
                    "print(repr(fps))"
                ),
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
            timeout=60,
        )
        if probe.returncode != 0:
            detail = probe.stderr.strip() or probe.stdout.strip()
            raise PreflightError(f"OpenCV FPS probe failed for {video_path}: {detail}")
        try:
            fps = float(probe.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise PreflightError(
                f"Could not parse OpenCV FPS for {video_path}"
            ) from exc
        if not math.isfinite(fps) or fps <= 0:
            raise PreflightError(f"Invalid video FPS reported for {video_path}: {fps}")
        return fps
    result = subprocess.run(
        [
            str(executable),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"ffprobe failed for {video_path}: {detail}")
    try:
        streams = json.loads(result.stdout).get("streams") or []
        stream = streams[0]
        raw_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = float(Fraction(str(raw_rate)))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise PreflightError(
            f"Could not parse video FPS from ffprobe output for {video_path}"
        ) from exc
    if not math.isfinite(fps) or fps <= 0:
        raise PreflightError(f"Invalid video FPS reported for {video_path}: {fps}")
    return fps


def _inject_probed_frame_fps(
    config: Mapping[str, Any],
    video: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Return a config copy with FPS filled for source-frame bounds."""

    resolved = copy.deepcopy(dict(config))
    frames = dict(_get(resolved, "frames", default={}))
    has_frame_bounds = (
        frames.get("start_frame") is not None
        or frames.get("end_frame_inclusive") is not None
    )
    if has_frame_bounds and frames.get("source_fps") is None:
        frames["source_fps"] = _probe_video_fps(resolved, video)
        resolved["frames"] = frames
    return resolved


def load_pipeline_config(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load a JSON/YAML pipeline configuration without changing repository state."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on caller env
            raise RuntimeError(
                "PyYAML is required to load non-JSON Video2Mesh configuration"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Pipeline config must be a mapping: {config_path}")
    config = _expand_config_strings(
        copy.deepcopy(dict(payload)),
        _portable_config_variables(config_path),
    )
    config["__config_path__"] = str(config_path)
    config["__config_dir__"] = str(config_path.parent)
    return config


def compute_frame_window(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve frame/time selection, including an exact source-frame window."""

    resolved = _deep_merge(config, overrides or {})
    frames = dict(_get(resolved, "frames", default={}))
    every = int(frames.get("every", 1))
    max_frames = int(frames.get("max_frames", 200))
    if every < 1:
        raise ValueError("frames.every must be at least 1")
    if max_frames < 0:
        raise ValueError("frames.max_frames must be non-negative")

    start_sec = frames.get("start_sec")
    end_sec = frames.get("end_sec")
    duration_sec = frames.get("duration_sec")
    start_frame = frames.get("start_frame")
    end_frame = frames.get("end_frame_inclusive")
    source_fps = frames.get("source_fps")

    if start_frame is not None or end_frame is not None:
        if start_sec is not None or end_sec is not None or duration_sec is not None:
            raise ValueError("Use source-frame bounds or time bounds, not both")
        if start_frame is None or end_frame is None:
            raise ValueError(
                "frames.start_frame and frames.end_frame_inclusive are both required"
            )
        if source_fps is None or float(source_fps) <= 0:
            raise ValueError(
                "frames.source_fps must be positive for source-frame bounds"
            )
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        source_fps = float(source_fps)
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("Invalid source-frame bounds")
        # Pick timestamps inside the first/last frame intervals.  Video2Mesh
        # floors the start and ceils the exclusive end, so this avoids an
        # off-by-one caused by floating point roundoff.
        start_sec = (start_frame + 0.25) / source_fps
        end_sec = (end_frame + 0.75) / source_fps

    if duration_sec is not None:
        duration_sec = float(duration_sec)
        if duration_sec <= 0:
            raise ValueError("frames.duration_sec must be positive")
        if end_sec is not None:
            raise ValueError("frames.end_sec and duration_sec are mutually exclusive")
    if start_sec is not None:
        start_sec = float(start_sec)
        if start_sec < 0:
            raise ValueError("frames.start_sec must be non-negative")
    if end_sec is not None:
        end_sec = float(end_sec)
        if end_sec < 0 or (start_sec is not None and end_sec <= start_sec):
            raise ValueError("frames.end_sec must be greater than start_sec")

    expected_source_indices: list[int] | None = None
    if start_frame is not None and end_frame is not None:
        candidates = list(range(int(start_frame), int(end_frame) + 1, every))
        if max_frames > 0 and len(candidates) > max_frames:
            if max_frames == 1:
                positions = [0]
            else:
                positions = [
                    int(round(index * (len(candidates) - 1) / (max_frames - 1)))
                    for index in range(max_frames)
                ]
            candidates = [candidates[position] for position in positions]
        expected_source_indices = candidates

    return {
        "every": every,
        "max_frames": max_frames,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "start_frame": start_frame,
        "end_frame_inclusive": end_frame,
        "source_fps": float(source_fps) if source_fps is not None else None,
        "expected_source_indices": expected_source_indices,
    }


def _python_command(
    python: str,
    stage: str,
    project_root: Path,
    *arguments: str,
) -> tuple[str, ...]:
    return (
        python,
        "-B",
        "-m",
        "video2mesh.cli",
        stage,
        "--project-root",
        str(project_root),
        *arguments,
    )


def _conceptgraphs_command(
    python: str,
    stage: str,
    project_root: Path,
    *arguments: str,
) -> tuple[str, ...]:
    return (
        python,
        "-B",
        "-m",
        "conceptgraph.integrations.video2mesh.object_normalization",
        stage,
        "--project-root",
        str(project_root),
        *arguments,
    )


def build_stage_commands(
    config: Mapping[str, Any],
    video: os.PathLike[str] | str,
    scene_id: str,
    project_root: os.PathLike[str] | str,
    overrides: Mapping[str, Any] | None = None,
) -> list[StageCommand]:
    """Build the exact, shell-free Video2Mesh stage sequence."""

    resolved = _deep_merge(config, overrides or {})
    resolved = _inject_probed_frame_fps(resolved, video)
    root, _ = _safe_project_root(resolved, project_root)
    video_path = Path(video).expanduser().resolve()
    v2m_root = _as_path(
        resolved,
        "paths.video2mesh_root",
        "video2mesh.root",
        "video2mesh_root",
    )
    gdino_root = _as_path(
        resolved,
        "paths.groundingdino_root",
        "groundingdino.root",
        "groundingdino_root",
    )
    sam2_source = _as_path(
        resolved,
        "paths.sam2_source",
        "bootstrap.source_dir",
        "sam2.source",
    )
    sam2_checkpoint = _as_path(
        resolved,
        "paths.sam2_checkpoint",
        "bootstrap.checkpoint",
        "models.sam2_checkpoint",
    )
    gdino_config = _as_path(
        resolved,
        "models.groundingdino_config",
        "groundingdino.config",
    )
    gdino_checkpoint = _as_path(
        resolved,
        "models.groundingdino_checkpoint",
        "groundingdino.checkpoint",
    )

    v2m_python = _as_command(
        resolved,
        "tools.video2mesh_python",
        "runtime.video2mesh_python",
        default=sys.executable,
    )
    gdino_python = _as_command(
        resolved,
        "tools.groundingdino_python",
        "runtime.groundingdino_python",
        default=v2m_python,
    )
    sam2_python = _as_command(
        resolved,
        "tools.sam2_python",
        "runtime.sam2_python",
        default=str(
            _as_path(
                resolved,
                "bootstrap.prefix",
                default="/nonexistent/sam2-prefix",
            )
            / "bin"
            / "python"
        ),
    )
    conceptgraphs_python = _as_command(
        resolved,
        "tools.conceptgraphs_python",
        "runtime.conceptgraphs_python",
        default=sys.executable,
    )
    colmap_binary = _as_command(
        resolved,
        "tools.colmap_binary",
        "reconstruction.colmap_binary",
        default="colmap",
    )
    colmap_ok, colmap_detail, colmap_profile = _probe_colmap_cli_compatibility(
        colmap_binary
    )
    if not colmap_ok or colmap_profile is None:
        raise PreflightError(f"COLMAP CLI compatibility probe failed: {colmap_detail}")
    colmap_wrapper = str(colmap_profile["wrapper"])

    common_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root)}
    gdino_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root, gdino_root)}
    sam2_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root, sam2_source)}
    conceptgraphs_root = Path(__file__).resolve().parents[3]
    conceptgraphs_env = {"PYTHONPATH": _prepend_pythonpath(conceptgraphs_root)}
    colmap_env = {
        **common_env,
        **_colmap_compat_environment(colmap_profile),
    }
    frame_window = compute_frame_window(resolved)

    init_argv = _python_command(
        v2m_python,
        "init",
        root,
        "--scene-id",
        str(scene_id),
        "--video",
        str(video_path),
    )

    extract_args: list[str] = [
        "--video",
        str(video_path),
        "--every",
        str(frame_window["every"]),
        "--max-frames",
        str(frame_window["max_frames"]),
        "--renumber",
    ]
    for key, flag in (
        ("start_sec", "--start-sec"),
        ("end_sec", "--end-sec"),
        ("duration_sec", "--duration-sec"),
    ):
        value = frame_window[key]
        if value is not None:
            extract_args.extend([flag, format(float(value), ".17g")])

    reconstruction = dict(_get(resolved, "reconstruction", default={}))
    run_colmap_args = [
        "--colmap-binary",
        colmap_wrapper,
        "--camera-model",
        str(reconstruction.get("camera_model", "PINHOLE")),
        _bool_flag(
            "single-camera",
            _as_bool(reconstruction.get("single_camera", True), name="single_camera"),
        ),
        "--focal-scale",
        str(float(reconstruction.get("focal_scale", 1.2))),
        "--matcher",
        str(reconstruction.get("matcher", "exhaustive")),
        _bool_flag(
            "use-gpu",
            _as_bool(reconstruction.get("use_gpu", False), name="use_gpu"),
        ),
        _bool_flag(
            "refine-focal-length",
            _as_bool(
                reconstruction.get("refine_focal_length", True),
                name="refine_focal_length",
            ),
        ),
        _bool_flag(
            "refine-principal-point",
            _as_bool(
                reconstruction.get("refine_principal_point", False),
                name="refine_principal_point",
            ),
        ),
        _bool_flag(
            "refine-extra-params",
            _as_bool(
                reconstruction.get("refine_extra_params", False),
                name="refine_extra_params",
            ),
        ),
        "--no-dense-reconstruction",
        "--no-overwrite",
    ]

    readiness_args = [
        "--min-frames",
        str(int(reconstruction.get("min_frames", 3))),
        "--min-camera-poses",
        str(int(reconstruction.get("min_camera_poses", 2))),
        "--min-point-count",
        str(int(reconstruction.get("min_point_count", 100))),
        "--min-camera-coverage",
        str(float(reconstruction.get("min_camera_coverage", 0.8))),
        "--min-visible-point-ratio",
        str(float(reconstruction.get("min_visible_point_ratio", 0.05))),
        "--fail-on-not-ready",
    ]

    detection = dict(_get(resolved, "detection", default={}))
    discover_args = [
        "--provider",
        "groundingdino",
        "--anchor-frame-count",
        str(int(detection.get("anchor_frame_count", 5))),
        "--groundingdino-root",
        str(gdino_root),
        "--groundingdino-config",
        str(gdino_config),
        "--groundingdino-checkpoint",
        str(gdino_checkpoint),
        "--groundingdino-device",
        str(detection.get("device", "auto")),
        "--box-threshold",
        str(float(detection.get("box_threshold", 0.28))),
        "--text-threshold",
        str(float(detection.get("text_threshold", 0.25))),
        "--max-objects",
        str(int(detection.get("max_objects", 20))),
        "--nms-iou",
        str(float(detection.get("nms_iou", 0.65))),
        "--instance-iou",
        str(float(detection.get("instance_iou", 0.18))),
        "--instance-center-distance",
        str(float(detection.get("instance_center_distance", 0.75))),
        "--max-instances-per-label",
        str(int(detection.get("max_instances_per_label", 4))),
        _bool_flag(
            "merge-bed-parts",
            _as_bool(detection.get("merge_bed_parts", True), name="merge_bed_parts"),
        ),
        "--single-instance-labels",
        str(detection.get("single_instance_labels", "bed")),
        "--keep-raw-detections",
    ]
    queries = detection.get("queries")
    if queries:
        discover_args.extend(["--queries", str(queries)])

    tracking = dict(_get(resolved, "tracking", default={}))
    normalization = dict(_get(resolved, "normalization", default={}))
    normalization_args = (
        "--config-json",
        json.dumps(normalization, sort_keys=True, separators=(",", ":")),
    )
    identity_quality = dict(_get(resolved, "identity_quality", default={}))
    fail_on_unresolved_identities = _as_bool(
        identity_quality.pop("fail_on_unresolved", True),
        name="identity_quality.fail_on_unresolved",
    )
    identity_quality_args = [
        "--quality-config-json",
        json.dumps(identity_quality, sort_keys=True, separators=(",", ":")),
    ]
    if fail_on_unresolved_identities:
        identity_quality_args.append("--fail-on-unresolved")
    sam2_model_cfg = str(
        _get(
            resolved,
            "models.sam2_model_cfg",
            "sam2.model_cfg",
            default=SAM2_TINY_MODEL_CFG,
        )
    )
    track_args = [
        "--prompts",
        str(root / "masks" / "object_prompts_normalized.json"),
        "--output-dir",
        str(root / "masks" / "2d_raw"),
        "--mask-backend",
        "sam2",
        "--sam2-checkpoint",
        str(sam2_checkpoint),
        "--sam2-model-cfg",
        sam2_model_cfg,
        "--sam2-device",
        str(tracking.get("sam2_device", "auto")),
        "--sam2-mask-threshold",
        str(float(tracking.get("mask_threshold", 0.0))),
        _bool_flag(
            "sam2-autocast",
            _as_bool(tracking.get("autocast", True), name="autocast"),
        ),
        _bool_flag(
            "sam2-offload-video-to-cpu",
            _as_bool(
                tracking.get("offload_video_to_cpu", True),
                name="offload_video_to_cpu",
            ),
        ),
        _bool_flag(
            "sam2-offload-state-to-cpu",
            _as_bool(
                tracking.get("offload_state_to_cpu", True),
                name="offload_state_to_cpu",
            ),
        ),
        "--max-frames",
        str(int(tracking.get("max_frames", 0))),
    ]

    quality = dict(_get(resolved, "quality", default={}))
    quality_args = [
        "--min-coverage-ratio",
        str(float(quality.get("min_coverage_ratio", 0.7))),
        "--max-area-cv",
        str(float(quality.get("max_area_cv", 1.0))),
        "--max-center-jump-ratio",
        str(float(quality.get("max_center_jump_ratio", 0.35))),
    ]

    fusion = dict(_get(resolved, "fusion", default={}))
    fuse_args = [
        "--mask-root",
        str(root / "masks" / "2d_fusion"),
        "--fusion-mode",
        str(fusion.get("mode", "probability")),
        "--min-probability",
        str(float(fusion.get("min_probability", 0.5))),
        "--min-votes",
        str(int(fusion.get("min_votes", 1))),
        _bool_flag(
            "occlusion-filter",
            _as_bool(fusion.get("occlusion_filter", True), name="occlusion_filter"),
        ),
        "--depth-tolerance",
        str(float(fusion.get("depth_tolerance", 0.05))),
        "--relative-depth-tolerance",
        str(float(fusion.get("relative_depth_tolerance", 0.03))),
        _bool_flag(
            "exclusive-objects",
            _as_bool(fusion.get("exclusive_objects", True), name="exclusive_objects"),
        ),
    ]

    commands = [
        StageCommand(
            "init",
            init_argv,
            v2m_python,
            str(v2m_root),
            common_env,
            (str(root / "manifest.json"),),
        ),
        StageCommand(
            "extract_frames",
            _python_command(v2m_python, "extract-frames", root, *extract_args),
            v2m_python,
            str(v2m_root),
            common_env,
            (str(root / "scene" / "frames_manifest.json"),),
        ),
        StageCommand(
            "run_colmap",
            _python_command(v2m_python, "run-colmap", root, *run_colmap_args),
            v2m_python,
            str(v2m_root),
            colmap_env,
            (
                str(root / "scene" / "cameras" / "camera_info.json"),
                str(root / "scene" / "reconstruction" / "point_cloud.ply"),
                str(root / "external" / "colmap" / "colmap_run_report.json"),
            ),
        ),
        StageCommand(
            "reconstruction_readiness",
            _python_command(
                v2m_python,
                "reconstruction-readiness",
                root,
                *readiness_args,
            ),
            v2m_python,
            str(v2m_root),
            common_env,
            (str(root / "simulator_assets" / "reconstruction_readiness_report.json"),),
        ),
        StageCommand(
            "discover_object_prompts",
            _python_command(
                gdino_python,
                "discover-object-prompts",
                root,
                *discover_args,
            ),
            gdino_python,
            str(v2m_root),
            gdino_env,
            (
                str(root / "masks" / "object_prompts_groundingdino.json"),
                str(root / "masks" / "object_labels.json"),
            ),
        ),
        StageCommand(
            "normalize_object_prompts",
            _conceptgraphs_command(
                conceptgraphs_python,
                "normalize-prompts",
                root,
                *normalization_args,
            ),
            conceptgraphs_python,
            str(conceptgraphs_root),
            conceptgraphs_env,
            (str(root / "masks" / "object_prompts_normalized.json"),),
        ),
        StageCommand(
            "track_masks",
            _python_command(sam2_python, "track-masks", root, *track_args),
            sam2_python,
            str(v2m_root),
            sam2_env,
            (str(root / "masks" / "2d_raw" / "tracking_manifest.json"),),
        ),
        StageCommand(
            "normalize_mask_tracks",
            _conceptgraphs_command(
                conceptgraphs_python,
                "normalize-tracks",
                root,
                *normalization_args,
            ),
            conceptgraphs_python,
            str(conceptgraphs_root),
            conceptgraphs_env,
            (
                str(root / "masks" / "2d" / "tracking_manifest.json"),
                str(root / "masks" / "2d_fusion" / "tracking_manifest.json"),
            ),
        ),
        StageCommand(
            "identity_quality_report",
            _conceptgraphs_command(
                conceptgraphs_python,
                "inspect-identities",
                root,
                *identity_quality_args,
            ),
            conceptgraphs_python,
            str(conceptgraphs_root),
            conceptgraphs_env,
            (
                str(
                    root
                    / "simulator_assets"
                    / "identity_quality_report.json"
                ),
            ),
        ),
        StageCommand(
            "mask_track_quality_report",
            _python_command(
                v2m_python,
                "mask-track-quality-report",
                root,
                *quality_args,
            ),
            v2m_python,
            str(v2m_root),
            common_env,
            (str(root / "simulator_assets" / "mask_track_quality_report.json"),),
        ),
        StageCommand(
            "fuse_masks",
            _python_command(v2m_python, "fuse-masks", root, *fuse_args),
            v2m_python,
            str(v2m_root),
            common_env,
            (str(root / "masks" / "3d" / "object_masks.json"),),
        ),
        StageCommand(
            "finalize_fusion_manifest",
            _conceptgraphs_command(
                conceptgraphs_python,
                "finalize-fusion",
                root,
            ),
            conceptgraphs_python,
            str(conceptgraphs_root),
            conceptgraphs_env,
            (str(root / "masks" / "3d" / "object_masks.json"),),
        ),
    ]
    for command in commands:
        joined = " ".join(command.argv)
        unsafe_argument = next(
            (
                argument
                for argument in command.argv
                if argument == "--overwrite" or argument.startswith("--clear")
            ),
            None,
        )
        if "run_video2mesh_quick.sh" in joined or unsafe_argument is not None:
            raise Video2MeshRunnerError(
                f"Unsafe argument unexpectedly present in {command.name}: {joined}"
            )
    return commands


def _resolve_executable(command: str) -> Path | None:
    candidate = Path(command).expanduser()
    if candidate.is_absolute() or "/" in command:
        resolved = candidate.resolve()
        return resolved if resolved.is_file() and os.access(resolved, os.X_OK) else None
    found = shutil.which(command)
    return Path(found).resolve() if found else None


def _colmap_compat_wrapper_path() -> Path:
    return Path(__file__).resolve().with_name("colmap_compat.py")


def _colmap_compat_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    return {
        REAL_BINARY_ENV: str(profile["binary"]),
        REAL_SHA256_ENV: str(profile["binary_sha256"]),
        WRAPPER_SHA256_ENV: str(profile["wrapper_sha256"]),
        EXTRACTION_OPTION_ENV: str(profile["extraction_gpu_option"]),
        MATCHING_OPTION_ENV: str(profile["matching_gpu_option"]),
    }


def _run_colmap_probe(
    executable: Path,
    *arguments: str,
) -> tuple[int, str]:
    try:
        result = subprocess.run(
            [str(executable), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return result.returncode, result.stdout


def _select_colmap_gpu_option(
    help_output: str,
    *,
    legacy: str,
    modern: str,
    subcommand: str,
) -> str:
    if modern in help_output:
        return modern
    if legacy in help_output:
        return legacy
    raise PreflightError(f"COLMAP {subcommand} exposes neither {modern} nor {legacy}")


def _probe_colmap_cli_compatibility(
    command: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Probe the installed COLMAP option groups without modifying any data."""

    executable = _resolve_executable(command)
    if executable is None:
        return False, f"COLMAP executable not found: {command}", None
    wrapper = _colmap_compat_wrapper_path()
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        return (
            False,
            f"COLMAP compatibility wrapper is not executable: {wrapper}",
            None,
        )

    version_returncode, version_output = _run_colmap_probe(executable, "version")
    if version_returncode != 0:
        version_returncode, version_output = _run_colmap_probe(executable, "-h")
    version_lines = [
        line.strip() for line in version_output.splitlines() if line.strip()
    ]
    version = version_lines[0] if version_lines else "unknown"
    if version_returncode != 0:
        return False, f"Cannot query COLMAP version: {version_output.strip()}", None

    help_outputs: dict[str, str] = {}
    for subcommand in ("feature_extractor", "exhaustive_matcher", "sequential_matcher"):
        returncode, output = _run_colmap_probe(executable, subcommand, "-h")
        if returncode != 0:
            return (
                False,
                f"Cannot query COLMAP {subcommand} options: {output.strip()}",
                None,
            )
        help_outputs[subcommand] = output

    try:
        extraction_option = _select_colmap_gpu_option(
            help_outputs["feature_extractor"],
            legacy=LEGACY_EXTRACTION_OPTION,
            modern=MODERN_EXTRACTION_OPTION,
            subcommand="feature_extractor",
        )
        exhaustive_option = _select_colmap_gpu_option(
            help_outputs["exhaustive_matcher"],
            legacy=LEGACY_MATCHING_OPTION,
            modern=MODERN_MATCHING_OPTION,
            subcommand="exhaustive_matcher",
        )
        sequential_option = _select_colmap_gpu_option(
            help_outputs["sequential_matcher"],
            legacy=LEGACY_MATCHING_OPTION,
            modern=MODERN_MATCHING_OPTION,
            subcommand="sequential_matcher",
        )
    except PreflightError as exc:
        return False, str(exc), None
    if exhaustive_option != sequential_option:
        return (
            False,
            "COLMAP matcher commands expose inconsistent GPU option groups: "
            f"exhaustive={exhaustive_option}, sequential={sequential_option}",
            None,
        )

    profile: dict[str, Any] = {
        "binary": str(executable),
        "binary_sha256": _sha256_file(executable),
        "version": version,
        "wrapper": str(wrapper),
        "wrapper_sha256": _sha256_file(wrapper),
        "extraction_gpu_option": extraction_option,
        "matching_gpu_option": exhaustive_option,
        "translations": {
            LEGACY_EXTRACTION_OPTION: extraction_option,
            LEGACY_MATCHING_OPTION: exhaustive_option,
        },
    }
    smoke_environment = os.environ.copy()
    smoke_environment.update(_colmap_compat_environment(profile))
    smoke_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        smoke = subprocess.run(
            [str(wrapper), "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=smoke_environment,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"COLMAP compatibility wrapper smoke test failed: {exc}", None
    if smoke.returncode != 0:
        detail = smoke.stderr.strip() or smoke.stdout.strip()
        return (
            False,
            "COLMAP compatibility wrapper could not delegate to the real "
            f"binary: {detail}",
            None,
        )

    detail = (
        f"{version}; binary={executable}; "
        f"extraction_option={extraction_option}; "
        f"matching_option={exhaustive_option}; wrapper={wrapper}"
    )
    return True, detail, profile


def _git_head(repository: Path, git: str) -> tuple[str | None, str | None]:
    executable = _resolve_executable(git)
    if executable is None:
        return None, f"git executable not found: {git}"
    if not repository.is_dir():
        return None, f"repository not found: {repository}"
    result = subprocess.run(
        [str(executable), "-C", str(repository), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return None, f"cannot read git HEAD for {repository}: {detail}"
    return result.stdout.strip(), None


def _git_tracked_worktree_clean(
    repository: Path,
    git: str,
) -> tuple[bool, str]:
    executable = _resolve_executable(git)
    if executable is None:
        return False, f"git executable not found: {git}"
    if not repository.is_dir():
        return False, f"repository not found: {repository}"
    result = subprocess.run(
        [
            str(executable),
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return False, f"cannot inspect tracked worktree for {repository}: {detail}"
    dirty = result.stdout.strip()
    if dirty:
        return False, f"tracked worktree has modifications: {dirty}"
    return True, f"tracked worktree is clean: {repository}"


def _record_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    item = {
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "detail": detail,
    }
    checks.append(item)
    if required and not ok:
        errors.append(f"{name}: {detail}")


def _check_import(
    python: str,
    module: str,
    env_overlay: Mapping[str, str],
) -> tuple[bool, str]:
    executable = _resolve_executable(python)
    if executable is None:
        return False, f"python executable not found: {python}"
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in env_overlay.items()})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            str(executable),
            "-B",
            "-c",
            (f"import importlib;importlib.import_module({module!r})"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )
    if result.returncode == 0:
        return True, f"{module} is importable with {executable}"
    detail = result.stderr.strip() or result.stdout.strip() or "module not found"
    return False, f"{module} is not importable with {executable}: {detail}"


def _check_sam2_runtime(
    python: str,
    env_overlay: Mapping[str, str],
    *,
    require_cuda: bool,
) -> tuple[bool, str, dict[str, Any] | None]:
    executable = _resolve_executable(python)
    if executable is None:
        return False, f"python executable not found: {python}", None
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in env_overlay.items()})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe = (
        "import cv2,json,platform,torch,torchvision;"
        "print(json.dumps({"
        "'python':platform.python_version(),"
        "'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,"
        "'opencv':cv2.__version__,"
        "'cuda_available':bool(torch.cuda.is_available())"
        "}))"
    )
    result = subprocess.run(
        [str(executable), "-B", "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return False, f"SAM2 runtime import failed: {detail}", None
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return False, f"Cannot parse SAM2 runtime versions: {exc}", None
    versions_ok = (
        ".".join(str(payload.get("python", "")).split(".")[:2]) == SAM2_PYTHON_VERSION
        and str(payload.get("torch", "")).split("+", 1)[0] == SAM2_TORCH_VERSION
        and str(payload.get("torchvision", "")).split("+", 1)[0]
        == SAM2_TORCHVISION_VERSION
        and str(payload.get("opencv", "")) == SAM2_OPENCV_VERSION
    )
    cuda_ok = bool(payload.get("cuda_available")) or not require_cuda
    ok = versions_ok and cuda_ok
    detail = (
        f"runtime={payload}; required python={SAM2_PYTHON_VERSION}, "
        f"torch={SAM2_TORCH_VERSION}, "
        f"torchvision={SAM2_TORCHVISION_VERSION}, opencv={SAM2_OPENCV_VERSION}, "
        f"cuda_required={require_cuda}"
    )
    return ok, detail, payload


def preflight_environment(
    config: Mapping[str, Any],
    video: os.PathLike[str] | str | None = None,
    project_root: os.PathLike[str] | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit all external dependencies without modifying them."""

    resolved_config = _deep_merge(config, overrides or {})
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict[str, Any] = {
        "expected_commits": {
            "video2mesh": VIDEO2MESH_COMMIT,
            "groundingdino": GROUNDINGDINO_COMMIT,
            "sam2": SAM2_COMMIT,
        }
    }

    configured_versions = {
        "video2mesh": str(
            _get(
                resolved_config,
                "versions.video2mesh_commit",
                default=VIDEO2MESH_COMMIT,
            )
        ),
        "groundingdino": str(
            _get(
                resolved_config,
                "versions.groundingdino_commit",
                default=GROUNDINGDINO_COMMIT,
            )
        ),
        "sam2": str(_get(resolved_config, "versions.sam2_commit", default=SAM2_COMMIT)),
    }
    expected_versions = {
        "video2mesh": VIDEO2MESH_COMMIT,
        "groundingdino": GROUNDINGDINO_COMMIT,
        "sam2": SAM2_COMMIT,
    }
    for name, expected in expected_versions.items():
        actual = configured_versions[name]
        _record_check(
            checks,
            errors,
            f"configured_{name}_commit",
            actual == expected,
            f"configured={actual}; required={expected}",
        )
    configured_sam2_sha256 = str(
        _get(
            resolved_config,
            "bootstrap.checkpoint_sha256",
            "models.sam2_checkpoint_sha256",
            default=SAM2_TINY_CHECKPOINT_SHA256,
        )
    )
    _record_check(
        checks,
        errors,
        "configured_sam2_checkpoint_sha256",
        configured_sam2_sha256 == SAM2_TINY_CHECKPOINT_SHA256,
        (
            f"configured={configured_sam2_sha256}; "
            f"required={SAM2_TINY_CHECKPOINT_SHA256}"
        ),
    )
    configured_gdino_hashes = {
        "groundingdino_config": str(
            _get(
                resolved_config,
                "models.groundingdino_config_sha256",
                default=GROUNDINGDINO_CONFIG_SHA256,
            )
        ),
        "groundingdino_checkpoint": str(
            _get(
                resolved_config,
                "models.groundingdino_checkpoint_sha256",
                default=GROUNDINGDINO_CHECKPOINT_SHA256,
            )
        ),
    }
    expected_gdino_hashes = {
        "groundingdino_config": GROUNDINGDINO_CONFIG_SHA256,
        "groundingdino_checkpoint": GROUNDINGDINO_CHECKPOINT_SHA256,
    }
    for name, expected_sha256 in expected_gdino_hashes.items():
        configured_sha256 = configured_gdino_hashes[name]
        _record_check(
            checks,
            errors,
            f"configured_{name}_sha256",
            configured_sha256 == expected_sha256,
            f"configured={configured_sha256}; required={expected_sha256}",
        )
    configured_clip_sha256 = str(
        _get(
            resolved_config,
            "models.clip_model_sha256",
            default=CLIP_CHECKPOINT_SHA256,
        )
    )
    _record_check(
        checks,
        errors,
        "configured_clip_model_sha256",
        configured_clip_sha256 == CLIP_CHECKPOINT_SHA256,
        (f"configured={configured_clip_sha256}; required={CLIP_CHECKPOINT_SHA256}"),
    )
    configured_bootstrap_runtime = {
        "python_version": str(
            _get(
                resolved_config,
                "bootstrap.python_version",
                default=SAM2_PYTHON_VERSION,
            )
        ),
        "torch_version": str(
            _get(
                resolved_config,
                "bootstrap.torch_version",
                default=SAM2_TORCH_VERSION,
            )
        ),
        "torchvision_version": str(
            _get(
                resolved_config,
                "bootstrap.torchvision_version",
                default=SAM2_TORCHVISION_VERSION,
            )
        ),
        "pytorch_index_url": str(
            _get(
                resolved_config,
                "bootstrap.pytorch_index_url",
                default=SAM2_PYTORCH_INDEX_URL,
            )
        ),
    }
    expected_bootstrap_runtime = {
        "python_version": SAM2_PYTHON_VERSION,
        "torch_version": SAM2_TORCH_VERSION,
        "torchvision_version": SAM2_TORCHVISION_VERSION,
        "pytorch_index_url": SAM2_PYTORCH_INDEX_URL,
    }
    for name, expected_value in expected_bootstrap_runtime.items():
        configured_value = configured_bootstrap_runtime[name]
        _record_check(
            checks,
            errors,
            f"configured_sam2_{name}",
            configured_value == expected_value,
            f"configured={configured_value}; required={expected_value}",
        )

    try:
        v2m_root = _as_path(
            resolved_config,
            "paths.video2mesh_root",
            "video2mesh.root",
            "video2mesh_root",
        )
        gdino_root = _as_path(
            resolved_config,
            "paths.groundingdino_root",
            "groundingdino.root",
            "groundingdino_root",
        )
        sam2_root = _as_path(
            resolved_config,
            "paths.sam2_source",
            "bootstrap.source_dir",
            "sam2.source",
        )
        gdino_config = _as_path(
            resolved_config,
            "models.groundingdino_config",
            "groundingdino.config",
        )
        gdino_checkpoint = _as_path(
            resolved_config,
            "models.groundingdino_checkpoint",
            "groundingdino.checkpoint",
        )
        sam2_checkpoint = _as_path(
            resolved_config,
            "paths.sam2_checkpoint",
            "bootstrap.checkpoint",
            "models.sam2_checkpoint",
        )
        clip_model_path = _as_path(
            resolved_config,
            "models.clip_model_path",
        )
        clip_checkpoint = clip_model_path / "pytorch_model.bin"
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"configuration_paths: {exc}")
        report = {
            "ok": False,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "resolved": resolved,
        }
        if raise_on_error:
            raise PreflightError("; ".join(errors))
        return report

    git = _as_command(
        resolved_config,
        "tools.git",
        "bootstrap.git",
        default="git",
    )
    repo_specs = (
        ("video2mesh", v2m_root, VIDEO2MESH_COMMIT),
        ("groundingdino", gdino_root, GROUNDINGDINO_COMMIT),
        ("sam2", sam2_root, SAM2_COMMIT),
    )
    for name, repository, expected in repo_specs:
        head, git_error = _git_head(repository, git)
        ok = git_error is None and head == expected
        detail = (
            git_error
            if git_error
            else f"HEAD={head}; required={expected}; root={repository}"
        )
        _record_check(checks, errors, f"{name}_repository", ok, str(detail))
        resolved[f"{name}_root"] = str(repository)
        resolved[f"{name}_commit"] = head
        if ok:
            clean, clean_detail = _git_tracked_worktree_clean(repository, git)
        else:
            clean = False
            clean_detail = "not checked because repository/commit validation failed"
        _record_check(
            checks,
            errors,
            f"{name}_tracked_worktree_clean",
            clean,
            clean_detail,
            required=ok,
        )

    model_specs = (
        (
            "groundingdino_config",
            gdino_config,
            GROUNDINGDINO_CONFIG_SHA256,
        ),
        (
            "groundingdino_checkpoint",
            gdino_checkpoint,
            GROUNDINGDINO_CHECKPOINT_SHA256,
        ),
        (
            "clip_checkpoint",
            clip_checkpoint,
            CLIP_CHECKPOINT_SHA256,
        ),
    )
    for name, path, expected_sha256 in model_specs:
        file_ok = path.is_file() and path.stat().st_size > 0
        actual_sha256 = _sha256_file(path) if file_ok else None
        ok = file_ok and actual_sha256 == expected_sha256
        _record_check(
            checks,
            errors,
            name,
            ok,
            (
                f"{path}; sha256={actual_sha256}; required={expected_sha256}"
                if file_ok
                else f"missing or empty: {path}"
            ),
        )
        resolved[name] = str(path)
        resolved[f"{name}_sha256"] = actual_sha256
    sam2_checkpoint_ok = (
        sam2_checkpoint.is_file()
        and sam2_checkpoint.stat().st_size == SAM2_TINY_CHECKPOINT_SIZE
    )
    sam2_checkpoint_sha256: str | None = None
    if sam2_checkpoint_ok:
        sam2_checkpoint_sha256 = _sha256_file(sam2_checkpoint)
        sam2_checkpoint_ok = sam2_checkpoint_sha256 == SAM2_TINY_CHECKPOINT_SHA256
    _record_check(
        checks,
        errors,
        "sam2_checkpoint",
        sam2_checkpoint_ok,
        (
            f"{sam2_checkpoint}; size={sam2_checkpoint.stat().st_size}; "
            f"sha256={sam2_checkpoint_sha256}"
            if sam2_checkpoint.is_file()
            else f"missing: {sam2_checkpoint}"
        ),
    )
    resolved["sam2_checkpoint"] = str(sam2_checkpoint)
    resolved["sam2_checkpoint_sha256"] = sam2_checkpoint_sha256
    resolved["clip_model_path"] = str(clip_model_path)

    v2m_python = _as_command(
        resolved_config,
        "tools.video2mesh_python",
        "runtime.video2mesh_python",
        default=sys.executable,
    )
    gdino_python = _as_command(
        resolved_config,
        "tools.groundingdino_python",
        "runtime.groundingdino_python",
        default=v2m_python,
    )
    sam2_python = _as_command(
        resolved_config,
        "tools.sam2_python",
        "runtime.sam2_python",
        default=str(
            _as_path(resolved_config, "bootstrap.prefix", default="/nonexistent")
            / "bin"
            / "python"
        ),
    )
    conceptgraphs_python = _as_command(
        resolved_config,
        "tools.conceptgraphs_python",
        "runtime.conceptgraphs_python",
        default=sys.executable,
    )
    colmap = _as_command(
        resolved_config,
        "tools.colmap_binary",
        "reconstruction.colmap_binary",
        default="colmap",
    )
    ffprobe = _as_command(
        resolved_config,
        "tools.ffprobe",
        "runtime.ffprobe",
        default="ffprobe",
    )
    common_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root)}
    gdino_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root, gdino_root)}
    sam2_env = {"PYTHONPATH": _prepend_pythonpath(v2m_root, sam2_root)}

    executable_specs = (
        ("video2mesh_python", v2m_python),
        ("groundingdino_python", gdino_python),
        ("sam2_python", sam2_python),
        ("conceptgraphs_python", conceptgraphs_python),
        ("colmap_binary", colmap),
        ("ffprobe", ffprobe),
    )
    for name, command in executable_specs:
        executable = _resolve_executable(command)
        _record_check(
            checks,
            errors,
            name,
            executable is not None,
            str(executable) if executable else f"not executable: {command}",
        )
        resolved[name] = str(executable) if executable else command

    colmap_compat_ok, colmap_compat_detail, colmap_compat_profile = (
        _probe_colmap_cli_compatibility(colmap)
    )
    _record_check(
        checks,
        errors,
        "colmap_cli_compatibility",
        colmap_compat_ok,
        colmap_compat_detail,
    )
    resolved["colmap_cli_compatibility"] = colmap_compat_profile

    configured_conversion_executable = _resolve_executable(conceptgraphs_python)
    caller_executable = Path(sys.executable).resolve()
    conversion_matches_caller = (
        configured_conversion_executable is not None
        and configured_conversion_executable == caller_executable
    )
    _record_check(
        checks,
        errors,
        "conceptgraphs_python_matches_caller",
        conversion_matches_caller,
        (
            f"configured={configured_conversion_executable or conceptgraphs_python}; "
            f"caller={caller_executable}"
        ),
    )

    for check_name, python, module, environment in (
        ("video2mesh_import", v2m_python, "video2mesh.cli", common_env),
        ("groundingdino_import", gdino_python, "groundingdino", gdino_env),
        ("sam2_import", sam2_python, "sam2", sam2_env),
        ("sam2_cv2_import", sam2_python, "cv2", sam2_env),
    ):
        ok, detail = _check_import(python, module, environment)
        _record_check(checks, errors, check_name, ok, detail)
    for module in ("numpy", "PIL", "torch", "transformers"):
        ok, detail = _check_import(conceptgraphs_python, module, {})
        _record_check(
            checks,
            errors,
            f"conceptgraphs_{module.lower()}_import",
            ok,
            detail,
        )

    require_sam2_cuda = (
        str(
            _get(
                resolved_config,
                "tracking.sam2_device",
                default="auto",
            )
        )
        .lower()
        .startswith("cuda")
    )
    sam2_runtime_ok, sam2_runtime_detail, sam2_runtime = _check_sam2_runtime(
        sam2_python,
        sam2_env,
        require_cuda=require_sam2_cuda,
    )
    _record_check(
        checks,
        errors,
        "sam2_runtime_versions",
        sam2_runtime_ok,
        sam2_runtime_detail,
    )
    resolved["sam2_runtime"] = sam2_runtime

    if video is not None:
        video_path = Path(video).expanduser().resolve()
        video_ok = video_path.is_file() and video_path.stat().st_size > 0
        _record_check(
            checks,
            errors,
            "input_video",
            video_ok,
            str(video_path) if video_ok else f"missing or empty: {video_path}",
        )
        resolved["video"] = str(video_path)
        if video_ok:
            resolved["video_sha256"] = _sha256_file(video_path)
            resolved["video_size_bytes"] = video_path.stat().st_size

    if project_root is not None:
        try:
            root, output_base = _safe_project_root(resolved_config, project_root)
            _record_check(
                checks,
                errors,
                "isolated_project_root",
                True,
                f"{root} is below {output_base}",
            )
            if root.exists() and not root.is_dir():
                _record_check(
                    checks,
                    errors,
                    "project_root_type",
                    False,
                    f"project root is not a directory: {root}",
                )
            elif _path_is_nonempty(root):
                warnings.append(
                    f"Project root is non-empty and is usable only with resume: {root}"
                )
            resolved["project_root"] = str(root)
            resolved["output_base"] = str(output_base)
        except (KeyError, OSError, UnsafeOutputPathError, ValueError) as exc:
            _record_check(
                checks,
                errors,
                "isolated_project_root",
                False,
                str(exc),
            )

    try:
        frame_config = (
            _inject_probed_frame_fps(resolved_config, video)
            if video is not None
            else resolved_config
        )
        resolved_frame_window = compute_frame_window(frame_config)
        resolved["frame_window"] = resolved_frame_window
    except (PreflightError, TypeError, ValueError) as exc:
        _record_check(
            checks,
            errors,
            "frame_window",
            False,
            str(exc),
        )
    else:
        _record_check(
            checks,
            errors,
            "frame_window",
            True,
            "frame selection is valid",
        )

    report = {
        "ok": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "resolved": resolved,
    }
    if errors and raise_on_error:
        raise PreflightError("; ".join(errors))
    return report


def _expected_outputs_exist(command: StageCommand) -> tuple[bool, list[str]]:
    missing = [
        output
        for output in command.expected_outputs
        if not Path(output).is_file() or Path(output).stat().st_size <= 0
    ]
    return not missing, missing


def _command_digest(command: StageCommand) -> str:
    return _sha256_json(
        {
            "name": command.name,
            "argv": command.argv,
            "cwd": command.cwd,
            "env": command.env,
            "expected_outputs": command.expected_outputs,
        }
    )


def _stage_marker_path(project_root: Path, stage_name: str) -> Path:
    return project_root / _RUNNER_DIR / "stage_markers" / f"{stage_name}.json"


def _stage_artifact_paths(
    project_root: Path,
    stage_name: str,
) -> tuple[Path, ...]:
    """Return stage outputs that remain immutable after later stages."""

    artifacts: dict[str, tuple[Path, ...]] = {
        # manifest.json is intentionally updated by every later stage.
        "init": (),
        # run-colmap is allowed to remove frames that COLMAP did not register.
        # Keep the extraction marker tied to its immutable manifest and make
        # the post-COLMAP frame tree part of the run-colmap artifact instead.
        "extract_frames": (project_root / "scene" / "frames_manifest.json",),
        "run_colmap": (
            project_root / "scene" / "frames",
            project_root / "scene" / "cameras" / "camera_info.json",
            project_root / "scene" / "reconstruction" / "point_cloud.ply",
            project_root / "external" / "colmap" / "colmap_run_report.json",
        ),
        "reconstruction_readiness": (
            project_root / "simulator_assets" / "reconstruction_readiness_report.json",
        ),
        # object_labels.json is finalized by track-masks, so hash it there.
        "discover_object_prompts": (
            project_root / "masks" / "object_prompts_groundingdino.json",
        ),
        "normalize_object_prompts": (
            project_root / "masks" / "object_prompts_normalized.json",
        ),
        "track_masks": (
            project_root / "masks" / "2d_raw",
        ),
        "normalize_mask_tracks": (
            project_root / "masks" / "2d",
            project_root / "masks" / "2d_fusion",
            project_root / "masks" / "object_labels.json",
        ),
        "identity_quality_report": (
            project_root
            / "simulator_assets"
            / "identity_quality_report.json",
        ),
        "mask_track_quality_report": (
            project_root / "simulator_assets" / "mask_track_quality_report.json",
        ),
        # finalize_fusion_manifest updates the fusion-produced JSON so that
        # downstream observations refer to uncarved normalized masks.
        "fuse_masks": (),
        "finalize_fusion_manifest": (project_root / "masks" / "3d",),
    }
    return artifacts.get(stage_name, ())


def _artifact_sha256(path: Path, project_root: Path) -> str:
    if path.is_symlink():
        raise StageExecutionError(f"Stage artifact must not be a symlink: {path}")
    resolved = path.resolve()
    if not _is_relative_to(resolved, project_root):
        raise StageExecutionError(
            f"Stage artifact escapes project root {project_root}: {path}"
        )
    if resolved.is_file():
        if resolved.stat().st_size <= 0:
            raise StageExecutionError(f"Stage artifact is empty: {resolved}")
        return _sha256_file(resolved)
    if not resolved.is_dir():
        raise StageExecutionError(f"Stage artifact is missing: {resolved}")

    digest = hashlib.sha256()
    file_count = 0
    for child in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise StageExecutionError(
                f"Stage artifact tree contains a symlink: {child}"
            )
        if not child.is_file():
            continue
        child_resolved = child.resolve()
        if not _is_relative_to(child_resolved, project_root):
            raise StageExecutionError(
                f"Stage artifact escapes project root {project_root}: {child}"
            )
        relative = child.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child_resolved).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
    digest.update(f"files={file_count}".encode("ascii"))
    return digest.hexdigest()


def _stage_artifact_hashes(
    project_root: Path,
    stage_name: str,
) -> dict[str, str]:
    return {
        path.relative_to(project_root).as_posix(): _artifact_sha256(path, project_root)
        for path in _stage_artifact_paths(project_root, stage_name)
    }


def _stage_marker_is_valid(
    project_root: Path,
    marker_path: Path,
    command: StageCommand,
    fingerprint: str,
) -> tuple[bool, str]:
    if not marker_path.is_file():
        return False, "marker is missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read marker: {exc}"
    if marker.get("fingerprint") != fingerprint:
        return False, "marker fingerprint differs"
    if marker.get("command_sha256") != _command_digest(command):
        return False, "marker command differs"
    outputs_ok, missing = _expected_outputs_exist(command)
    if not outputs_ok:
        return False, f"expected outputs missing: {', '.join(missing)}"
    recorded_hashes = marker.get("artifact_sha256")
    if not isinstance(recorded_hashes, Mapping):
        return False, "marker has no artifact SHA-256 mapping"
    try:
        actual_hashes = _stage_artifact_hashes(project_root, command.name)
    except (OSError, StageExecutionError) as exc:
        return False, f"cannot hash stage artifacts: {exc}"
    if dict(recorded_hashes) != actual_hashes:
        return False, "stage artifact SHA-256 differs"
    if marker.get("status") != "completed":
        return False, f"marker status is {marker.get('status')!r}"
    return True, "verified marker, outputs, and artifact SHA-256"


def _stage_residual_paths(project_root: Path, stage_name: str) -> list[Path]:
    candidates: dict[str, tuple[Path, ...]] = {
        "init": (project_root / "manifest.json",),
        "extract_frames": (
            project_root / "scene" / "frames_manifest.json",
            project_root / "scene" / "frames",
        ),
        "run_colmap": (project_root / "external" / "colmap",),
        "reconstruction_readiness": (
            project_root / "simulator_assets" / "reconstruction_readiness_report.json",
        ),
        "discover_object_prompts": (
            project_root / "masks" / "object_prompts_groundingdino.json",
            project_root / "masks" / "object_labels.json",
        ),
        "normalize_object_prompts": (
            project_root / "masks" / "object_prompts_normalized.json",
        ),
        "track_masks": (project_root / "masks" / "2d_raw",),
        "normalize_mask_tracks": (
            project_root / "masks" / "2d",
            project_root / "masks" / "2d_fusion",
        ),
        "identity_quality_report": (
            project_root
            / "simulator_assets"
            / "identity_quality_report.json",
        ),
        "mask_track_quality_report": (
            project_root / "simulator_assets" / "mask_track_quality_report.json",
        ),
        "fuse_masks": (project_root / "masks" / "3d",),
        "finalize_fusion_manifest": (project_root / "masks" / "3d",),
    }
    residuals: list[Path] = []
    for path in candidates.get(stage_name, ()):
        if path.is_dir():
            if _path_is_nonempty(path):
                residuals.append(path)
        elif path.exists():
            residuals.append(path)
    return residuals


def _verify_exact_frame_window(
    project_root: Path,
    frame_window: Mapping[str, Any],
) -> None:
    expected = frame_window.get("expected_source_indices")
    if expected is None:
        return
    manifest_path = project_root / "scene" / "frames_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = [
        int(record["source_frame_index"])
        for record in payload.get("frames", [])
        if isinstance(record, Mapping) and "source_frame_index" in record
    ]
    if actual != list(expected):
        raise StageExecutionError(
            "Extracted source frames do not match the exact requested window: "
            f"expected={list(expected)}, actual={actual}"
        )


def _run_fingerprint(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    video_sha256: str,
    model_sha256: Mapping[str, Any],
    colmap_runtime: Mapping[str, Any],
    scene_id: str,
    project_root: Path,
) -> str:
    return _sha256_json(
        {
            "schema": 2,
            "config": _without_internal_config(config),
            "overrides": overrides,
            "video_sha256": video_sha256,
            "model_sha256": model_sha256,
            "colmap_runtime": colmap_runtime,
            "scene_id": str(scene_id),
            "project_root": str(project_root),
            "commits": {
                "video2mesh": VIDEO2MESH_COMMIT,
                "groundingdino": GROUNDINGDINO_COMMIT,
                "sam2": SAM2_COMMIT,
            },
        }
    )


def run_video2mesh_stages(
    config: Mapping[str, Any],
    video: os.PathLike[str] | str,
    scene_id: str,
    project_root: os.PathLike[str] | str,
    overrides: Mapping[str, Any] | None = None,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the pinned Video2Mesh stages with isolated logs and resume markers."""

    override_values = copy.deepcopy(dict(overrides or {}))
    resolved_config = _deep_merge(config, override_values)
    root, output_base = _safe_project_root(resolved_config, project_root)
    video_path = Path(video).expanduser().resolve()

    if root.exists() and not root.is_dir():
        raise UnsafeOutputPathError(f"Project root is not a directory: {root}")
    nonempty = _path_is_nonempty(root)
    if resume:
        if not nonempty:
            raise Video2MeshRunnerError(
                f"Resume requires an existing non-empty project root: {root}"
            )
    elif nonempty:
        raise UnsafeOutputPathError(
            f"Refusing to use an existing non-empty project root without resume: {root}"
        )

    preflight = preflight_environment(
        config,
        video=video_path,
        project_root=root,
        overrides=override_values,
        raise_on_error=False,
    )
    commands = build_stage_commands(
        config,
        video_path,
        scene_id,
        root,
        overrides=override_values,
    )
    video_sha256 = str(
        preflight.get("resolved", {}).get("video_sha256")
        or (_sha256_file(video_path) if video_path.is_file() else "")
    )
    model_sha256 = {
        "groundingdino_config": preflight.get("resolved", {}).get(
            "groundingdino_config_sha256"
        ),
        "groundingdino_checkpoint": preflight.get("resolved", {}).get(
            "groundingdino_checkpoint_sha256"
        ),
        "clip_checkpoint": preflight.get("resolved", {}).get("clip_checkpoint_sha256"),
        "sam2_checkpoint": preflight.get("resolved", {}).get("sam2_checkpoint_sha256"),
    }
    colmap_runtime = preflight.get("resolved", {}).get("colmap_cli_compatibility") or {}
    fingerprint = _run_fingerprint(
        config,
        override_values,
        video_sha256=video_sha256,
        model_sha256=model_sha256,
        colmap_runtime=colmap_runtime,
        scene_id=scene_id,
        project_root=root,
    )
    frame_config = _inject_probed_frame_fps(resolved_config, video_path)
    frame_window = compute_frame_window(frame_config)
    planned = [command.to_dict() for command in commands]

    if dry_run:
        return {
            "ok": bool(preflight["ok"]),
            "status": "dry_run",
            "project_root": str(root),
            "output_base": str(output_base),
            "fingerprint": fingerprint,
            "model_sha256": model_sha256,
            "colmap_runtime": colmap_runtime,
            "preflight": preflight,
            "frame_window": frame_window,
            "stages": planned,
            "manifest_path": str(root / _RUNNER_DIR / _RUN_MANIFEST_NAME),
        }
    if not preflight["ok"]:
        raise PreflightError("; ".join(preflight["errors"]))

    runner_root = root / _RUNNER_DIR
    logs_root = runner_root / "stage_logs"
    markers_root = runner_root / "stage_markers"
    runner_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    markers_root.mkdir(parents=True, exist_ok=True)
    manifest_path = runner_root / _RUN_MANIFEST_NAME

    if resume:
        if not manifest_path.is_file():
            raise Video2MeshRunnerError(f"Resume manifest is missing: {manifest_path}")
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Video2MeshRunnerError(
                f"Cannot read resume manifest {manifest_path}: {exc}"
            ) from exc
        if previous_manifest.get("fingerprint") != fingerprint:
            raise Video2MeshRunnerError(
                "Resume fingerprint differs from the original input/config/commit "
                f"fingerprint for {root}"
            )

    started_at = time.time()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "scene_id": str(scene_id),
        "project_root": str(root),
        "output_base": str(output_base),
        "input_video": str(video_path),
        "input_video_sha256": video_sha256,
        "fingerprint": fingerprint,
        "versions": {
            "video2mesh_commit": VIDEO2MESH_COMMIT,
            "groundingdino_commit": GROUNDINGDINO_COMMIT,
            "sam2_commit": SAM2_COMMIT,
        },
        "model_sha256": model_sha256,
        "colmap_runtime": colmap_runtime,
        "frame_window": frame_window,
        "config": _without_internal_config(config),
        "overrides": override_values,
        "started_at_unix": started_at,
        "resume": bool(resume),
        "stages": [],
    }
    _write_json_atomic(manifest_path, manifest)

    stage_results: list[dict[str, Any]] = []
    try:
        for command in commands:
            marker_path = _stage_marker_path(root, command.name)
            if resume:
                marker_valid, marker_detail = _stage_marker_is_valid(
                    root, marker_path, command, fingerprint
                )
                if marker_valid:
                    if command.name == "extract_frames":
                        _verify_exact_frame_window(root, frame_window)
                    result = {
                        "name": command.name,
                        "status": "skipped_verified",
                        "detail": marker_detail,
                        "marker": str(marker_path),
                        "command_sha256": _command_digest(command),
                    }
                    stage_results.append(result)
                    manifest["stages"] = stage_results
                    _write_json_atomic(manifest_path, manifest)
                    continue
                residuals = _stage_residual_paths(root, command.name)
                if residuals:
                    raise Video2MeshRunnerError(
                        f"Cannot safely resume stage {command.name}: "
                        f"{marker_detail}; unverified outputs already exist: "
                        + ", ".join(str(path) for path in residuals)
                    )

            log_path = logs_root / f"{command.name}.log"
            environment = os.environ.copy()
            environment.update(
                {str(key): str(value) for key, value in command.env.items()}
            )
            stage_started = time.time()
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    list(command.argv),
                    cwd=command.cwd,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = round(time.time() - stage_started, 3)
            if process.returncode != 0:
                result = {
                    "name": command.name,
                    "status": "failed",
                    "returncode": process.returncode,
                    "elapsed_seconds": elapsed,
                    "log": str(log_path),
                    "command_sha256": _command_digest(command),
                }
                stage_results.append(result)
                raise StageExecutionError(
                    f"Video2Mesh stage {command.name} failed with exit code "
                    f"{process.returncode}; see {log_path}"
                )

            outputs_ok, missing = _expected_outputs_exist(command)
            if not outputs_ok:
                result = {
                    "name": command.name,
                    "status": "failed_missing_outputs",
                    "returncode": process.returncode,
                    "elapsed_seconds": elapsed,
                    "log": str(log_path),
                    "missing_outputs": missing,
                    "command_sha256": _command_digest(command),
                }
                stage_results.append(result)
                raise StageExecutionError(
                    f"Video2Mesh stage {command.name} completed but did not produce: "
                    + ", ".join(missing)
                )
            if command.name == "extract_frames":
                _verify_exact_frame_window(root, frame_window)

            artifact_sha256 = _stage_artifact_hashes(root, command.name)
            marker = {
                "schema_version": 1,
                "status": "completed",
                "stage": command.name,
                "fingerprint": fingerprint,
                "command_sha256": _command_digest(command),
                "expected_outputs": list(command.expected_outputs),
                "artifact_sha256": artifact_sha256,
                "completed_at_unix": time.time(),
                "elapsed_seconds": elapsed,
                "log": str(log_path),
            }
            _write_json_atomic(marker_path, marker)
            result = {
                "name": command.name,
                "status": "completed",
                "returncode": process.returncode,
                "elapsed_seconds": elapsed,
                "log": str(log_path),
                "marker": str(marker_path),
                "command_sha256": marker["command_sha256"],
                "artifact_sha256": artifact_sha256,
            }
            stage_results.append(result)
            manifest["stages"] = stage_results
            _write_json_atomic(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["error_type"] = type(exc).__name__
        manifest["stages"] = stage_results
        manifest["finished_at_unix"] = time.time()
        manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["stages"] = stage_results
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
    _write_json_atomic(manifest_path, manifest)
    return {
        "ok": True,
        "status": "completed",
        "project_root": str(root),
        "output_base": str(output_base),
        "fingerprint": fingerprint,
        "model_sha256": manifest["model_sha256"],
        "frame_window": frame_window,
        "stages": stage_results,
        "manifest_path": str(manifest_path),
    }


def _bootstrap_paths(
    config: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    prefix = _as_path(config, "bootstrap.prefix")
    source = _as_path(
        config,
        "bootstrap.source_dir",
        "paths.sam2_source",
    )
    checkpoint = _as_path(
        config,
        "bootstrap.checkpoint",
        "paths.sam2_checkpoint",
    )
    dependency_root = _as_path(
        config,
        "paths.dependency_root",
        default=str(Path(os.path.commonpath([prefix, source, checkpoint.parent]))),
    )
    if (
        dependency_root
        in {
            Path("/"),
            Path.home().resolve(),
            Path(sys.prefix).resolve(),
        }
        or len(dependency_root.parts) < 3
    ):
        raise UnsafeOutputPathError(
            f"Refusing unsafe dependency_root: {dependency_root}"
        )
    for name, path in (
        ("bootstrap prefix", prefix),
        ("SAM2 source", source),
        ("SAM2 checkpoint", checkpoint),
    ):
        if path in {Path("/"), Path.home().resolve(), Path(sys.prefix).resolve()}:
            raise UnsafeOutputPathError(f"Refusing unsafe {name} path: {path}")
        if not _is_relative_to(path, dependency_root):
            raise UnsafeOutputPathError(
                f"{name} must be below dependency_root {dependency_root}: {path}"
            )
    if len({prefix, source, checkpoint}) != 3:
        raise UnsafeOutputPathError(
            "SAM2 prefix, source, and checkpoint paths must be distinct"
        )
    for left_name, left, right_name, right in (
        ("prefix", prefix, "source", source),
        ("prefix", prefix, "checkpoint", checkpoint),
        ("source", source, "checkpoint", checkpoint),
    ):
        if _is_relative_to(left, right) or _is_relative_to(right, left):
            raise UnsafeOutputPathError(
                f"SAM2 {left_name} and {right_name} paths must be independent: "
                f"{left}, {right}"
            )
    return prefix, source, checkpoint, dependency_root


def _run_bootstrap_command(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env_overlay: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in (env_overlay or {}).items()}
    )
    result = subprocess.run(
        [str(item) for item in argv],
        cwd=str(cwd) if cwd else None,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    record = {
        "name": name,
        "argv": [str(item) for item in argv],
        "env_overlay": dict(env_overlay or {}),
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "output": result.stdout[-12000:],
    }
    if result.returncode != 0:
        raise StageExecutionError(
            f"SAM2 bootstrap step {name} failed with exit code "
            f"{result.returncode}: {result.stdout[-2000:]}"
        )
    return record


def bootstrap_sam2(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create the isolated, pinned SAM2 environment when explicitly called."""

    prefix, source, checkpoint, dependency_root = _bootstrap_paths(config)
    bootstrap = dict(_get(config, "bootstrap", default={}))
    configured_repository = str(bootstrap.get("repository", SAM2_REPOSITORY)).rstrip(
        "/"
    )
    normalized_repository = (
        configured_repository[:-4]
        if configured_repository.endswith(".git")
        else configured_repository
    )
    normalized_official_repository = (
        SAM2_REPOSITORY[:-4] if SAM2_REPOSITORY.endswith(".git") else SAM2_REPOSITORY
    )
    if normalized_repository != normalized_official_repository:
        raise PreflightError(
            f"SAM2 bootstrap repository must be official: {SAM2_REPOSITORY}"
        )
    configured_commit = str(_get(config, "versions.sam2_commit", default=SAM2_COMMIT))
    if configured_commit != SAM2_COMMIT:
        raise PreflightError(
            f"SAM2 commit must be pinned to {SAM2_COMMIT}, got {configured_commit}"
        )
    checkpoint_url = str(bootstrap.get("sam2_checkpoint_url", SAM2_TINY_CHECKPOINT_URL))
    if checkpoint_url != SAM2_TINY_CHECKPOINT_URL:
        raise PreflightError("SAM2 tiny checkpoint URL must be the pinned official URL")
    configured_checkpoint_sha256 = str(
        bootstrap.get("checkpoint_sha256", SAM2_TINY_CHECKPOINT_SHA256)
    )
    if configured_checkpoint_sha256 != SAM2_TINY_CHECKPOINT_SHA256:
        raise PreflightError(
            "SAM2 tiny checkpoint SHA256 must be pinned to "
            f"{SAM2_TINY_CHECKPOINT_SHA256}"
        )

    conda = _as_command(
        config,
        "bootstrap.conda_binary",
        "tools.conda",
        default="conda",
    )
    git = _as_command(config, "tools.git", default="git")
    conda_executable = _resolve_executable(conda)
    git_executable = _resolve_executable(git)
    if conda_executable is None:
        raise PreflightError(f"conda executable not found: {conda}")
    if git_executable is None:
        raise PreflightError(f"git executable not found: {git}")

    python_version = str(bootstrap.get("python_version", SAM2_PYTHON_VERSION))
    torch_version = str(bootstrap.get("torch_version", SAM2_TORCH_VERSION))
    torchvision_version = str(
        bootstrap.get("torchvision_version", SAM2_TORCHVISION_VERSION)
    )
    index_url = str(bootstrap.get("pytorch_index_url", SAM2_PYTORCH_INDEX_URL))
    pinned_runtime_values = (
        ("python_version", python_version, SAM2_PYTHON_VERSION),
        ("torch_version", torch_version, SAM2_TORCH_VERSION),
        (
            "torchvision_version",
            torchvision_version,
            SAM2_TORCHVISION_VERSION,
        ),
        ("pytorch_index_url", index_url, SAM2_PYTORCH_INDEX_URL),
    )
    for name, actual, expected in pinned_runtime_values:
        if actual != expected:
            raise PreflightError(
                f"SAM2 {name} must be pinned to {expected}, got {actual}"
            )
    prefix_python = prefix / "bin" / "python"
    ownership_marker = (
        dependency_root / f".{prefix.name}.conceptgraphs_sam2_bootstrap.json"
    )
    bootstrap_fingerprint = _sha256_json(
        {
            "prefix": str(prefix),
            "source": str(source),
            "checkpoint": str(checkpoint),
            "repository": SAM2_REPOSITORY,
            "commit": SAM2_COMMIT,
            "checkpoint_url": SAM2_TINY_CHECKPOINT_URL,
            "checkpoint_sha256": SAM2_TINY_CHECKPOINT_SHA256,
            "python_version": python_version,
            "torch_version": torch_version,
            "torchvision_version": torchvision_version,
            "index_url": index_url,
            "video_runtime_packages": SAM2_VIDEO_RUNTIME_PACKAGES,
            "sam2_build_cuda": False,
        }
    )

    existing_marker: dict[str, Any] | None = None
    if ownership_marker.is_file():
        try:
            existing_marker = json.loads(ownership_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Video2MeshRunnerError(
                f"Cannot read bootstrap ownership marker {ownership_marker}: {exc}"
            ) from exc
        if existing_marker.get("fingerprint") != bootstrap_fingerprint:
            raise Video2MeshRunnerError(
                f"Existing bootstrap marker has a different fingerprint: "
                f"{ownership_marker}"
            )

    source_ready = False
    if source.exists():
        head, error = _git_head(source, str(git_executable))
        if error or head != SAM2_COMMIT:
            raise Video2MeshRunnerError(
                f"Existing SAM2 source is not the pinned checkout: "
                f"{error or head}; path={source}"
            )
        source_clean, source_clean_detail = _git_tracked_worktree_clean(
            source, str(git_executable)
        )
        if not source_clean:
            raise Video2MeshRunnerError(
                "Refusing to install from a modified SAM2 tracked worktree: "
                f"{source_clean_detail}"
            )
        source_ready = True
    prefix_ready = prefix_python.is_file() and os.access(prefix_python, os.X_OK)
    if prefix.exists() and _path_is_nonempty(prefix) and not prefix_ready:
        if existing_marker is None:
            raise Video2MeshRunnerError(
                f"Refusing to modify an existing unowned prefix: {prefix}"
            )
    if prefix_ready and existing_marker is None:
        raise Video2MeshRunnerError(
            f"Refusing to modify an existing unowned Python prefix: {prefix}"
        )
    checkpoint_ready = False
    if checkpoint.exists():
        if not checkpoint.is_file():
            raise Video2MeshRunnerError(
                f"SAM2 checkpoint path is not a regular file: {checkpoint}"
            )
        checkpoint_size = checkpoint.stat().st_size
        checkpoint_sha256 = _sha256_file(checkpoint)
        if (
            checkpoint_size != SAM2_TINY_CHECKPOINT_SIZE
            or checkpoint_sha256 != SAM2_TINY_CHECKPOINT_SHA256
        ):
            raise Video2MeshRunnerError(
                "Refusing to replace an existing SAM2 checkpoint that does not "
                f"match the pinned artifact: path={checkpoint}, "
                f"size={checkpoint_size}, sha256={checkpoint_sha256}"
            )
        checkpoint_ready = True

    planned_commands: list[dict[str, Any]] = []
    if not prefix_ready:
        planned_commands.append(
            {
                "name": "create_conda_prefix",
                "argv": [
                    str(conda_executable),
                    "create",
                    "--yes",
                    "--prefix",
                    str(prefix),
                    f"python={python_version}",
                    "pip",
                ],
            }
        )
    if not source_ready:
        planned_commands.extend(
            [
                {
                    "name": "clone_sam2",
                    "argv": [
                        str(git_executable),
                        "clone",
                        "--no-checkout",
                        SAM2_REPOSITORY,
                        str(source),
                    ],
                },
                {
                    "name": "checkout_sam2",
                    "argv": [
                        str(git_executable),
                        "-C",
                        str(source),
                        "checkout",
                        "--detach",
                        SAM2_COMMIT,
                    ],
                },
            ]
        )
    if not (
        existing_marker
        and existing_marker.get("status") == "completed"
        and prefix_ready
    ):
        planned_commands.extend(
            [
                {
                    "name": "install_torch",
                    "argv": [
                        str(prefix_python),
                        "-m",
                        "pip",
                        "install",
                        f"torch=={torch_version}",
                        f"torchvision=={torchvision_version}",
                        "--index-url",
                        index_url,
                    ],
                },
                {
                    "name": "install_sam2",
                    "argv": [
                        str(prefix_python),
                        "-m",
                        "pip",
                        "install",
                        str(source),
                    ],
                    "env": {"SAM2_BUILD_CUDA": "0"},
                },
                {
                    "name": "install_video_runtime",
                    "argv": [
                        str(prefix_python),
                        "-m",
                        "pip",
                        "install",
                        *SAM2_VIDEO_RUNTIME_PACKAGES,
                    ],
                },
            ]
        )
    if not checkpoint_ready:
        planned_commands.append(
            {
                "name": "download_checkpoint",
                "url": SAM2_TINY_CHECKPOINT_URL,
                "destination": str(checkpoint),
            }
        )
    planned_commands.append(
        {
            "name": "verify_sam2",
            "argv": [
                str(prefix_python),
                "-B",
                "-c",
                "import cv2, sam2, torch; print(torch.__version__, cv2.__version__)",
            ],
        }
    )

    if dry_run:
        return {
            "ok": True,
            "status": "dry_run",
            "fingerprint": bootstrap_fingerprint,
            "dependency_root": str(dependency_root),
            "prefix": str(prefix),
            "source": str(source),
            "checkpoint": str(checkpoint),
            "commands": planned_commands,
            "ownership_marker": str(ownership_marker),
        }

    dependency_root.mkdir(parents=True, exist_ok=True)
    ownership = {
        "schema_version": 1,
        "status": "bootstrapping",
        "fingerprint": bootstrap_fingerprint,
        "prefix": str(prefix),
        "source": str(source),
        "checkpoint": str(checkpoint),
        "started_at_unix": time.time(),
        "steps": [],
    }
    _write_json_atomic(ownership_marker, ownership)
    steps: list[dict[str, Any]] = []
    try:
        for planned in planned_commands:
            name = str(planned["name"])
            if name == "download_checkpoint":
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_name(
                    f".{checkpoint.name}.part-{os.getpid()}"
                )
                download_started = time.time()
                try:
                    with (
                        urllib.request.urlopen(
                            SAM2_TINY_CHECKPOINT_URL, timeout=120
                        ) as response,
                        temporary.open("wb") as output_handle,
                    ):
                        shutil.copyfileobj(response, output_handle)
                    downloaded_size = temporary.stat().st_size
                    downloaded_sha256 = _sha256_file(temporary)
                    if (
                        downloaded_size != SAM2_TINY_CHECKPOINT_SIZE
                        or downloaded_sha256 != SAM2_TINY_CHECKPOINT_SHA256
                    ):
                        raise StageExecutionError(
                            "Downloaded SAM2 checkpoint failed pinned artifact "
                            f"verification: size={downloaded_size}, "
                            f"sha256={downloaded_sha256}"
                        )
                    os.replace(temporary, checkpoint)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                steps.append(
                    {
                        "name": name,
                        "status": "completed",
                        "destination": str(checkpoint),
                        "size_bytes": checkpoint.stat().st_size,
                        "sha256": _sha256_file(checkpoint),
                        "elapsed_seconds": round(time.time() - download_started, 3),
                    }
                )
            else:
                steps.append(
                    _run_bootstrap_command(
                        name,
                        planned["argv"],
                        env_overlay=planned.get("env"),
                    )
                )
            ownership["steps"] = steps
            _write_json_atomic(ownership_marker, ownership)
    except Exception as exc:
        ownership["status"] = "failed"
        ownership["error"] = str(exc)
        ownership["error_type"] = type(exc).__name__
        ownership["steps"] = steps
        ownership["finished_at_unix"] = time.time()
        _write_json_atomic(ownership_marker, ownership)
        raise

    try:
        head, git_error = _git_head(source, str(git_executable))
        if git_error or head != SAM2_COMMIT:
            raise StageExecutionError(
                f"SAM2 checkout verification failed: {git_error or head}"
            )
        source_clean, source_clean_detail = _git_tracked_worktree_clean(
            source, str(git_executable)
        )
        if not source_clean:
            raise StageExecutionError(
                f"SAM2 tracked worktree verification failed: {source_clean_detail}"
            )
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != SAM2_TINY_CHECKPOINT_SIZE
            or _sha256_file(checkpoint) != SAM2_TINY_CHECKPOINT_SHA256
        ):
            raise StageExecutionError(
                f"SAM2 checkpoint verification failed: {checkpoint}"
            )
        sam2_environment = {"PYTHONPATH": _prepend_pythonpath(source)}
        sam2_import_ok, sam2_import_detail = _check_import(
            str(prefix_python), "sam2", sam2_environment
        )
        if not sam2_import_ok:
            raise StageExecutionError(
                f"SAM2 import verification failed: {sam2_import_detail}"
            )
        require_cuda = (
            str(_get(config, "tracking.sam2_device", default="auto"))
            .lower()
            .startswith("cuda")
        )
        runtime_ok, runtime_detail, runtime_versions = _check_sam2_runtime(
            str(prefix_python),
            sam2_environment,
            require_cuda=require_cuda,
        )
        if not runtime_ok:
            raise StageExecutionError(
                f"SAM2 runtime verification failed: {runtime_detail}"
            )
    except Exception as exc:
        ownership["status"] = "failed"
        ownership["error"] = str(exc)
        ownership["error_type"] = type(exc).__name__
        ownership["steps"] = steps
        ownership["finished_at_unix"] = time.time()
        _write_json_atomic(ownership_marker, ownership)
        raise
    ownership["status"] = "completed"
    ownership["steps"] = steps
    ownership["finished_at_unix"] = time.time()
    ownership["source_commit"] = head
    ownership["checkpoint_size_bytes"] = checkpoint.stat().st_size
    ownership["checkpoint_sha256"] = _sha256_file(checkpoint)
    ownership["runtime"] = runtime_versions
    _write_json_atomic(ownership_marker, ownership)
    return {
        "ok": True,
        "status": "completed",
        "fingerprint": bootstrap_fingerprint,
        "dependency_root": str(dependency_root),
        "prefix": str(prefix),
        "source": str(source),
        "source_commit": head,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ownership["checkpoint_sha256"],
        "runtime": runtime_versions,
        "commands": steps,
        "ownership_marker": str(ownership_marker),
    }


__all__ = [
    "CLIP_CHECKPOINT_SHA256",
    "GROUNDINGDINO_COMMIT",
    "GROUNDINGDINO_CHECKPOINT_SHA256",
    "GROUNDINGDINO_CONFIG_SHA256",
    "SAM2_COMMIT",
    "SAM2_TINY_CHECKPOINT_URL",
    "SAM2_TINY_CHECKPOINT_SHA256",
    "SAM2_TINY_CHECKPOINT_SIZE",
    "SAM2_TINY_MODEL_CFG",
    "VIDEO2MESH_COMMIT",
    "PreflightError",
    "StageCommand",
    "StageExecutionError",
    "UnsafeOutputPathError",
    "Video2MeshRunnerError",
    "bootstrap_sam2",
    "build_stage_commands",
    "compute_frame_window",
    "load_pipeline_config",
    "preflight_environment",
    "run_video2mesh_stages",
]
