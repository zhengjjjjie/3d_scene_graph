#!/usr/bin/env python3
"""Exec a pinned COLMAP binary after translating Video2Mesh CLI options.

Video2Mesh's pinned revision emits the COLMAP 3.x SIFT GPU option names.
COLMAP 4.x moved those switches to the FeatureExtraction/FeatureMatching
option groups.  The ConceptGraphs runner probes the installed binary and
passes the supported target names through the environment below.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Sequence


REAL_BINARY_ENV = "CONCEPTGRAPHS_V2M_COLMAP_REAL_BINARY"
REAL_SHA256_ENV = "CONCEPTGRAPHS_V2M_COLMAP_REAL_SHA256"
WRAPPER_SHA256_ENV = "CONCEPTGRAPHS_V2M_COLMAP_WRAPPER_SHA256"
EXTRACTION_OPTION_ENV = "CONCEPTGRAPHS_V2M_COLMAP_EXTRACTION_GPU_OPTION"
MATCHING_OPTION_ENV = "CONCEPTGRAPHS_V2M_COLMAP_MATCHING_GPU_OPTION"

LEGACY_EXTRACTION_OPTION = "--SiftExtraction.use_gpu"
MODERN_EXTRACTION_OPTION = "--FeatureExtraction.use_gpu"
LEGACY_MATCHING_OPTION = "--SiftMatching.use_gpu"
MODERN_MATCHING_OPTION = "--FeatureMatching.use_gpu"

_ALLOWED_EXTRACTION_OPTIONS = {
    LEGACY_EXTRACTION_OPTION,
    MODERN_EXTRACTION_OPTION,
}
_ALLOWED_MATCHING_OPTIONS = {
    LEGACY_MATCHING_OPTION,
    MODERN_MATCHING_OPTION,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_colmap_arguments(
    arguments: Sequence[str],
    *,
    extraction_option: str,
    matching_option: str,
) -> tuple[str, ...]:
    """Rewrite only the two exact legacy option-name tokens."""

    if extraction_option not in _ALLOWED_EXTRACTION_OPTIONS:
        raise ValueError(
            f"Unsupported extraction GPU option target: {extraction_option!r}"
        )
    if matching_option not in _ALLOWED_MATCHING_OPTIONS:
        raise ValueError(f"Unsupported matching GPU option target: {matching_option!r}")
    replacements = {
        LEGACY_EXTRACTION_OPTION: extraction_option,
        LEGACY_MATCHING_OPTION: matching_option,
    }
    return tuple(
        replacements.get(str(argument), str(argument)) for argument in arguments
    )


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    wrapper_path = Path(__file__).resolve()
    real_binary = Path(_required_environment(REAL_BINARY_ENV)).expanduser()
    if not real_binary.is_absolute():
        raise RuntimeError(f"Real COLMAP binary must be absolute: {real_binary}")
    real_binary = real_binary.resolve()
    if real_binary == wrapper_path:
        raise RuntimeError("Refusing recursive COLMAP compatibility wrapper execution")
    if not real_binary.is_file() or not os.access(real_binary, os.X_OK):
        raise RuntimeError(f"Real COLMAP binary is not executable: {real_binary}")

    expected_real_sha256 = _required_environment(REAL_SHA256_ENV)
    actual_real_sha256 = _sha256_file(real_binary)
    if actual_real_sha256 != expected_real_sha256:
        raise RuntimeError(
            "Real COLMAP binary SHA-256 changed after preflight: "
            f"expected={expected_real_sha256}, actual={actual_real_sha256}"
        )
    expected_wrapper_sha256 = _required_environment(WRAPPER_SHA256_ENV)
    actual_wrapper_sha256 = _sha256_file(wrapper_path)
    if actual_wrapper_sha256 != expected_wrapper_sha256:
        raise RuntimeError(
            "COLMAP compatibility wrapper SHA-256 changed after preflight: "
            f"expected={expected_wrapper_sha256}, actual={actual_wrapper_sha256}"
        )

    extraction_option = _required_environment(EXTRACTION_OPTION_ENV)
    matching_option = _required_environment(MATCHING_OPTION_ENV)
    source_arguments = tuple(arguments if arguments is not None else sys.argv[1:])
    rewritten = rewrite_colmap_arguments(
        source_arguments,
        extraction_option=extraction_option,
        matching_option=matching_option,
    )
    if rewritten != source_arguments:
        translations = [
            f"{source} -> {target}"
            for source, target in (
                (LEGACY_EXTRACTION_OPTION, extraction_option),
                (LEGACY_MATCHING_OPTION, matching_option),
            )
            if source in source_arguments and source != target
        ]
        print(
            "[conceptgraphs-colmap-compat] " + ", ".join(translations),
            file=sys.stderr,
            flush=True,
        )
    os.execv(str(real_binary), [str(real_binary), *rewritten])
    return 127  # pragma: no cover - os.execv replaces the process on success.


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"conceptgraphs-colmap-compat: {exc}", file=sys.stderr)
        raise SystemExit(127) from exc
