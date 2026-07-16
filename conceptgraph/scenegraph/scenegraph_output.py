"""Generic attribute extraction and sparse scene-graph serialization.

This module intentionally does not depend on ``GPTPrompt``.  Attribute semantics
come entirely from a user-provided prompt, while this file only enforces the
machine-readable ``property``/``state`` contract and normalizes labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import pprint
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from conceptgraph.scenegraph.build_scenegraph_cfslam import (
    OPENAI_API_KEY_FILE,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    make_openai_client,
    parse_json_object_text,
    request_openai_text,
    save_bytes_atomic,
    save_json_atomic,
    sha256_file,
    validate_openai_base_url,
)


SCHEMA_VERSION = 1
ATTRIBUTE_FIELDS = ("property", "state")
RELATION_MAP = {
    "a on b": (0, "ON", 1),
    "b on a": (1, "ON", 0),
    "a in b": (0, "INSIDE", 1),
    "b in a": (1, "INSIDE", 0),
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value independently of dictionary formatting."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from None


def validate_identifier(value: Any, context: str) -> int:
    """Validate the original, non-reindexed ConceptGraphs object ID."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def load_nodes(path: Path) -> list[dict[str, Any]]:
    nodes = load_json(path)
    if not isinstance(nodes, list):
        raise ValueError(f"{path} must contain a JSON list of nodes")

    seen_ids: set[Any] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a JSON object")
        if "id" not in node:
            raise ValueError(f"Node {index} is missing id")
        node_id = validate_identifier(node["id"], f"Node {index} id")
        if node_id in seen_ids:
            raise ValueError(f"Duplicate node id: {node_id!r}")
        seen_ids.add(node_id)
    return nodes


def load_captions_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    captions = load_json(path)
    if not isinstance(captions, list):
        raise ValueError(f"{path} must contain a JSON list of caption entries")

    by_id: dict[Any, dict[str, Any]] = {}
    for index, entry in enumerate(captions):
        if not isinstance(entry, dict) or "id" not in entry:
            raise ValueError(f"Caption entry {index} must be an object with id")
        object_id = validate_identifier(entry["id"], f"Caption entry {index} id")
        if object_id in by_id:
            raise ValueError(f"Duplicate caption id: {object_id!r}")
        values = entry.get("captions")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Caption entry {object_id!r} captions must be list[str]")
        by_id[object_id] = entry
    return by_id


def normalize_token(value: str, *, uppercase: bool) -> str:
    """Normalize an unconstrained model label without assigning semantics."""
    if not isinstance(value, str):
        raise ValueError("Attribute labels must be strings")
    token = unicodedata.normalize("NFKC", value.strip())
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", token)
    token = re.sub(r"[^\w]+", "_", token, flags=re.UNICODE)
    token = re.sub(r"_+", "_", token).strip("_")
    token = token.upper() if uppercase else token.lower()
    if not token:
        raise ValueError(f"Label becomes empty after normalization: {value!r}")
    return token


def normalize_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Model response field {field!r} must be list[str]")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = normalize_token(item, uppercase=True)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def parse_attribute_response(content: str) -> dict[str, list[str]]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned an empty attribute response")
    try:
        parsed = parse_json_object_text(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("Model response does not contain a valid JSON object") from None
    if not isinstance(parsed, dict) or set(parsed) != set(ATTRIBUTE_FIELDS):
        raise ValueError("Model response must contain exactly property and state")
    return {
        field: normalize_string_list(parsed[field], field)
        for field in ATTRIBUTE_FIELDS
    }


def cache_filename(index: int, object_id: Any) -> str:
    id_hash = canonical_sha256({"id": object_id})[:16]
    return f"{index:06d}_{id_hash}.json"


def load_attribute_cache(path: Path, request: dict[str, Any]) -> dict[str, list[str]] | None:
    if not path.is_file():
        return None
    try:
        cached = load_json(path)
        if not isinstance(cached, dict) or cached.get("request") != request:
            return None
        result = cached.get("result")
        if not isinstance(result, dict) or set(result) != set(ATTRIBUTE_FIELDS):
            return None
        # Revalidate instead of trusting a manually edited cache.
        return {
            field: normalize_string_list(result[field], field)
            for field in ATTRIBUTE_FIELDS
        }
    except (OSError, ValueError):
        return None


def attribute_manifest(
    *,
    state: str,
    identity: dict[str, Any],
    expected_ids: list[Any],
    completed_ids: list[Any],
    output_file: Path,
    api_requests: int,
    cache_hits: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "complete": state == "complete",
        "identity": identity,
        "expected_ids": expected_ids,
        "completed_ids": completed_ids,
        "api_requests_this_run": api_requests,
        "cache_hits_this_run": cache_hits,
        "output_file": str(output_file.resolve()),
    }


def extract_attributes(args: argparse.Namespace) -> None:
    base_url = validate_openai_base_url(args.openai_base_url)
    model = args.model.strip()
    if not model:
        raise ValueError("OpenAI model must not be empty")

    nodes_path = Path(args.nodes_file).resolve()
    captions_path = Path(args.captions_file).resolve()
    prompt_path = Path(args.prompt_file).resolve()
    output_path = Path(args.output_file).resolve()
    manifest_path = Path(args.manifest_file).resolve()
    cache_dir = ensure_private_dir(Path(args.cache_dir).resolve())

    nodes = load_nodes(nodes_path)
    captions_by_id = load_captions_by_id(captions_path)
    node_ids = [node["id"] for node in nodes]
    node_id_set = set(node_ids)
    if not node_id_set.issubset(captions_by_id):
        missing = [node_id for node_id in node_ids if node_id not in captions_by_id]
        raise ValueError(f"Captions are missing retained node IDs: {missing}")

    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("Attribute prompt file must not be empty")

    identity = {
        "nodes": {"path": str(nodes_path), "sha256": sha256_file(nodes_path)},
        "captions": {"path": str(captions_path), "sha256": sha256_file(captions_path)},
        "prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
        "provider": {"base_url": base_url, "model": model},
    }
    completed_ids: list[Any] = []
    api_requests = 0
    cache_hits = 0
    save_json_atomic(
        attribute_manifest(
            state="running",
            identity=identity,
            expected_ids=node_ids,
            completed_ids=completed_ids,
            output_file=output_path,
            api_requests=api_requests,
            cache_hits=cache_hits,
        ),
        manifest_path,
    )

    client = None
    attributes: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        object_id = node["id"]
        input_value = {
            "node": node,
            "caption_entry": captions_by_id[object_id],
        }
        request = {
            "schema_version": SCHEMA_VERSION,
            "base_url": base_url,
            "model": model,
            "prompt_sha256": identity["prompt"]["sha256"],
            "input_sha256": canonical_sha256(input_value),
            "input": input_value,
        }
        cache_path = cache_dir / cache_filename(index, object_id)
        result = None if args.force else load_attribute_cache(cache_path, request)
        if result is None:
            if client is None:
                client = make_openai_client(
                    api_key_file=args.openai_api_key_file,
                    base_url=base_url,
                    max_retries=args.openai_max_retries,
                )
            response = request_openai_text(
                client,
                [
                    {
                        "role": "user",
                        "content": prompt.rstrip()
                        + "\n\nINPUT JSON:\n"
                        + json.dumps(input_value, ensure_ascii=False, sort_keys=True),
                    }
                ],
                timeout=args.timeout,
                model=model,
            )
            attribute_error = None
            try:
                result = parse_attribute_response(response)
            except ValueError as exc:
                attribute_error = type(exc).__name__
            if attribute_error is not None:
                raise RuntimeError(
                    f"Invalid attribute response for node id {object_id!r} "
                    f"({attribute_error}); response body omitted"
                )
            save_json_atomic(
                {"request": request, "result": result, "raw_response": response},
                cache_path,
            )
            api_requests += 1
        else:
            cache_hits += 1

        attributes.append({"id": object_id, **result})
        completed_ids.append(object_id)
        save_json_atomic(
            attribute_manifest(
                state="running",
                identity=identity,
                expected_ids=node_ids,
                completed_ids=completed_ids,
                output_file=output_path,
                api_requests=api_requests,
                cache_hits=cache_hits,
            ),
            manifest_path,
        )

    save_json_atomic(attributes, output_path)
    manifest = attribute_manifest(
        state="complete",
        identity=identity,
        expected_ids=node_ids,
        completed_ids=completed_ids,
        output_file=output_path,
        api_requests=api_requests,
        cache_hits=cache_hits,
    )
    manifest["output_sha256"] = sha256_file(output_path)
    save_json_atomic(manifest, manifest_path)
    print(
        f"Saved attributes for {len(attributes)} nodes "
        f"({api_requests} API requests, {cache_hits} cache hits)."
    )


def load_attributes_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list of attributes")
    by_id: dict[Any, dict[str, Any]] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != {"id", *ATTRIBUTE_FIELDS}:
            raise ValueError(
                f"Attribute entry {index} must contain exactly id, property, and state"
            )
        object_id = validate_identifier(entry["id"], f"Attribute entry {index} id")
        if object_id in by_id:
            raise ValueError(f"Duplicate attribute id: {object_id!r}")
        by_id[object_id] = {
            field: normalize_string_list(entry[field], field)
            for field in ATTRIBUTE_FIELDS
        }
    return by_id


def object_key(node: dict[str, Any]) -> str:
    object_tag = node.get("object_tag")
    if not isinstance(object_tag, str):
        raise ValueError(f"Node id {node['id']!r} has no string object_tag")
    if object_tag.strip().lower() in {"fail", "invalid"}:
        raise ValueError(f"Node id {node['id']!r} has an invalid object_tag")
    normalized_tag = normalize_token(object_tag, uppercase=False)
    return f"{normalized_tag}_{node['id']}"


def load_edges(path: Path) -> list[tuple[Any, Any, str]]:
    try:
        with path.open("rb") as file:
            value = pickle.load(file)
    except (OSError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot load edge pickle {path}: {exc}") from None
    if not isinstance(value, list):
        raise ValueError("Edge pickle must contain a list")
    edges: list[tuple[Any, Any, str]] = []
    for index, edge in enumerate(value):
        if not isinstance(edge, (tuple, list)) or len(edge) != 3:
            raise ValueError(f"Edge {index} must be a three-item tuple/list")
        source_id, target_id, relation = edge
        validate_identifier(source_id, f"Edge {index} first id")
        validate_identifier(target_id, f"Edge {index} second id")
        if not isinstance(relation, str) or relation.strip().lower() not in RELATION_MAP:
            raise ValueError(f"Edge {index} has unsupported relation: {relation!r}")
        edges.append((source_id, target_id, relation.strip().lower()))
    return edges


def validate_relation_targets(graph: dict[str, dict[str, list[str]]]) -> None:
    for source_key, fields in graph.items():
        relations = fields.get("relation", [])
        if not isinstance(relations, list) or not all(isinstance(item, str) for item in relations):
            raise ValueError(f"Relations for {source_key!r} must be list[str]")
        for relation in relations:
            try:
                predicate, target_key = relation.split(" ", 1)
            except ValueError:
                raise ValueError(f"Malformed relation on {source_key!r}: {relation!r}") from None
            if predicate not in {"ON", "INSIDE"}:
                raise ValueError(f"Unsupported output predicate: {predicate!r}")
            if target_key not in graph:
                raise ValueError(
                    f"Relation target {target_key!r} from {source_key!r} is not a node"
                )
            if target_key == source_key:
                raise ValueError(f"Self-relation is not allowed for {source_key!r}")


def format_scenegraph(args: argparse.Namespace) -> None:
    nodes_path = Path(args.nodes_file).resolve()
    attributes_path = Path(args.attributes_file).resolve()
    edges_path = Path(args.edges_file).resolve()
    output_json_path = Path(args.output_json).resolve()
    output_repr_path = Path(args.output_repr).resolve()
    manifest_path = Path(args.manifest_file).resolve()

    identity = {
        "nodes": {"path": str(nodes_path), "sha256": sha256_file(nodes_path)},
        "attributes": {
            "path": str(attributes_path),
            "sha256": sha256_file(attributes_path),
        },
        "edges": {"path": str(edges_path), "sha256": sha256_file(edges_path)},
    }
    save_json_atomic(
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "complete": False,
            "identity": identity,
        },
        manifest_path,
    )

    nodes = load_nodes(nodes_path)
    attributes_by_id = load_attributes_by_id(attributes_path)
    node_ids = [node["id"] for node in nodes]
    if set(attributes_by_id) != set(node_ids):
        missing = [node_id for node_id in node_ids if node_id not in attributes_by_id]
        extra = [object_id for object_id in attributes_by_id if object_id not in set(node_ids)]
        raise ValueError(f"Node/attribute ID mismatch; missing={missing}, extra={extra}")

    key_by_id: dict[Any, str] = {}
    graph: dict[str, dict[str, list[str]]] = {}
    for node in nodes:
        node_id = node["id"]
        key = object_key(node)
        if key in graph:
            raise ValueError(f"Duplicate normalized object key: {key!r}")
        key_by_id[node_id] = key
        attributes = attributes_by_id[node_id]
        # Sparse means empty fields are omitted; every source node still exists.
        fields = {
            field: attributes[field]
            for field in ATTRIBUTE_FIELDS
            if attributes[field]
        }
        graph[key] = fields

    edges = load_edges(edges_path)
    for edge_index, (first_id, second_id, relation) in enumerate(edges):
        if first_id not in key_by_id or second_id not in key_by_id:
            raise ValueError(
                f"Edge {edge_index} references missing node id(s): "
                f"{first_id!r}, {second_id!r}"
            )
        source_slot, predicate, target_slot = RELATION_MAP[relation]
        endpoint_ids = (first_id, second_id)
        source_key = key_by_id[endpoint_ids[source_slot]]
        target_key = key_by_id[endpoint_ids[target_slot]]
        output_relation = f"{predicate} {target_key}"
        relation_list = graph[source_key].setdefault("relation", [])
        if output_relation not in relation_list:
            relation_list.append(output_relation)

    validate_relation_targets(graph)
    save_json_atomic(graph, output_json_path)
    repr_payload = pprint.pformat(graph, sort_dicts=False, width=120) + "\n"
    save_bytes_atomic(repr_payload.encode("utf-8"), output_repr_path)

    # Read both files back: JSON must be legal, repr must be a complete textual
    # representation rather than a log fragment.
    reloaded_json = load_json(output_json_path)
    if reloaded_json != graph:
        raise RuntimeError("Scene-graph JSON changed during serialization")
    import ast

    with output_repr_path.open("r", encoding="utf-8") as file:
        reloaded_repr = ast.literal_eval(file.read())
    if reloaded_repr != graph:
        raise RuntimeError("Scene-graph Python repr changed during serialization")
    validate_relation_targets(reloaded_json)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "complete": True,
        "identity": identity,
        "node_count": len(graph),
        "input_edge_count": len(edges),
        "output_relation_count": sum(
            len(fields.get("relation", [])) for fields in graph.values()
        ),
        "outputs": {
            "json": {
                "path": str(output_json_path),
                "sha256": sha256_file(output_json_path),
            },
            "python_repr": {
                "path": str(output_repr_path),
                "sha256": sha256_file(output_repr_path),
            },
        },
    }
    save_json_atomic(manifest, manifest_path)
    print(
        f"Saved sparse scene graph with {len(graph)} nodes and "
        f"{manifest['output_relation_count']} relations."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract generic attributes and serialize a sparse ConceptGraphs graph."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract-attributes",
        help="Use a user-owned prompt and OpenAI Responses to extract property/state lists.",
    )
    extract.add_argument("--nodes-file", default="scene_graph_nodes.json")
    extract.add_argument("--captions-file", default="cfslam_openai_captions.json")
    extract.add_argument("--prompt-file", required=True)
    extract.add_argument("--output-file", default="scene_graph_attributes.json")
    extract.add_argument("--cache-dir", default="scene_graph_attribute_cache")
    extract.add_argument(
        "--manifest-file",
        default="scene_graph_attributes_manifest.json",
    )
    extract.add_argument(
        "--openai-api-key-file",
        default=OPENAI_API_KEY_FILE,
        help="Read the credential from this private file; falls back to OPENAI_API_KEY.",
    )
    extract.add_argument("--openai-base-url", default=OPENAI_BASE_URL)
    extract.add_argument("--model", "--openai-model", dest="model", default=OPENAI_MODEL)
    extract.add_argument(
        "--timeout",
        "--openai-timeout",
        dest="timeout",
        type=float,
        default=OPENAI_TIMEOUT,
    )
    extract.add_argument(
        "--openai-max-retries",
        type=int,
        default=OPENAI_MAX_RETRIES,
    )
    extract.add_argument(
        "--force",
        action="store_true",
        help="Ignore compatible per-node caches and request every node again.",
    )
    extract.set_defaults(handler=extract_attributes)

    formatter = subparsers.add_parser(
        "format",
        help="Combine nodes, generic attributes, and ConceptGraphs structural edges.",
    )
    formatter.add_argument("--nodes-file", default="scene_graph_nodes.json")
    formatter.add_argument("--attributes-file", default="scene_graph_attributes.json")
    formatter.add_argument("--edges-file", default="cfslam_scenegraph_edges.pkl")
    formatter.add_argument("--output-json", default="scene_graph.json")
    formatter.add_argument("--output-repr", default="scene_graph.txt")
    formatter.add_argument("--manifest-file", default="scene_graph_manifest.json")
    formatter.set_defaults(handler=format_scenegraph)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "timeout") and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        parser.error("--timeout must be positive")
    if hasattr(args, "openai_max_retries") and args.openai_max_retries < 0:
        parser.error("--openai-max-retries must be non-negative")
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
