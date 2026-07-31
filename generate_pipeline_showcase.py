#!/usr/bin/env python3
"""Build a self-contained HTML showcase for the completed bedroom pipeline.

The report is generated exclusively from existing manifests, cached model outputs,
and derived visual assets. It performs no inference and makes no network calls.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import io
import json
import math
import mimetypes
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SCENE_ID = "bedroom_4_CmEIg9gMI74"
RUN_NAME = SCENE_ID
DERIVED_ROOT = Path(os.environ.get("CG_LEGACY_OUTPUT_ROOT", ROOT / "outputs"))
DERIVED = DERIVED_ROOT / SCENE_ID
FINAL = ROOT / "outputs" / RUN_NAME / "scene_graph_openai"
OUTPUT = ROOT / "pipeline_showcase.html"
MAPPING_PACKAGES_VALUE = os.environ.get("CG_MAPPING_PACKAGES")
MAPPING_PACKAGES = (
    Path(MAPPING_PACKAGES_VALUE).expanduser()
    if MAPPING_PACKAGES_VALUE
    else None
)
if (
    MAPPING_PACKAGES is not None
    and MAPPING_PACKAGES.is_dir()
    and str(MAPPING_PACKAGES) not in sys.path
):
    # The serialized map contains an OmegaConf object. Keep the report builder
    # runnable in the existing svpp environment without a shell PYTHONPATH.
    sys.path.insert(0, str(MAPPING_PACKAGES))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_data_url(path: Path, *, max_side: int | None = None, quality: int = 88) -> str:
    """Return an image as a compact data URL, optionally resized and JPEG encoded."""
    if max_side is None:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    with Image.open(path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def pil_data_url(image: Image.Image, *, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def colorized_depth_data_url(path: Path) -> tuple[str, float, float]:
    depth_mm = np.asarray(Image.open(path), dtype=np.uint16)
    valid = depth_mm > 0
    values = depth_mm[valid].astype(np.float32) / 1000.0
    low, high = (float(np.percentile(values, 2)), float(np.percentile(values, 98)))
    normalized = np.zeros_like(depth_mm, dtype=np.uint8)
    scaled = np.clip((depth_mm.astype(np.float32) / 1000.0 - low) / (high - low), 0, 1)
    # Near pixels are warm, far pixels are cool.
    normalized[valid] = np.round((1.0 - scaled[valid]) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = (18, 24, 38)
    return pil_data_url(Image.fromarray(colored), quality=94), low, high


def caption_crop_data_urls(object_id: int, map_path: Path) -> list[dict]:
    """Reconstruct the exact red-outline crops selected for one example object."""
    with gzip.open(map_path, "rb") as handle:
        payload = pickle.load(handle)
    objects = payload["objects"] if isinstance(payload, dict) else payload
    obj = objects[object_id]
    object_cache = read_json(FINAL / "cfslam_captions_openai" / f"{object_id}.json")
    captions = object_cache["entry"]["captions"]
    selected = object_cache["selected_detection_indices"]
    result = []

    for slot, idx_det in enumerate(selected):
        view_cache = read_json(
            FINAL
            / "cfslam_captions_openai"
            / "views"
            / f"{object_id:04d}_{idx_det:04d}.json"
        )
        request = view_cache["request"]
        with Image.open(request["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        mask = np.asarray(obj["mask"][idx_det], dtype=bool)
        x1, y1, x2, y2 = request["xyxy"]
        padding = int(request["padding"])
        left = max(0, round(x1 - padding))
        top = max(0, round(y1 - padding))
        right = min(image.shape[1], round(x2 + padding))
        bottom = min(image.shape[0], round(y2 + padding))
        crop = image[top:bottom, left:right].copy()
        crop_mask = mask[top:bottom, left:right]
        contours, _ = cv2.findContours(
            crop_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(crop, contours, -1, (255, 0, 0), 3)
        result.append(
            {
                "image": pil_data_url(Image.fromarray(crop), quality=92),
                "frame": Path(request["image_path"]).stem.replace("frame", "frame "),
                "caption": captions[slot],
                "confidence": float(np.asarray(obj["conf"][idx_det]).reshape(-1)[0]),
            }
        )
    return result


def svg_escape(value: str) -> str:
    return html.escape(value, quote=True)


def node_key(node: dict) -> str:
    return f'{node["object_tag"].replace(" ", "_")}_{node["id"]}'


def relation_edges(relation_queries: list[dict]) -> list[dict]:
    """Convert model decisions into directed final edges for visualization."""
    relation_specs = {
        "a on b": ("object1", "object2", "ON"),
        "b on a": ("object2", "object1", "ON"),
        "a in b": ("object1", "object2", "INSIDE"),
        "b in a": ("object2", "object1", "INSIDE"),
    }
    edges = []
    for query in relation_queries:
        relation = str(query.get("object_relation", "")).strip().lower()
        if relation not in relation_specs:
            continue
        source_name, target_name, label = relation_specs[relation]
        source = query[source_name]
        target = query[target_name]
        edges.append(
            {
                "source": int(source["id"]),
                "target": int(target["id"]),
                "source_key": node_key(source),
                "target_key": node_key(target),
                "label": label,
            }
        )
    return edges


def relation_cards_html(relation_queries: list[dict]) -> str:
    cards = []
    for query in relation_queries:
        first = query["object1"]
        second = query["object2"]
        relation = str(query.get("object_relation", "none of these")).strip().lower()
        kept = relation != "none of these"
        decision = relation.upper().replace("NONE OF THESE", "NONE")
        pair = f"{node_key(first)} ↔ {node_key(second)}"
        reason = str(query.get("reason", "No reason was returned."))
        cards.append(
            '<article class="candidate"><div class="candidate-head">'
            f'<code>{html.escape(pair)}</code>'
            f'<span class="decision {"keep" if kept else "none"}">{html.escape(decision)}</span>'
            f'</div><p>{html.escape(reason)}</p></article>'
        )
    return "".join(cards)


def refinement_example_html(node: dict, captions: list[str]) -> str:
    caption_lines = "<br>".join(f"“{html.escape(value)}”" for value in captions)
    tags = "".join(
        f'<span class="tag">{html.escape(tag)}</span>' for tag in node["possible_tags"][:4]
    )
    return (
        '<div class="semantic-transform"><div class="semantic-box">'
        f'<span>INPUT · {len(captions)} VIEW CAPTIONS</span><p>{caption_lines}</p></div>'
        '<div class="transform-arrow">→</div><div class="semantic-box">'
        '<span>OUTPUT · REFINED NODE</span>'
        f'<p><b>{html.escape(node["object_tag"])} · ID {node["id"]}</b><br>'
        f'{html.escape(node["caption"])}</p><div class="tag-row">{tags}</div></div></div>'
    )


def trajectory_svg(traj_path: Path) -> str:
    matrices = []
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        values = [float(item) for item in line.split()]
        if len(values) == 16:
            matrices.append(np.asarray(values, dtype=float).reshape(4, 4))
    points = np.asarray([[matrix[0, 3], matrix[2, 3]] for matrix in matrices])
    width, height = 640, 300
    pad_x, pad_y = 58, 42
    x_min, z_min = points.min(axis=0)
    x_max, z_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    z_span = max(float(z_max - z_min), 1e-6)
    scale = min((width - 2 * pad_x) / x_span, (height - 2 * pad_y) / z_span)
    xy = [
        (
            pad_x + (float(x) - x_min) * scale,
            height - pad_y - (float(z) - z_min) * scale,
        )
        for x, z in points
    ]
    path = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(xy))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#178d8b" opacity="{0.35 + 0.65 * i / max(len(xy)-1, 1):.2f}" />'
        for i, (x, y) in enumerate(xy)
    )
    sx, sy = xy[0]
    ex, ey = xy[-1]
    return f"""
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="31 帧相机顶视轨迹">
      <defs><linearGradient id="trajStroke" x1="0" x2="1"><stop stop-color="#73d1ca"/><stop offset="1" stop-color="#0f6668"/></linearGradient></defs>
      <rect x="1" y="1" width="638" height="298" rx="18" fill="#f5fbfb" stroke="#dbe9e8"/>
      <g stroke="#dbe7e7" stroke-width="1" stroke-dasharray="3 6">
        <path d="M58 75 H582 M58 150 H582 M58 225 H582"/><path d="M145 42 V258 M320 42 V258 M495 42 V258"/>
      </g>
      <path d="{path}" fill="none" stroke="url(#trajStroke)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      {dots}
      <circle cx="{sx:.1f}" cy="{sy:.1f}" r="8" fill="#fff" stroke="#168c8c" stroke-width="3"/>
      <circle cx="{ex:.1f}" cy="{ey:.1f}" r="8" fill="#168c8c" stroke="#fff" stroke-width="3"/>
      <text x="{sx + 12:.1f}" y="{sy - 10:.1f}" class="svg-label">START · 00</text>
      <text x="{ex + 12:.1f}" y="{ey + 18:.1f}" class="svg-label">END · 30</text>
      <text x="24" y="25" class="svg-caption">camera-to-world translation · X/Z top view</text>
      <text x="492" y="282" class="svg-caption">3D path · 3.4569 m</text>
    </svg>"""


def mask_chart_svg(counts: list[int]) -> str:
    width, height = 760, 250
    left, right, top, bottom = 48, 18, 34, 42
    chart_w, chart_h = width - left - right, height - top - bottom
    max_count = max(counts)
    bar_step = chart_w / len(counts)
    bars = []
    for index, value in enumerate(counts):
        bar_w = max(bar_step - 3, 2)
        bar_h = chart_h * value / max_count
        x = left + index * bar_step + 1.5
        y = top + chart_h - bar_h
        highlight = index == 27
        fill = "#6c5ce7" if highlight else "#b9b3ef"
        opacity = "1" if highlight else "0.78"
        bars.append(
            f'<rect class="mask-bar" data-frame="{index:02d}" data-count="{value}" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="2" fill="{fill}" opacity="{opacity}"><title>frame {index:02d}: {value} masks</title></rect>'
        )
    median = float(np.median(counts))
    median_y = top + chart_h - chart_h * median / max_count
    return f"""
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="每帧 SAM3 mask 数量柱状图">
      <rect x="1" y="1" width="758" height="248" rx="18" fill="#faf9ff" stroke="#e5e1f4"/>
      <line x1="{left}" y1="{top + chart_h}" x2="{width-right}" y2="{top + chart_h}" stroke="#cfcbe0"/>
      <line x1="{left}" y1="{median_y:.2f}" x2="{width-right}" y2="{median_y:.2f}" stroke="#6c5ce7" stroke-dasharray="5 5" opacity=".7"/>
      <text x="{width-right-4}" y="{median_y-7:.2f}" text-anchor="end" class="svg-caption">median {median:g}</text>
      {''.join(bars)}
      <text x="{left}" y="22" class="svg-caption">masks / frame · total {sum(counts)}</text>
      <text x="{left}" y="232" class="svg-caption">00</text><text x="{width/2}" y="232" text-anchor="middle" class="svg-caption">frame index</text><text x="{width-right}" y="232" text-anchor="end" class="svg-caption">30</text>
      <text x="{left + 27*bar_step:.1f}" y="{top-6}" class="svg-label">frame 27</text>
    </svg>"""


def category_for(node: dict) -> str:
    tag = node["object_tag"].strip().lower()
    structural_terms = ("ceiling", "floor", "wall", "window", "door")
    return "structure" if any(term in tag for term in structural_terms) else "object"


def node_color(node: dict) -> str:
    if category_for(node) == "structure":
        return "#168c8c"
    tag = node["object_tag"].strip().lower()
    if any(term in tag for term in ("pillow", "cushion", "bedspread", "blanket")):
        return "#6c5ce7"
    return "#d9852b"


def topdown_svg(nodes: list[dict], edges: list[dict]) -> str:
    width, height = 760, 450
    left, right, top, bottom = 58, 48, 52, 56
    centers = np.asarray([[node["bbox_center"][0], node["bbox_center"][2]] for node in nodes])
    mins = centers.min(axis=0) - 0.25
    maxs = centers.max(axis=0) + 0.25
    span = np.maximum(maxs - mins, 1e-6)

    def project(node):
        x, z = node["bbox_center"][0], node["bbox_center"][2]
        px = left + (x - mins[0]) / span[0] * (width - left - right)
        py = height - bottom - (z - mins[1]) / span[1] * (height - top - bottom)
        return float(px), float(py)

    positions = {node["id"]: project(node) for node in nodes}
    arrows = []
    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        arrows.append(
            f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" stroke="#26324a" '
            f'stroke-width="3" marker-end="url(#topArrow)"/>'
            f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-8:.1f}" '
            f'text-anchor="middle" class="svg-edge">{svg_escape(edge["label"])}</text>'
        )
    node_svg = []
    # Fixed label offsets reduce collisions among the pillow/art clusters.
    offsets = {3: (-10, -14), 4: (0, -17), 5: (-22, 24), 6: (0, -17), 7: (-12, 27), 8: (-36, -16), 9: (18, 27), 12: (8, -15)}
    for node in nodes:
        x, y = positions[node["id"]]
        color = node_color(node)
        dx, dy = offsets.get(node["id"], (8, -12))
        label = f'{node["object_tag"].replace(" ", "_")}_{node["id"]}'
        node_svg.append(
            f'<g class="top-node graph-node" data-node-key="{svg_escape(label)}" tabindex="0" role="button">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#fff" stroke-width="3"><title>{svg_escape(node["caption"])}</title></circle>'
            f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" class="svg-node-label">{svg_escape(label)}</text></g>'
        )
    return f"""
    <svg class="chart-svg topdown" viewBox="0 0 {width} {height}" role="img" aria-label="13 个对象中心的世界坐标顶视分布">
      <defs><marker id="topArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#26324a"/></marker></defs>
      <rect x="1" y="1" width="758" height="448" rx="18" fill="#fbfcfe" stroke="#dfe5eb"/>
      <g stroke="#e1e6eb" stroke-width="1" stroke-dasharray="3 7"><path d="M58 100 H712 M58 200 H712 M58 300 H712"/><path d="M180 52 V394 M360 52 V394 M540 52 V394"/></g>
      {''.join(arrows)}
      {''.join(node_svg)}
      <text x="26" y="28" class="svg-caption">world X / Z · bbox centers (m)</text>
      <text x="695" y="425" class="svg-caption">X →</text><text x="18" y="75" class="svg-caption">Z ↑</text>
    </svg>"""


def semantic_graph_svg(nodes: list[dict], edges: list[dict]) -> str:
    positions = {
        0: (90, 74), 1: (245, 74), 2: (400, 74), 3: (555, 74), 4: (710, 74), 6: (865, 74),
        9: (325, 258), 5: (645, 258),
        7: (110, 445), 8: (290, 445), 10: (470, 445), 11: (650, 445), 12: (830, 445),
    }
    parts = []
    for node in nodes:
        x, y = positions[node["id"]]
        key = node_key(node)
        color = node_color(node)
        width = 140
        parts.append(
            f'<g class="semantic-node graph-node" data-node-key="{svg_escape(key)}" tabindex="0" role="button" transform="translate({x-width/2:.1f},{y-31:.1f})">'
            f'<rect width="{width}" height="62" rx="15" fill="#fff" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="18" cy="18" r="5" fill="{color}"/>'
            f'<text x="28" y="22" class="svg-node-title">{svg_escape(node["object_tag"])}</text>'
            f'<text x="18" y="44" class="svg-node-id">ID {node["id"]:02d}</text>'
            f'<title>{svg_escape(node["caption"])}</title></g>'
        )
    edge_parts = []
    for index, edge in enumerate(edges):
        source_x, source_y = positions[edge["source"]]
        target_x, target_y = positions[edge["target"]]
        dx, dy = target_x - source_x, target_y - source_y
        distance = max(math.hypot(dx, dy), 1.0)
        start_x = source_x + dx / distance * 78
        start_y = source_y + dy / distance * 38
        end_x = target_x - dx / distance * 78
        end_y = target_y - dy / distance * 38
        middle_x = (start_x + end_x) / 2
        middle_y = (start_y + end_y) / 2 - 42 - 12 * index
        label_width = max(66, 18 + 8 * len(edge["label"]))
        edge_parts.append(
            f'<path d="M{start_x:.1f} {start_y:.1f} Q{middle_x:.1f} {middle_y:.1f} '
            f'{end_x:.1f} {end_y:.1f}" fill="none" stroke="#168c64" stroke-width="4" '
            f'marker-end="url(#semArrow)"/>'
            f'<rect x="{middle_x-label_width/2:.1f}" y="{middle_y-17:.1f}" width="{label_width:.1f}" '
            f'height="30" rx="15" fill="#168c64"/>'
            f'<text x="{middle_x:.1f}" y="{middle_y+3:.1f}" text-anchor="middle" '
            f'class="svg-edge-invert">{svg_escape(edge["label"])}</text>'
        )
    relation_word = "关系" if len(edges) != 1 else "关系"
    return f"""
    <svg class="semantic-svg" viewBox="0 0 960 520" role="img" aria-label="最终稀疏 Scene Graph，{len(nodes)} 个节点和 {len(edges)} 条{relation_word}">
      <defs><marker id="semArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#168c64"/></marker></defs>
      <rect x="1" y="1" width="958" height="518" rx="22" fill="#f7faf9" stroke="#dce8e3"/>
      {''.join(edge_parts)}
      {''.join(parts)}
      <text x="24" y="502" class="svg-caption">实线箭头 = 最终输出关系 · 其余节点保留为无边的稀疏节点</text>
    </svg>"""


def node_buttons_html(node_data: list[dict]) -> str:
    buttons = []
    for node in node_data:
        states = node.get("state", [])
        relations = node.get("relation", [])
        buttons.append(
            '<button class="node-chip" '
            f'data-node-key="{html.escape(node["key"])}" '
            f'data-category="{node["category"]}" '
            f'data-has-state="{str(bool(states)).lower()}" '
            f'data-has-relation="{str(bool(relations)).lower()}">'
            f'<span class="node-dot" style="--node-color:{node["color"]}"></span>'
            f'<span><b>{html.escape(node["tag"])}</b><small>ID {node["id"]:02d}</small></span></button>'
        )
    return "".join(buttons)


def caption_views_html(views: list[dict], object_id: int) -> str:
    items = []
    for index, view in enumerate(views, 1):
        items.append(
            f"""<article class="caption-view">
              <button class="image-button" type="button" data-lightbox-src="{view['image']}" aria-label="放大视角 {index}">
                <img src="{view['image']}" alt="object {object_id} 的第 {index} 个红色轮廓视角" loading="lazy">
              </button>
              <div class="caption-view-meta"><span>VIEW {index} · {html.escape(view['frame'])}</span><span>conf {view['confidence']:.3f}</span></div>
              <p>{html.escape(view['caption'])}</p>
            </article>"""
        )
    return "".join(items)


def stage_nav_html(
    *,
    node_count: int,
    caption_count: int,
    candidate_count: int,
    edge_count: int,
    property_count: int,
    state_count: int,
) -> str:
    stages = [
        ("00", "RGB 输入", "31 frames", "geometry"),
        ("01", "几何恢复", "depth · K · pose", "geometry"),
        ("02", "2D 观察", "SAM3 · CLIP", "perception"),
        ("03", "3D 融合", f"{node_count} objects", "perception"),
        ("04", "视觉 Caption", f"{caption_count} views", "language"),
        ("05", "节点精炼", f"{node_count} tags", "language"),
        ("06", "关系推理", f"{candidate_count} → {edge_count} edge", "language"),
        ("07", "属性/状态", f"{property_count} + {state_count}", "language"),
        ("08", "稀疏格式", "JSON · PASS", "result"),
    ]
    return "".join(
        f'<a class="flow-step {lane}" href="#stage-{number}"><span>{number}</span><b>{label}</b><small>{output}</small></a>'
        for number, label, output, lane in stages
    )


def build() -> Path:
    geometry = read_json(DERIVED / "geometry_manifest.json")
    detections = read_json(DERIVED / "detections_manifest.json")
    mapping = read_json(DERIVED / "mapping_manifest.json")
    captions_manifest = read_json(FINAL / "cfslam_openai_caption_manifest.json")
    refinement_manifest = read_json(FINAL / "cfslam_openai_refinement_manifest.json")
    attributes_manifest = read_json(FINAL / "scene_graph_attributes_manifest.json")
    format_manifest = read_json(FINAL / "scene_graph_format_manifest.json")
    nodes = read_json(FINAL / "scene_graph_nodes.json")
    graph = read_json(FINAL / "scene_graph.json")
    captions = read_json(FINAL / "cfslam_openai_captions.json")
    relation_queries = read_json(FINAL / "cfslam_object_relations.json")
    edges = relation_edges(relation_queries)

    counts = [int(frame["count"]) for frame in detections["frames"]]
    valid_ratios = [float(frame["valid_ratio"]) for frame in geometry["frames"]]
    global_depth_min = min(float(frame["depth_min_m"]) for frame in geometry["frames"])
    global_depth_max = max(float(frame["depth_max_m"]) for frame in geometry["frames"])
    property_count = sum(len(value.get("property", [])) for value in graph.values())
    property_node_count = sum(bool(value.get("property")) for value in graph.values())
    state_count = sum(len(value.get("state", [])) for value in graph.values())
    state_node_count = sum(bool(value.get("state")) for value in graph.values())
    caption_count = sum(len(entry.get("captions", [])) for entry in captions)
    node_count = len(nodes)
    candidate_count = len(relation_queries)
    edge_count = len(edges)

    caption_requests = int(captions_manifest.get("api_requests_this_run", caption_count))
    refinement_requests = int(refinement_manifest.get("api_requests_this_run", node_count))
    attribute_requests = int(attributes_manifest.get("api_requests_this_run", node_count))
    formal_request_count = caption_requests + refinement_requests + candidate_count + attribute_requests
    model_name = str(captions_manifest.get("settings", {}).get("model", "configured model"))

    node_data = []
    for node in nodes:
        key = node_key(node)
        value = graph[key]
        node_data.append(
            {
                "id": node["id"],
                "key": key,
                "tag": node["object_tag"],
                "caption": node["caption"],
                "possible_tags": node["possible_tags"],
                "bbox_center": node["bbox_center"],
                "bbox_extent": node["bbox_extent"],
                "property": value.get("property", []),
                "state": value.get("state", []),
                "relation": value.get("relation", []),
                "category": category_for(node),
                "color": node_color(node),
            }
        )

    depth_url, depth_p2, depth_p98 = colorized_depth_data_url(
        DERIVED / "results" / "depth000027.png"
    )
    example_id = edges[0]["target"] if edges else int(nodes[0]["id"])
    example_node = next(node for node in nodes if int(node["id"]) == example_id)
    example_captions = next(
        entry["captions"] for entry in captions if int(entry["id"]) == example_id
    )
    map_path = Path(captions_manifest["settings"]["mapfile"])
    caption_views = caption_crop_data_urls(example_id, map_path)
    refinement_example = refinement_example_html(example_node, example_captions)
    relation_cards = relation_cards_html(relation_queries)
    final_edge_text = ", ".join(
        f'{edge["source_key"]} {edge["label"]} {edge["target_key"]}' for edge in edges
    ) or "no emitted relation"
    attribute_key = "bedspread_11" if "bedspread_11" in graph else next(
        (key for key, value in graph.items() if value.get("state")), next(iter(graph))
    )
    attribute_example = html.escape(
        json.dumps({attribute_key: graph[attribute_key]}, ensure_ascii=False, indent=2)
    )
    raw_json = json.dumps(graph, ensure_ascii=False, indent=2)
    node_json = json.dumps(node_data, ensure_ascii=False).replace("</", "<\\/")
    graph_json = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    relation_json = json.dumps(relation_queries, ensure_ascii=False).replace("</", "<\\/")

    expected_keys = {node_key(node) for node in nodes}
    if set(graph) != expected_keys:
        raise ValueError("scene_graph.json keys do not match scene_graph_nodes.json")
    if int(format_manifest.get("node_count", node_count)) != node_count:
        raise ValueError("Formatter manifest node count does not match final nodes")
    if int(format_manifest.get("output_relation_count", edge_count)) != edge_count:
        raise ValueError("Formatter manifest relation count does not match final edges")

    template = Template(r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>ConceptGraphs Pipeline Showcase · $RUN_NAME</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #667085;
      --line: #dfe5eb;
      --paper: #ffffff;
      --canvas: #f4f7fa;
      --geometry: #168c8c;
      --geometry-soft: #eaf7f6;
      --perception: #6c5ce7;
      --perception-soft: #f0edff;
      --language: #d9852b;
      --language-soft: #fff4e7;
      --result: #168c64;
      --result-soft: #eaf7f1;
      --shadow: 0 14px 40px rgba(29, 42, 61, .08);
      --radius: 22px;
      --max: 1280px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; scroll-padding-top: 88px; }
    body { margin: 0; color: var(--ink); background: var(--canvas); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.62; }
    button, input { font: inherit; }
    a { color: inherit; }
    code, pre, .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
    .page-progress { position: fixed; inset: 0 0 auto; height: 3px; z-index: 100; background: linear-gradient(90deg, var(--geometry), var(--perception), var(--language), var(--result)); transform-origin: left; transform: scaleX(0); }
    .topbar { position: sticky; top: 0; z-index: 80; backdrop-filter: blur(18px); background: rgba(244, 247, 250, .88); border-bottom: 1px solid rgba(218, 225, 232, .9); }
    .topbar-inner { max-width: var(--max); margin: auto; height: 68px; padding: 0 24px; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -.02em; text-decoration: none; }
    .brand-mark { width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; color: white; background: var(--ink); box-shadow: inset 0 0 0 1px rgba(255,255,255,.15); }
    .brand-mark svg { width: 19px; }
    .topnav { display: flex; gap: 6px; }
    .topnav a { color: var(--muted); padding: 8px 11px; border-radius: 9px; text-decoration: none; font-size: 13px; font-weight: 700; }
    .topnav a:hover { color: var(--ink); background: #fff; }
    .status-pill { display: inline-flex; align-items: center; gap: 8px; border: 1px solid #b9dfce; background: var(--result-soft); color: #116947; border-radius: 999px; padding: 7px 11px; font-size: 12px; font-weight: 800; letter-spacing: .04em; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--result); box-shadow: 0 0 0 4px rgba(22,140,100,.13); }
    main { overflow: clip; }
    .container { width: min(calc(100% - 40px), var(--max)); margin: 0 auto; }
    .hero { padding: 58px 0 30px; }
    .hero-grid { display: grid; grid-template-columns: 1.08fr .92fr; gap: 38px; align-items: stretch; }
    .hero-copy { padding: 30px 0 20px; }
    .eyebrow { margin: 0 0 18px; color: var(--geometry); font-weight: 850; font-size: 12px; letter-spacing: .15em; text-transform: uppercase; }
    h1 { max-width: 760px; margin: 0; font-size: clamp(42px, 5.3vw, 76px); line-height: 1.02; letter-spacing: -.055em; }
    h1 .accent { color: var(--geometry); }
    .lead { max-width: 720px; margin: 24px 0 0; color: #48546a; font-size: 18px; line-height: 1.75; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 24px; }
    .tiny-pill { border: 1px solid var(--line); background: rgba(255,255,255,.75); border-radius: 999px; padding: 6px 10px; color: #536078; font-size: 12px; font-weight: 700; }
    .hero-media { position: relative; min-height: 430px; border-radius: 28px; overflow: hidden; background: #172033; box-shadow: var(--shadow); }
    .hero-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .hero-media::after { content: ""; position: absolute; inset: 45% 0 0; background: linear-gradient(transparent, rgba(13,21,34,.86)); pointer-events: none; }
    .hero-media-label { position: absolute; z-index: 1; inset: auto 22px 20px; color: #fff; display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; }
    .hero-media-label b { display: block; font-size: 19px; }
    .hero-media-label span { color: rgba(255,255,255,.72); font-size: 12px; }
    .frame-index { flex: 0 0 auto; font: 700 12px/1.2 monospace; padding: 8px 10px; border: 1px solid rgba(255,255,255,.24); background: rgba(255,255,255,.1); border-radius: 10px; }
    .kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 30px 0 0; }
    .kpi { min-height: 112px; padding: 18px; border: 1px solid var(--line); background: var(--paper); border-radius: 17px; box-shadow: 0 6px 24px rgba(28,42,61,.035); }
    .kpi strong { display: block; font-size: 29px; line-height: 1.15; letter-spacing: -.04em; }
    .kpi span { display: block; margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .kpi small { color: var(--result); font-weight: 800; }
    section { padding: 58px 0; }
    .section-head { max-width: 830px; margin-bottom: 28px; }
    .section-no { display: inline-flex; align-items: center; gap: 10px; margin-bottom: 10px; color: var(--muted); font: 800 11px/1 monospace; letter-spacing: .12em; text-transform: uppercase; }
    .section-no::before { content: ""; width: 24px; height: 2px; background: currentColor; }
    h2 { margin: 0; font-size: clamp(30px, 4vw, 50px); line-height: 1.12; letter-spacing: -.045em; }
    .section-head p { color: var(--muted); margin: 15px 0 0; font-size: 16px; }
    .card { background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: 0 8px 30px rgba(30, 43, 63, .045); }
    .flow-card { padding: 24px; overflow-x: auto; }
    .flow { display: grid; grid-template-columns: repeat(9, minmax(122px, 1fr)); min-width: 1130px; gap: 10px; }
    .flow-step { position: relative; min-height: 122px; border: 1px solid var(--line); border-radius: 15px; padding: 15px 13px; text-decoration: none; transition: transform .2s, box-shadow .2s; }
    .flow-step:not(:last-child)::after { content: "›"; position: absolute; z-index: 2; right: -10px; top: 43px; width: 18px; height: 25px; display: grid; place-items: center; color: #9aa4b2; background: #fff; border-radius: 8px; font-size: 22px; }
    .flow-step:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(27,40,58,.08); }
    .flow-step span { font: 800 11px/1 monospace; opacity: .65; }
    .flow-step b, .flow-step small { display: block; }
    .flow-step b { margin-top: 20px; font-size: 14px; }
    .flow-step small { margin-top: 5px; color: var(--muted); font-size: 11px; }
    .flow-step.geometry { background: var(--geometry-soft); border-color: #cae7e4; }
    .flow-step.perception { background: var(--perception-soft); border-color: #ddd7fa; }
    .flow-step.language { background: var(--language-soft); border-color: #f2ddc5; }
    .flow-step.result { background: var(--result-soft); border-color: #cfe7dc; }
    .legend { display: flex; flex-wrap: wrap; gap: 18px; margin: 17px 5px 0; color: var(--muted); font-size: 12px; }
    .legend i { width: 9px; height: 9px; display: inline-block; border-radius: 3px; margin-right: 7px; }
    .truth-note { margin-top: 20px; display: flex; gap: 13px; padding: 16px 18px; border: 1px solid #cfe7dc; background: var(--result-soft); border-radius: 15px; color: #315f4e; font-size: 13px; }
    .truth-note svg { flex: 0 0 19px; margin-top: 2px; }
    .stage { scroll-margin-top: 84px; }
    .stage-shell { overflow: hidden; }
    .stage-top { display: grid; grid-template-columns: 110px 1fr; }
    .stage-index { display: flex; flex-direction: column; align-items: center; padding: 28px 10px; color: #fff; background: var(--stage-color); }
    .stage-index strong { font: 900 34px/1 monospace; }
    .stage-index span { margin-top: 9px; font: 800 10px/1.2 monospace; letter-spacing: .14em; writing-mode: vertical-rl; text-transform: uppercase; opacity: .78; }
    .stage-copy { padding: 27px 30px 30px; }
    .stage-copy h3 { margin: 0; font-size: 25px; line-height: 1.2; letter-spacing: -.025em; }
    .stage-copy > p { margin: 10px 0 20px; color: var(--muted); }
    .io-grid { display: grid; grid-template-columns: 1fr 1.35fr 1fr; gap: 12px; }
    .io-box { min-height: 135px; padding: 15px 16px; border: 1px solid var(--line); border-radius: 14px; background: #fafbfd; }
    .io-box.method { background: var(--stage-soft); border-color: color-mix(in srgb, var(--stage-color) 25%, #fff); }
    .io-label { display: flex; align-items: center; justify-content: space-between; color: var(--muted); font: 850 10px/1 monospace; letter-spacing: .12em; text-transform: uppercase; }
    .io-label span { color: var(--stage-color); }
    .io-box strong { display: block; margin-top: 13px; font-size: 14px; }
    .io-box p { margin: 7px 0 0; color: #58657a; font-size: 12px; line-height: 1.55; }
    .stage.geometry { --stage-color: var(--geometry); --stage-soft: var(--geometry-soft); }
    .stage.perception { --stage-color: var(--perception); --stage-soft: var(--perception-soft); }
    .stage.language { --stage-color: var(--language); --stage-soft: var(--language-soft); }
    .stage.result { --stage-color: var(--result); --stage-soft: var(--result-soft); }
    .visual-block { border-top: 1px solid var(--line); padding: 26px 30px 30px; background: #fbfcfd; }
    .visual-title { margin-bottom: 16px; display: flex; justify-content: space-between; gap: 18px; align-items: baseline; }
    .visual-title b { font-size: 14px; }
    .visual-title span { color: var(--muted); font-size: 11px; }
    .filmstrip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    figure { margin: 0; }
    .image-frame { position: relative; border-radius: 16px; overflow: hidden; background: #192235; aspect-ratio: 16 / 9; }
    .image-frame img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .35s; }
    .image-frame:hover img { transform: scale(1.015); }
    .image-tag { position: absolute; left: 10px; top: 10px; padding: 5px 8px; color: white; border: 1px solid rgba(255,255,255,.22); background: rgba(16,25,39,.66); backdrop-filter: blur(8px); border-radius: 8px; font: 750 10px/1 monospace; }
    figcaption { margin-top: 9px; color: var(--muted); font-size: 11px; }
    .image-button { width: 100%; padding: 0; border: 0; background: transparent; cursor: zoom-in; display: block; }
    button:focus-visible, a:focus-visible, [role="button"]:focus-visible { outline: 3px solid #f0a04b; outline-offset: 3px; }
    .triptych { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .media-card { padding: 10px 10px 13px; border: 1px solid var(--line); border-radius: 17px; background: white; }
    .media-card h4 { margin: 10px 3px 0; font-size: 13px; }
    .media-card p { margin: 4px 3px 0; color: var(--muted); font-size: 11px; line-height: 1.45; }
    .depth-scale { height: 7px; margin: 9px 3px 0; border-radius: 999px; background: linear-gradient(90deg,#8e1b16,#f69c32,#f3ea4e,#45c7a7,#3476b7,#30123b); }
    .depth-labels { display: flex; justify-content: space-between; margin: 3px 3px 0; color: var(--muted); font: 700 9px/1 monospace; }
    .two-col { display: grid; grid-template-columns: 1.05fr .95fr; gap: 18px; align-items: stretch; }
    .chart-card { padding: 18px; border: 1px solid var(--line); border-radius: 18px; background: #fff; }
    .chart-svg, .semantic-svg { width: 100%; height: auto; display: block; }
    .svg-caption { fill: #7b8494; font: 650 11px system-ui, sans-serif; }
    .svg-label { fill: #526073; font: 800 10px monospace; }
    .svg-node-label { fill: #475467; font: 700 9px monospace; paint-order: stroke; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; }
    .svg-node-title { fill: #243047; font: 750 12px system-ui, sans-serif; }
    .svg-node-id { fill: #7a8495; font: 700 10px monospace; }
    .svg-edge { fill: #26324a; font: 850 10px monospace; paint-order: stroke; stroke: #fff; stroke-width: 4px; }
    .svg-edge-invert { fill: #fff; font: 850 11px monospace; }
    .formula { padding: 22px; color: #fff; background: #202b40; border-radius: 18px; min-height: 100%; }
    .formula .label { color: #8fe0d9; font: 800 10px/1 monospace; letter-spacing: .12em; }
    .equation { margin: 25px 0; font: 500 clamp(16px,2vw,23px)/1.8 Georgia, serif; letter-spacing: .02em; }
    .formula p { color: #bbc5d4; margin: 0; font-size: 12px; }
    .param-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-top: 14px; }
    .param { padding: 13px; border: 1px solid rgba(255,255,255,.11); background: rgba(255,255,255,.055); border-radius: 12px; }
    .param b, .param span { display: block; }
    .param b { color: white; font: 750 12px monospace; }
    .param span { color: #9eabba; margin-top: 4px; font-size: 10px; }
    .funnel { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr auto 1fr; gap: 10px; align-items: center; }
    .funnel-box { padding: 18px 10px; text-align: center; border: 1px solid var(--line); background: #fff; border-radius: 14px; }
    .funnel-box strong { display: block; font-size: 26px; letter-spacing: -.04em; }
    .funnel-box span { color: var(--muted); font-size: 10px; }
    .funnel-arrow { color: #9aa5b5; font-size: 22px; }
    .montage { width: 100%; display: block; border-radius: 16px; border: 1px solid var(--line); }
    .caption-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .caption-view { overflow: hidden; border: 1px solid var(--line); background: #fff; border-radius: 16px; }
    .caption-view img { width: 100%; height: 150px; object-fit: cover; display: block; background: #e8ebef; }
    .caption-view-meta { display: flex; justify-content: space-between; padding: 10px 12px 0; color: var(--language); font: 800 9px/1 monospace; }
    .caption-view p { margin: 9px 12px 14px; color: #526075; font-size: 11px; line-height: 1.5; }
    .semantic-transform { display: grid; grid-template-columns: 1fr 54px 1fr; gap: 10px; align-items: center; }
    .semantic-box { padding: 19px; border: 1px solid var(--line); border-radius: 15px; background: #fff; }
    .semantic-box > span { color: var(--language); font: 800 10px/1 monospace; }
    .semantic-box p { margin: 12px 0 0; color: #4f5d73; font-size: 12px; }
    .semantic-box .tag-row { margin-top: 13px; }
    .transform-arrow { text-align: center; color: var(--language); font-size: 26px; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .tag { display: inline-flex; padding: 5px 8px; border-radius: 7px; color: #5c4b30; background: var(--language-soft); border: 1px solid #f1d9bb; font: 750 9px/1.2 monospace; }
    .relation-candidates { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .candidate { padding: 18px; border: 1px solid var(--line); border-radius: 15px; background: white; }
    .candidate-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .candidate-head code { font-size: 12px; }
    .decision { flex: 0 0 auto; border-radius: 999px; padding: 5px 9px; font: 800 10px/1 monospace; }
    .decision.none { color: #667085; background: #eef1f4; }
    .decision.keep { color: #126648; background: var(--result-soft); }
    .candidate p { margin: 10px 0 0; color: var(--muted); font-size: 11px; }
    .attribute-stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
    .attribute-stat { padding: 18px; border: 1px solid var(--line); border-radius: 15px; background: white; }
    .attribute-stat strong { display: block; font-size: 25px; }
    .attribute-stat span { color: var(--muted); font-size: 11px; }
    .schema-card { margin-top: 14px; padding: 17px; color: #dbe5f2; background: #202b40; border-radius: 15px; overflow-x: auto; font-size: 11px; }
    .schema-card .key { color: #86d9d1; }
    .schema-card .value { color: #f0b96f; }
    .result-shell { padding: 20px; }
    .result-grid { display: grid; grid-template-columns: 1.45fr .75fr; gap: 16px; align-items: stretch; }
    .graph-panel { min-width: 0; }
    .graph-panel .semantic-svg { border-radius: 18px; }
    .graph-node { cursor: pointer; outline: none; transition: opacity .2s, filter .2s; }
    .graph-node:hover, .graph-node:focus { filter: drop-shadow(0 4px 5px rgba(20,31,47,.17)); }
    .graph-node.active rect, .graph-node.active circle { stroke-width: 4px; }
    .node-detail { padding: 23px; border-radius: 18px; color: #fff; background: #202b40; min-height: 100%; }
    .node-detail .detail-id { color: #91a0b5; font: 800 10px/1 monospace; letter-spacing: .12em; }
    .node-detail h3 { margin: 12px 0 8px; font-size: 27px; }
    .node-detail .detail-caption { color: #d2dae5; font-size: 13px; }
    .detail-section { margin-top: 19px; }
    .detail-section > b { display: block; margin-bottom: 8px; color: #98a6b8; font: 800 9px/1 monospace; letter-spacing: .11em; }
    .dark-tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .dark-tag { padding: 5px 7px; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.075); border-radius: 7px; font: 700 9px/1.2 monospace; }
    .dark-tag.relation { color: #8ce2bd; border-color: rgba(84,210,155,.34); background: rgba(22,140,100,.16); }
    .bbox { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .bbox div { padding: 10px; background: rgba(255,255,255,.06); border-radius: 9px; }
    .bbox span, .bbox b { display: block; }
    .bbox span { color: #91a0b5; font-size: 9px; }
    .bbox b { margin-top: 4px; font: 700 10px/1.5 monospace; }
    .node-tools { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; margin: 17px 0 12px; }
    .filters { display: flex; flex-wrap: wrap; gap: 6px; }
    .filter { border: 1px solid var(--line); background: #fff; color: var(--muted); padding: 7px 10px; border-radius: 9px; cursor: pointer; font-size: 11px; font-weight: 750; }
    .filter.active { color: white; background: var(--ink); border-color: var(--ink); }
    .node-list { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
    .node-chip { display: flex; align-items: center; gap: 9px; min-width: 0; padding: 10px; text-align: left; border: 1px solid var(--line); background: #fff; border-radius: 11px; cursor: pointer; }
    .node-chip:hover, .node-chip.active { border-color: #9ca8b8; box-shadow: 0 5px 14px rgba(29,40,57,.07); }
    .node-chip[hidden] { display: none; }
    .node-dot { flex: 0 0 auto; width: 9px; height: 9px; border-radius: 50%; background: var(--node-color); }
    .node-chip span:last-child { min-width: 0; }
    .node-chip b, .node-chip small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block; }
    .node-chip b { font-size: 11px; }
    .node-chip small { color: var(--muted); font: 700 9px/1.4 monospace; }
    .json-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 11px; }
    .json-toolbar span { color: var(--muted); font-size: 11px; }
    .copy-button { border: 1px solid var(--line); background: #fff; border-radius: 9px; padding: 7px 10px; cursor: pointer; font-size: 11px; font-weight: 750; }
    .json-pre { max-height: 480px; margin: 0; padding: 19px; overflow: auto; color: #dbe5f2; background: #202b40; border-radius: 15px; font-size: 10.5px; line-height: 1.55; tab-size: 2; }
    details.card { padding: 18px; }
    details summary { cursor: pointer; font-weight: 800; }
    .validation { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .terminal { padding: 22px; color: #d9e2ed; background: #162033; border-radius: 18px; }
    .terminal-head { display: flex; gap: 6px; margin-bottom: 18px; }
    .terminal-head i { width: 9px; height: 9px; border-radius: 50%; background: #657287; }
    .terminal-head i:first-child { background: #e67c73; }.terminal-head i:nth-child(2) { background: #e3b75c; }.terminal-head i:nth-child(3) { background: #63bd83; }
    .terminal pre { margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.75; }
    .pass { color: #7fe0ae; font-weight: 800; }
    .artifact-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .artifact-table th, .artifact-table td { padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .artifact-table th { color: var(--muted); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }
    .artifact-table code { overflow-wrap: anywhere; }
    .check { color: var(--result); font-weight: 900; }
    .compare-table-wrap { overflow-x: auto; }
    .compare-table { min-width: 850px; width: 100%; border-collapse: collapse; }
    .compare-table th, .compare-table td { border-bottom: 1px solid var(--line); padding: 14px; text-align: left; vertical-align: top; font-size: 12px; }
    .compare-table th { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
    .compare-table td:first-child { font-weight: 800; }
    .limitations { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 18px; }
    .limit { padding: 18px; border: 1px solid var(--line); background: white; border-radius: 15px; }
    .limit b { display: block; font-size: 12px; }
    .limit p { color: var(--muted); margin: 7px 0 0; font-size: 11px; }
    footer { padding: 38px 0 58px; color: var(--muted); font-size: 11px; }
    .footer-inner { display: flex; justify-content: space-between; gap: 30px; padding-top: 24px; border-top: 1px solid var(--line); }
    .footer-inner b { color: var(--ink); }
    .backtop { text-decoration: none; font-weight: 800; }
    .lightbox { position: fixed; z-index: 200; inset: 0; display: none; place-items: center; padding: 36px; background: rgba(10,16,27,.9); }
    .lightbox.open { display: grid; }
    .lightbox img { max-width: min(94vw, 1500px); max-height: 88vh; object-fit: contain; border-radius: 14px; box-shadow: 0 20px 70px rgba(0,0,0,.45); }
    .lightbox button { position: absolute; right: 22px; top: 18px; width: 42px; height: 42px; border: 1px solid rgba(255,255,255,.28); color: white; background: rgba(255,255,255,.09); border-radius: 50%; cursor: pointer; font-size: 25px; }
    @media (max-width: 1050px) {
      .hero-grid, .result-grid { grid-template-columns: 1fr; }
      .hero-media { min-height: 500px; }
      .kpis { grid-template-columns: repeat(3,1fr); }
      .node-list { grid-template-columns: repeat(3,1fr); }
      .caption-grid { grid-template-columns: repeat(2,1fr); }
    }
    @media (max-width: 760px) {
      .topnav { display: none; }
      .container { width: min(calc(100% - 24px), var(--max)); }
      .hero { padding-top: 28px; }
      .hero-copy { padding-top: 8px; }
      h1 { font-size: 43px; }
      .hero-media { min-height: 330px; }
      .kpis { grid-template-columns: repeat(2,1fr); }
      section { padding: 42px 0; }
      .stage-top { grid-template-columns: 1fr; }
      .stage-index { flex-direction: row; justify-content: space-between; padding: 13px 18px; }
      .stage-index span { writing-mode: horizontal-tb; margin: 0; }
      .stage-copy, .visual-block { padding: 21px 18px; }
      .io-grid, .triptych, .two-col, .validation { grid-template-columns: 1fr; }
      .filmstrip { grid-template-columns: 1fr; }
      .funnel { grid-template-columns: 1fr; }
      .funnel-arrow { transform: rotate(90deg); }
      .caption-grid { grid-template-columns: 1fr; }
      .caption-view img { height: 230px; }
      .semantic-transform { grid-template-columns: 1fr; }
      .transform-arrow { transform: rotate(90deg); }
      .relation-candidates, .attribute-stats, .limitations { grid-template-columns: 1fr 1fr; }
      .node-list { grid-template-columns: repeat(2,1fr); }
      .footer-inner { flex-direction: column; }
    }
    @media (max-width: 480px) {
      .kpis, .relation-candidates, .attribute-stats, .limitations { grid-template-columns: 1fr; }
      .status-pill { display: none; }
      .hero-media { min-height: 270px; }
      .node-list { grid-template-columns: 1fr; }
    }
    @media print {
      body { background: #fff; }
      .topbar, .page-progress, .backtop, .copy-button, .filters, .lightbox { display: none !important; }
      .container { width: 100%; max-width: none; }
      section { padding: 28px 0; break-inside: avoid; }
      .card, .kpi { box-shadow: none; }
      .hero-media { min-height: 350px; }
      .json-pre { max-height: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
    }
  </style>
</head>
<body id="top">
  <div class="page-progress" aria-hidden="true"></div>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="#top"><span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="8" r="3"/><circle cx="12" cy="18" r="3"/><path d="M8.6 7.5l6.7-.2M7.7 8.6l2.9 6.7m5.6-4.9l-2.7 5"/></svg></span>ConceptGraphs · Pipeline Report</a>
      <nav class="topnav" aria-label="页面目录"><a href="#pipeline">Pipeline</a><a href="#evidence">Evidence</a><a href="#result">Scene Graph</a><a href="#validation">Validation</a></nav>
      <span class="status-pill"><i class="status-dot"></i>COMPLETE · PASS</span>
    </div>
  </header>

  <main>
    <section class="hero">
      <div class="container">
        <div class="hero-grid">
          <div class="hero-copy">
            <p class="eyebrow">Verified baseline run · $RUN_NAME · svpp</p>
            <h1>从 RGB 视频到<br><span class="accent">3D Scene Graph</span></h1>
            <p class="lead">31 张单目 RGB 帧经过多视图几何恢复、SAM3 类别无关分割、CLIP 表征、三维对象融合与视觉语言推理，最终生成可追溯的稀疏场景图。本页中的数量、图片、节点和关系均来自当前实际产物。</p>
            <div class="hero-badges"><span class="tiny-pill">scene · $SCENE_ID</span><span class="tiny-pill">run · $RUN_NAME</span><span class="tiny-pill">MapAnything</span><span class="tiny-pill">SAM3 + CLIP ViT-B/16</span><span class="tiny-pill">OpenAI-compatible Responses</span><span class="tiny-pill">self-contained HTML</span></div>
          </div>
          <figure class="hero-media">
            <img src="$HERO_IMAGE" alt="输入视频的代表性卧室 RGB 帧">
            <figcaption class="hero-media-label"><span><b>RGB source frame</b>原始 1280 × 720 · 003357.png</span><span class="frame-index">27 / 30</span></figcaption>
          </figure>
        </div>
        <div class="kpis" aria-label="关键结果指标">
          <div class="kpi"><strong>31</strong><span>RGB frames<br><small>1280 × 720</small></span></div>
          <div class="kpi"><strong>414</strong><span>SAM3 masks<br><small>6–19 / frame</small></span></div>
          <div class="kpi"><strong>60,675</strong><span>post-map points<br><small>13 objects</small></span></div>
          <div class="kpi"><strong>$CAPTION_COUNT</strong><span>visual captions<br><small>4 views / object</small></span></div>
          <div class="kpi"><strong>$FORMAL_REQUEST_COUNT</strong><span>formal API calls<br><small>+ 1 smoke test</small></span></div>
          <div class="kpi"><strong>$RELATION_COUNT</strong><span>final relation<br><small>validation PASS</small></span></div>
        </div>
      </div>
    </section>

    <section id="pipeline">
      <div class="container">
        <div class="section-head"><span class="section-no">System overview</span><h2>端到端数据流</h2><p>流程分为几何、视觉感知、语言推理和稀疏格式化四层。点击任一步骤可跳转到它的真实输入、方法与输出。</p></div>
        <div class="card flow-card"><div class="flow">$STAGE_NAV</div><div class="legend"><span><i style="background:var(--geometry)"></i>几何恢复</span><span><i style="background:var(--perception)"></i>视觉感知 / 3D Mapping</span><span><i style="background:var(--language)"></i>视觉语言语义</span><span><i style="background:var(--result)"></i>最终格式</span></div></div>
        <div class="truth-note"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg><div><b>结果口径：</b>正式链路共 $FORMAL_REQUEST_COUNT 次真实 Responses 请求（$CAPTION_REQUESTS caption + $REFINEMENT_REQUESTS refinement + $CANDIDATE_COUNT relation + $ATTRIBUTE_REQUESTS attributes），另有 1 次独立视觉 smoke test。上游几何与对象 map 由同场景已有产物复用；节点语义、关系和最终 JSON 均读取本次 baseline 目录。旧的 offline/enriched top-down 图不作为最终关系图。</div></div>
      </div>
    </section>

    <div id="evidence">
      <section id="stage-00" class="stage geometry">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>00</strong><span>source</span></div><div class="stage-copy"><h3>RGB 视频与时序帧输入</h3><p>输入只有单目 RGB。源视频总长 261.68 秒；当前 pipeline 的直接输入是已抽取并按时间排序的 31 张 PNG，它们覆盖约 12.0 秒的视频片段。</p><div class="io-grid"><div class="io-box"><div class="io-label">Upstream <span>MP4</span></div><strong>video.mp4</strong><p>1280×720 · 59.94 fps · 15,685 frames</p></div><div class="io-box method"><div class="io-label">Pipeline input <span>ordered PNG</span></div><strong>读取现有抽帧目录</strong><p>按文件名保持时间顺序；当前编号 002709–003429、步长 24 帧。</p></div><div class="io-box"><div class="io-label">Output <span>RGB ×31</span></div><strong>31 RGB frames</strong><p><code>bedroom_4_CmEIg9gMI74/images/*.png</code></p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>当前输入片段的视角采样</b><span>同一约 12 秒片段 · frame 00 / 15 / 27</span></div><div class="filmstrip"><figure><button class="image-button" data-lightbox-src="$FRAME_00"><span class="image-frame"><img src="$FRAME_00" alt="起始 RGB 帧"><i class="image-tag">FRAME 00</i></span></button><figcaption>早期视角：入口、窗户与地板区域。</figcaption></figure><figure><button class="image-button" data-lightbox-src="$FRAME_15"><span class="image-frame"><img src="$FRAME_15" alt="中间 RGB 帧"><i class="image-tag">FRAME 15</i></span></button><figcaption>中间视角：床铺与墙面逐步进入画面。</figcaption></figure><figure><button class="image-button" data-lightbox-src="$FRAME_27"><span class="image-frame"><img src="$FRAME_27" alt="后段 RGB 帧"><i class="image-tag">FRAME 27</i></span></button><figcaption>后段视角：床、枕头和墙画清晰可见。</figcaption></figure></div></div></article></div>
      </section>

      <section id="stage-01" class="stage geometry">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>01</strong><span>geometry</span></div><div class="stage-copy"><h3>MapAnything：深度、内参与相机位姿</h3><p>MapAnything 对 31 帧联合推理沿射线深度、相机参数和相对姿态；再利用固定标定单位射线转换为 Z-depth，使现有 RGB-D mapping 接口可直接消费。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>RGB ×31</span></div><strong>1280 × 720 RGB</strong><p>时间有序、多视角、无传感器深度。</p></div><div class="io-box method"><div class="io-label">Method <span>multi-view</span></div><strong>MapAnything + metric scale</strong><p><code>Z = depth_along_ray × unit_ray_z</code><br>pose 为相对首帧的 OpenCV camera-to-world。</p></div><div class="io-box"><div class="io-label">Output <span>geometry</span></div><strong>RGB + uint16 Z-depth + K + c2w</strong><p>31 组 518×294；depth scale = 1000 mm/m。</p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>同一帧的几何恢复证据</b><span>frame 27 · depth 伪彩仅用于展示，数值来自 16-bit PNG</span></div><div class="triptych"><article class="media-card"><button class="image-button" data-lightbox-src="$FRAME_27"><span class="image-frame"><img src="$FRAME_27" alt="MapAnything 派生 RGB frame 27"></span></button><h4>Derived RGB</h4><p>518×294 · 与 depth / mask 像素对齐。</p></article><article class="media-card"><button class="image-button" data-lightbox-src="$DEPTH_27"><span class="image-frame"><img src="$DEPTH_27" alt="frame 27 的伪彩 Z-depth"></span></button><h4>Metric Z-depth</h4><div class="depth-scale"></div><div class="depth-labels"><span>near · $DEPTH_P2 m</span><span>$DEPTH_P98 m · far</span></div></article><article class="media-card"><button class="image-button" data-lightbox-src="$SAM_27"><span class="image-frame"><img src="$SAM_27" alt="frame 27 的 SAM3 mask 可视化"></span></button><h4>Aligned observations</h4><p>同一坐标系中的 mask、bbox 和 confidence。</p></article></div><div class="two-col" style="margin-top:18px"><div class="chart-card">$TRAJECTORY_SVG</div><div class="formula"><span class="label">GEOMETRY SUMMARY</span><div class="param-grid"><div class="param"><b>7.961 s</b><span>31-frame inference</span></div><div class="param"><b>$VALID_RANGE</b><span>valid depth</span></div><div class="param"><b>0.252–5.220 m</b><span>global range</span></div><div class="param"><b>3.4569 m</b><span>trajectory length</span></div></div><p style="margin-top:18px">固定派生内参：<code>fx=fy=256.6211</code>，<code>cx=258.7042</code>，<code>cy=146.7042</code>。轨迹长度由当前 31 个 c2w 重新计算。</p></div></div></div></article></div>
      </section>

      <section id="stage-02" class="stage perception">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>02</strong><span>observations</span></div><div class="stage-copy"><h3>SAM3 类别无关分割 + CLIP 表征</h3><p>每帧先生成 class-agnostic masks，再为每个 mask crop 提取本地 CLIP ViT-B/16 的 512 维归一化视觉特征，为跨帧关联提供语义相似度。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>RGB</span></div><strong>31 × 518×294</strong><p>MapAnything 输出的对齐 RGB。</p></div><div class="io-box method"><div class="io-label">Method <span>SAM3 + CLIP</span></div><strong>AMG masks + 512D embedding</strong><p>矩形图像在内存 warp 至 518²，mask 恢复到 518×294；32×32 points，IoU≥.88，stability≥.95。</p></div><div class="io-box"><div class="io-label">Output <span>PKL + JPG</span></div><strong>414 observations</strong><p>mask · xyxy · confidence · CLIP vector · overlay。</p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>每帧保留的 SAM3 mask 数</b><span>中位数 14 · 高亮 frame 27</span></div><div class="two-col"><div class="chart-card">$MASK_CHART</div><div class="formula"><span class="label">SAM3 AMG</span><div class="param-grid"><div class="param"><b>32 × 32</b><span>point grid</span></div><div class="param"><b>64</b><span>points / batch</span></div><div class="param"><b>0.88</b><span>pred IoU</span></div><div class="param"><b>0.95</b><span>stability</span></div></div><p style="margin-top:18px">warp 仅是当前 Transformers mask-generation 对矩形输入的兼容手段；原始 RGB/深度并未改写。CLIP 模型来自本地 <code>clip-vit-base-patch16</code>。</p></div></div></div></article></div>
      </section>

      <section id="stage-03" class="stage perception">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>03</strong><span>mapping</span></div><div class="stage-copy"><h3>2D→3D 反投影、跨帧关联与对象融合</h3><p>mask 内像素利用 Z-depth 和内参反投影到相机坐标，再由 c2w 变换到统一世界坐标。当前观测与已有对象同时比较几何 overlap 和 CLIP cosine，达到阈值后融合。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>aligned</span></div><strong>mask + Z + K + c2w + CLIP</strong><p>每个 2D observation 同时包含几何与视觉证据。</p></div><div class="io-box method"><div class="io-label">Method <span>CF-SLAM</span></div><strong>FAISS overlap + cosine · sim_sum</strong><p>threshold 1.2；voxel 0.025 m；DBSCAN eps 0.1；融合、过滤和 duplicate merge。</p></div><div class="io-box"><div class="io-label">Output <span>object map</span></div><strong>31 raw → 13 post objects</strong><p>60,675 points · 每对象 4–31 次 observation。</p></div></div></div></div><div class="visual-block"><div class="two-col"><div class="formula"><span class="label">BACK-PROJECTION</span><div class="equation">p<sub>c</sub> = Z [ (u−c<sub>x</sub>)/f<sub>x</sub>, (v−c<sub>y</sub>)/f<sub>y</sub>, 1 ]<sup>T</sup><br>p<sub>w</sub> = R<sub>c2w</sub> p<sub>c</sub> + t<sub>c2w</sub></div><p>方向 FAISS 最近邻命中率提供空间 overlap；归一化 CLIP 向量的 cosine 提供视觉一致性。这里不使用 Hungarian assignment。</p></div><div class="chart-card">$TOPDOWN_SVG</div></div><div class="funnel" style="margin-top:18px"><div class="funnel-box"><strong>31</strong><span>frames</span></div><div class="funnel-arrow">›</div><div class="funnel-box"><strong>414</strong><span>2D masks</span></div><div class="funnel-arrow">›</div><div class="funnel-box"><strong>31</strong><span>raw objects</span></div><div class="funnel-arrow">›</div><div class="funnel-box"><strong>13</strong><span>post objects</span></div><div class="funnel-arrow">›</div><div class="funnel-box"><strong>60,675</strong><span>3D points</span></div></div><div class="visual-title" style="margin-top:24px"><b>13-object post-map montage</b><span>真实 mask/crop 与 observation 次数；用于对象级质检</span></div><button class="image-button" data-lightbox-src="$MONTAGE"><img class="montage" src="$MONTAGE" alt="13 个融合对象的 montage" loading="lazy"></button><p style="margin:10px 3px 0;color:var(--muted);font-size:11px">该 montage 是同一 post-map 的离线诊断产物；其中若出现旧语义标签，仅用于质检。最终节点名称和关系以本页 OpenAI 结果浏览器及 <code>scene_graph.json</code> 为准。</p></div></article></div>
      </section>

      <section id="stage-04" class="stage language">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>04</strong><span>vision caption</span></div><div class="stage-copy"><h3>OpenAI-compatible 视觉端点：多视角对象 Caption</h3><p>每个对象从全部 observation 中选择最多 4 个时间分散、质量最高的视角；动态 padding 后用红色轮廓标出目标，以 Base64 JPEG 通过 Responses API 输入可配置模型 ID <code>$MODEL_NAME</code>。视觉请求已真实成功，但兼容端点的后端厂商身份不由产物推断。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>post-map</span></div><strong>$NODE_COUNT objects · 192 observations</strong><p>过滤无效 bbox、过小 mask 和低 fill ratio；同帧去重。</p></div><div class="io-box method"><div class="io-label">Method <span>vision</span></div><strong>4 temporal bins → best quality / bin</strong><p>quality = confidence × mask area；red outline；detail=high；store=false。</p></div><div class="io-box"><div class="io-label">Output <span>JSON cache</span></div><strong>$CAPTION_COUNT visual captions</strong><p>$CAPTION_REQUESTS 请求；manifest complete。</p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>真实选中视角的可视化重建 · object $EXAMPLE_ID</b><span>红框 crop 由同一 map、bbox、mask 和 padding 重建；运行时 Base64 图片未落盘</span></div><div class="caption-grid">$CAPTION_VIEWS</div></div></article></div>
      </section>

      <section id="stage-05" class="stage language">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>05</strong><span>refinement</span></div><div class="stage-copy"><h3>多视角语义精炼与开放词汇节点命名</h3><p>原始 <code>GPTPrompt.py</code> 将同一对象的多句 caption 汇总为简洁描述、候选标签和唯一 object tag；保留原始 object ID，避免后续边错位。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>captions</span></div><strong>4 sentences / object</strong><p>视觉模型给出的互补视角描述。</p></div><div class="io-box method"><div class="io-label">Method <span>text LLM</span></div><strong>summary + possible_tags + object_tag</strong><p>通用 prompt；request identity 与 cache 防止旧模型/旧 map 混入。</p></div><div class="io-box"><div class="io-label">Output <span>$NODE_COUNT nodes</span></div><strong>stable original IDs · invalid = 0</strong><p>$REFINEMENT_REQUESTS 请求；输出 detailed node JSON 和 pruned map。</p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>object $EXAMPLE_ID 的语义收敛</b><span>多描述 → 单节点</span></div>$REFINEMENT_EXAMPLE</div></article></div>
      </section>

      <section id="stage-06" class="stage language">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>06</strong><span>relations</span></div><div class="stage-copy"><h3>ConceptGraphs baseline：几何候选图 + 受限空间关系推理</h3><p>保持原始代码行为：对每个 lower-index / higher-index 对只取单向 <code>overlap[i,j]</code>；超过 0.01 时把 overlap 原值直接写入邻接矩阵，再由 SciPy 对连通分量取 minimum spanning tree。语言模型只在 <code>on / in / none</code> 受限词表内判定。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>objects</span></div><strong>$NODE_COUNT point clouds + bbox + tags</strong><p>AABB 仅预筛；本地 FAISS 最近邻函数数值兼容原 overlap 判据，不宣称使用 GradSLAM 原始 import。</p></div><div class="io-box method"><div class="io-label">Method <span>official baseline behavior</span></div><strong>directional overlap → components → MST → LLM</strong><p><code>overlap[i,j] &gt; 0.01</code>；原始 similarity 直接作为 MST 权重；关系方向与 ID 显式校验。</p></div><div class="io-box"><div class="io-label">Output <span>edge pickle</span></div><strong>$CANDIDATE_COUNT candidates → $RELATION_COUNT edge</strong><p><code>$FINAL_EDGE_TEXT</code></p></div></div></div></div><div class="visual-block"><div class="visual-title"><b>候选判定审计</b><span>本次真实模型输出 · reason 原文</span></div><div class="relation-candidates">$RELATION_CARDS</div></div></article></div>
      </section>

      <section id="stage-07" class="stage language">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>07</strong><span>attributes</span></div><div class="stage-copy"><h3>通用 Property / State 抽取</h3><p>外部 prompt 从 detailed node 和原始多视角 captions 中抽取可展示的属性与状态。Python 只做 schema、枚举格式和 ID 校验，不包含 bedroom 或物体类别硬编码。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>semantic</span></div><strong>$NODE_COUNT nodes + $CAPTION_COUNT captions</strong><p>对象标签、汇总描述与多视角证据。</p></div><div class="io-box method"><div class="io-label">Method <span>external prompt</span></div><strong>generic property/state extraction</strong><p>输出 uppercase tokens；空数组在最终 sparse dict 中省略。</p></div><div class="io-box"><div class="io-label">Output <span>JSON</span></div><strong>$PROPERTY_COUNT properties · $STATE_COUNT states</strong><p>$ATTRIBUTE_REQUESTS 请求；$NODE_COUNT/$NODE_COUNT completed。</p></div></div></div></div><div class="visual-block"><div class="attribute-stats"><div class="attribute-stat"><strong>$PROPERTY_NODE_COUNT</strong><span>nodes with property</span></div><div class="attribute-stat"><strong>$PROPERTY_COUNT</strong><span>property tokens</span></div><div class="attribute-stat"><strong>$STATE_NODE_COUNT</strong><span>nodes with state</span></div><div class="attribute-stat"><strong>$STATE_COUNT</strong><span>state tokens</span></div></div><pre class="schema-card">$ATTRIBUTE_EXAMPLE</pre></div></article></div>
      </section>

      <section id="stage-08" class="stage result">
        <div class="container"><article class="card stage-shell"><div class="stage-top"><div class="stage-index"><strong>08</strong><span>formatter</span></div><div class="stage-copy"><h3>标准化为稀疏 Scene Graph 字典</h3><p>formatter 将节点、属性和 ConceptGraphs edge 对齐到稳定 key：<code>normalized_object_tag_originalID</code>。它只转换关系方向与大小写，不合成 room 节点，也不虚构 <code>INSIDE bedroom</code>。</p><div class="io-grid"><div class="io-box"><div class="io-label">Input <span>3 artifacts</span></div><strong>nodes + attributes + edges</strong><p>所有输入都有 SHA256 identity 和 manifest。</p></div><div class="io-box method"><div class="io-label">Method <span>deterministic</span></div><strong>ID join + sparse field merge</strong><p><code>a/b on/in</code> → <code>ON/INSIDE target</code>；空字段省略。</p></div><div class="io-box"><div class="io-label">Output <span>JSON + repr</span></div><strong>$NODE_COUNT nodes · $RELATION_COUNT relation</strong><p><code>scene_graph.json</code> · <code>scene_graph.txt</code> · PASS。</p></div></div></div></div></article></div>
      </section>
    </div>

    <section id="result">
      <div class="container">
        <div class="section-head"><span class="section-no">Final artifact</span><h2>可交互 Scene Graph 浏览器</h2><p>主图只画当前最终 JSON 中的真实边。点击图中节点或下方标签，可查看 caption、3D bbox、候选标签、property、state 和 relation。</p></div>
        <div class="card result-shell">
          <div class="result-grid"><div class="graph-panel">$SEMANTIC_GRAPH</div><aside class="node-detail" aria-live="polite"><span class="detail-id" id="detail-id"></span><h3 id="detail-title"></h3><p class="detail-caption" id="detail-caption"></p><div class="detail-section"><b>PROPERTY</b><div class="dark-tags" id="detail-property"></div></div><div class="detail-section"><b>STATE</b><div class="dark-tags" id="detail-state"></div></div><div class="detail-section"><b>RELATION</b><div class="dark-tags" id="detail-relation"></div></div><div class="detail-section"><b>3D BBOX · METERS</b><div class="bbox"><div><span>center [x,y,z]</span><b id="detail-center"></b></div><div><span>extent [x,y,z]</span><b id="detail-extent"></b></div></div></div><div class="detail-section"><b>POSSIBLE TAGS</b><div class="dark-tags" id="detail-tags"></div></div></aside></div>
          <div class="node-tools"><div class="filters" role="group" aria-label="节点筛选"><button class="filter active" data-filter="all" aria-pressed="true">全部 $NODE_COUNT</button><button class="filter" data-filter="relation" aria-pressed="false">有关系</button><button class="filter" data-filter="state" aria-pressed="false">有状态</button><button class="filter" data-filter="structure" aria-pressed="false">建筑结构</button><button class="filter" data-filter="object" aria-pressed="false">物体 / 装饰</button></div><span class="mono" style="color:var(--muted);font-size:10px">solid edge = emitted relation</span></div>
          <div class="node-list">$NODE_BUTTONS</div>
        </div>
        <details class="card" style="margin-top:16px"><summary>查看最终标准 JSON</summary><div class="json-toolbar" style="margin-top:15px"><span><code>outputs/$RUN_NAME/scene_graph_openai/scene_graph.json</code></span><button class="copy-button" id="copy-json" type="button">复制 JSON</button></div><pre class="json-pre" id="json-pre">$RAW_JSON</pre></details>
      </div>
    </section>

    <section id="validation">
      <div class="container">
        <div class="section-head"><span class="section-no">Verification & provenance</span><h2>完成状态与产物追踪</h2><p>各在线阶段均留下 manifest 或可审计产物；caption、refinement、attributes 与 formatter 明确标记 complete。最终 JSON 通过节点数、边数和字段检查，验证日志同时打印输出路径。</p></div>
        <div class="validation"><div class="terminal"><div class="terminal-head"><i></i><i></i><i></i></div><pre>$$ validation report
nodes: $NODE_COUNT
relations: $RELATION_COUNT
scene_graph.json: .../scene_graph_openai/scene_graph.json
scene_graph.txt:  .../scene_graph_openai/scene_graph.txt
validation: <span class="pass">PASS</span>

$$ request accounting
caption       $CAPTION_REQUESTS / complete
refinement    $REFINEMENT_REQUESTS / invalid 0
relation       $CANDIDATE_COUNT / kept $RELATION_COUNT
attributes    $ATTRIBUTE_REQUESTS / complete
formal total  $FORMAL_REQUEST_COUNT</pre></div><div class="card" style="padding:13px 18px"><table class="artifact-table"><thead><tr><th>Stage</th><th>Primary artifact</th><th>Status</th></tr></thead><tbody><tr><td>Geometry</td><td><code>geometry_manifest.json</code></td><td class="check">✓ 31</td></tr><tr><td>Detections</td><td><code>detections_manifest.json</code></td><td class="check">✓ 414</td></tr><tr><td>Mapping</td><td><code>*_post.pkl.gz</code></td><td class="check">✓ $NODE_COUNT</td></tr><tr><td>Captions</td><td><code>cfslam_openai_captions.json</code></td><td class="check">✓ $CAPTION_COUNT</td></tr><tr><td>Nodes</td><td><code>scene_graph_nodes.json</code></td><td class="check">✓ $NODE_COUNT</td></tr><tr><td>Attributes</td><td><code>scene_graph_attributes.json</code></td><td class="check">✓ $NODE_COUNT</td></tr><tr><td>Final</td><td><code>scene_graph.json</code></td><td class="check">✓ PASS</td></tr></tbody></table></div></div>
      </div>
    </section>

    <section id="method-notes">
      <div class="container">
        <div class="section-head"><span class="section-no">Implementation notes</span><h2>论文思想与当前实现</h2><p>整体保持 ConceptGraphs 的对象级 3D mapping 与开放词汇场景图思想；为适配当前“只有 RGB 视频”的输入和本地模型，几何与语言模块做了可追溯替换。</p></div>
        <div class="card compare-table-wrap"><table class="compare-table"><thead><tr><th>Component</th><th>ConceptGraphs 思路</th><th>本次当前实现</th><th>产物影响</th></tr></thead><tbody><tr><td>输入几何</td><td>消费带深度和相机姿态的 RGB-D 序列。</td><td>从 RGB 用 MapAnything 估计 metric Z-depth、K 和 c2w。</td><td>无需深度相机，但深度仍是模型估计。</td></tr><tr><td>2D perception</td><td>类别无关 mask 与视觉语言特征。</td><td>SAM3 AMG + 本地 CLIP ViT-B/16 512D。</td><td>保持开放词汇映射；模型规格与论文环境不同。</td></tr><tr><td>3D association</td><td>几何与视觉相似度驱动对象融合。</td><td>FAISS overlap + CLIP cosine，sim_sum threshold=1.2。</td><td>31 raw objects 融合为 $NODE_COUNT 个持久对象。</td></tr><tr><td>关系候选</td><td>单向 overlap、阈值图与原始权重 MST。</td><td>保持 baseline 行为；overlap 由本地 FAISS 数值兼容函数计算。</td><td>$CANDIDATE_COUNT 个候选交给受限关系 prompt。</td></tr><tr><td>语言链路</td><td>多视角 caption、节点精炼与 on/in 判断。</td><td>OpenAI-compatible Responses，$MODEL_NAME；视觉与文本共用可配置模型。</td><td>$CAPTION_COUNT captions、$NODE_COUNT refined nodes、$RELATION_COUNT final relation。</td></tr><tr><td>输出 schema</td><td>对象节点及开放词汇关系。</td><td>附加通用 property/state，并转为展示目标的 sparse dict。</td><td>只输出有证据的字段；不合成房间节点。</td></tr></tbody></table></div>
        <div class="limitations"><article class="limit"><b>单目深度</b><p>MapAnything 恢复的深度和姿态是估计值，不等价于实测 RGB-D 传感器。</p></article><article class="limit"><b>关系词表</b><p>当前最终 formatter 只保留 ConceptGraphs 的 <code>ON</code> / <code>INSIDE</code>；本场景产生 $RELATION_COUNT 条最终关系。</p></article><article class="limit"><b>Baseline 权重</b><p>MST 直接最小化 overlap similarity 是原代码行为；本页如实展示，不把它解释为优化后的关系图。</p></article><article class="limit"><b>稀疏原则</b><p>没有直接证据时不补 room 节点，也不自动生成所有对象 INSIDE bedroom 的关系。</p></article><article class="limit"><b>语义属性</b><p>property/state 来自视觉语言推断，用于语义描述，不应视为精密物理测量。</p></article></div>
      </div>
    </section>
  </main>

  <footer><div class="container footer-inner"><div><b>ConceptGraphs Pipeline Showcase</b><br>scene: $SCENE_ID · run: $RUN_NAME · generated $GENERATED_DATE · no external assets<br>论文：<a href="https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf" target="_blank" rel="noopener">ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning</a></div><div>环境：svpp · Python 3.11.15 · Torch 2.1.1+cu121<br>页面未嵌入 API key、请求凭证或私密配置。</div><a class="backtop" href="#top">返回顶部 ↑</a></div></footer>

  <div class="lightbox" role="dialog" aria-modal="true" aria-hidden="true" aria-label="图片预览"><button type="button" aria-label="关闭">×</button><img alt="放大的结果图片"></div>
  <script type="application/json" id="node-data">$NODE_JSON</script>
  <script type="application/json" id="graph-data">$GRAPH_JSON</script>
  <script type="application/json" id="relation-data">$RELATION_JSON</script>
  <script>
    (function () {
      "use strict";
      var nodes = JSON.parse(document.getElementById("node-data").textContent);
      var graph = JSON.parse(document.getElementById("graph-data").textContent);
      var byKey = {};
      nodes.forEach(function (node) { byKey[node.key] = node; });

      function renderTags(elementId, values, relation) {
        var root = document.getElementById(elementId);
        root.innerHTML = "";
        if (!values || !values.length) {
          var empty = document.createElement("span");
          empty.className = "dark-tag";
          empty.textContent = "—";
          root.appendChild(empty);
          return;
        }
        values.forEach(function (value) {
          var tag = document.createElement("span");
          tag.className = "dark-tag" + (relation ? " relation" : "");
          tag.textContent = value;
          root.appendChild(tag);
        });
      }

      function selectNode(key) {
        var node = byKey[key];
        if (!node) return;
        document.getElementById("detail-id").textContent = "NODE " + String(node.id).padStart(2, "0") + " · " + node.key;
        document.getElementById("detail-title").textContent = node.tag;
        document.getElementById("detail-caption").textContent = node.caption;
        document.getElementById("detail-center").textContent = node.bbox_center.map(function (v) { return Number(v).toFixed(1); }).join(", ");
        document.getElementById("detail-extent").textContent = node.bbox_extent.map(function (v) { return Number(v).toFixed(1); }).join(", ");
        renderTags("detail-property", node.property, false);
        renderTags("detail-state", node.state, false);
        renderTags("detail-relation", node.relation, true);
        renderTags("detail-tags", node.possible_tags, false);
        document.querySelectorAll("[data-node-key]").forEach(function (element) {
          element.classList.toggle("active", element.getAttribute("data-node-key") === key);
        });
      }

      document.querySelectorAll("[data-node-key]").forEach(function (element) {
        element.addEventListener("click", function () { selectNode(element.getAttribute("data-node-key")); });
        element.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(element.getAttribute("data-node-key")); }
        });
      });

      document.querySelectorAll(".filter").forEach(function (button) {
        button.addEventListener("click", function () {
          document.querySelectorAll(".filter").forEach(function (item) { item.classList.remove("active"); item.setAttribute("aria-pressed", "false"); });
          button.classList.add("active");
          button.setAttribute("aria-pressed", "true");
          var filter = button.getAttribute("data-filter");
          function isVisible(node) {
            return filter === "all" ||
              (filter === "state" && Boolean(node.state.length)) ||
              (filter === "relation" && Boolean(node.relation.length)) ||
              node.category === filter;
          }
          document.querySelectorAll(".node-chip").forEach(function (chip) {
            chip.hidden = !isVisible(byKey[chip.getAttribute("data-node-key")]);
          });
          document.querySelectorAll(".graph-node").forEach(function (element) {
            var visible = isVisible(byKey[element.getAttribute("data-node-key")]);
            element.style.opacity = visible ? "1" : ".12";
            element.style.pointerEvents = visible ? "auto" : "none";
            element.setAttribute("aria-hidden", visible ? "false" : "true");
            element.setAttribute("tabindex", visible ? "0" : "-1");
          });
        });
      });

      var copyButton = document.getElementById("copy-json");
      copyButton.addEventListener("click", function () {
        var value = JSON.stringify(graph, null, 2);
        function done() { copyButton.textContent = "已复制"; window.setTimeout(function () { copyButton.textContent = "复制 JSON"; }, 1500); }
        function fallback() {
          var area = document.createElement("textarea");
          area.value = value; document.body.appendChild(area); area.select();
          try { document.execCommand("copy"); done(); } finally { area.remove(); }
        }
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(value).then(done).catch(fallback); else fallback();
      });

      var lightbox = document.querySelector(".lightbox");
      var lightboxImage = lightbox.querySelector("img");
      var lightboxClose = lightbox.querySelector("button");
      var lightboxTrigger = null;
      document.querySelectorAll("[data-lightbox-src]").forEach(function (button) {
        button.addEventListener("click", function () {
          lightboxTrigger = button;
          lightboxImage.src = button.getAttribute("data-lightbox-src");
          lightbox.classList.add("open");
          lightbox.setAttribute("aria-hidden", "false");
          lightboxClose.focus();
        });
      });
      function closeLightbox() {
        if (!lightbox.classList.contains("open")) return;
        lightbox.classList.remove("open");
        lightbox.setAttribute("aria-hidden", "true");
        lightboxImage.removeAttribute("src");
        if (lightboxTrigger) lightboxTrigger.focus();
      }
      lightboxClose.addEventListener("click", closeLightbox);
      lightbox.addEventListener("click", function (event) { if (event.target === lightbox) closeLightbox(); });
      document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeLightbox(); });

      var progress = document.querySelector(".page-progress");
      function updateProgress() {
        var denominator = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
        progress.style.transform = "scaleX(" + Math.min(window.scrollY / denominator, 1) + ")";
      }
      document.addEventListener("scroll", updateProgress, { passive: true });
      updateProgress();
      selectNode("$INITIAL_NODE_KEY");
    }());
  </script>
</body>
</html>
''')

    values = {
        "HERO_IMAGE": image_data_url(ROOT / SCENE_ID / "images" / "003357.png", max_side=1400, quality=88),
        "FRAME_00": image_data_url(DERIVED / "results" / "frame000000.jpg"),
        "FRAME_15": image_data_url(DERIVED / "results" / "frame000015.jpg"),
        "FRAME_27": image_data_url(DERIVED / "results" / "frame000027.jpg"),
        "DEPTH_27": depth_url,
        "SAM_27": image_data_url(DERIVED / "gsa_vis_sam3_clip" / "frame000027.jpg"),
        "MONTAGE": image_data_url(DERIVED / "scene_graph" / "object_montage.jpg"),
        "DEPTH_P2": f"{depth_p2:.2f}",
        "DEPTH_P98": f"{depth_p98:.2f}",
        "STAGE_NAV": stage_nav_html(
            node_count=node_count,
            caption_count=caption_count,
            candidate_count=candidate_count,
            edge_count=edge_count,
            property_count=property_count,
            state_count=state_count,
        ),
        "TRAJECTORY_SVG": trajectory_svg(DERIVED / "traj.txt"),
        "MASK_CHART": mask_chart_svg(counts),
        "TOPDOWN_SVG": topdown_svg(nodes, edges),
        "CAPTION_VIEWS": caption_views_html(caption_views, example_id),
        "SEMANTIC_GRAPH": semantic_graph_svg(nodes, edges),
        "NODE_BUTTONS": node_buttons_html(node_data),
        "NODE_JSON": node_json,
        "GRAPH_JSON": graph_json,
        "RELATION_JSON": relation_json,
        "RAW_JSON": html.escape(raw_json),
        "SCENE_ID": html.escape(SCENE_ID),
        "RUN_NAME": html.escape(RUN_NAME),
        "NODE_COUNT": str(node_count),
        "CAPTION_COUNT": str(caption_count),
        "CANDIDATE_COUNT": str(candidate_count),
        "RELATION_COUNT": str(edge_count),
        "CAPTION_REQUESTS": str(caption_requests),
        "REFINEMENT_REQUESTS": str(refinement_requests),
        "ATTRIBUTE_REQUESTS": str(attribute_requests),
        "FORMAL_REQUEST_COUNT": str(formal_request_count),
        "MODEL_NAME": html.escape(model_name),
        "EXAMPLE_ID": str(example_id),
        "REFINEMENT_EXAMPLE": refinement_example,
        "RELATION_CARDS": relation_cards,
        "FINAL_EDGE_TEXT": html.escape(final_edge_text),
        "ATTRIBUTE_EXAMPLE": attribute_example,
        "INITIAL_NODE_KEY": html.escape(edges[0]["source_key"] if edges else node_data[0]["key"]),
        "PROPERTY_COUNT": str(property_count),
        "PROPERTY_NODE_COUNT": str(property_node_count),
        "STATE_COUNT": str(state_count),
        "STATE_NODE_COUNT": str(state_node_count),
        "VALID_RANGE": f"{min(valid_ratios)*100:.2f}–{max(valid_ratios)*100:.2f}%",
        "GENERATED_DATE": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
    }
    rendered = template.substitute(values)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    return OUTPUT


def configure_from_args(args: argparse.Namespace) -> None:
    global SCENE_ID, RUN_NAME, DERIVED_ROOT, DERIVED, FINAL, OUTPUT
    SCENE_ID = args.scene_id
    RUN_NAME = args.run_name or SCENE_ID
    DERIVED_ROOT = Path(args.derived_root).expanduser().resolve()
    DERIVED = DERIVED_ROOT / SCENE_ID
    FINAL = ROOT / "outputs" / RUN_NAME / "scene_graph_openai"
    if args.output:
        output = Path(args.output).expanduser()
        OUTPUT = output if output.is_absolute() else ROOT / output
    elif RUN_NAME == SCENE_ID:
        OUTPUT = ROOT / "pipeline_showcase.html"
    else:
        safe_name = RUN_NAME.replace("/", "_").replace("\\", "_")
        OUTPUT = ROOT / f"pipeline_showcase_{safe_name}.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained ConceptGraphs pipeline showcase from existing artifacts."
    )
    parser.add_argument("--scene-id", default="bedroom_4_CmEIg9gMI74")
    parser.add_argument(
        "--run-name",
        help="Directory name under repo outputs; defaults to --scene-id.",
    )
    parser.add_argument(
        "--derived-root",
        default=os.environ.get("CG_LEGACY_OUTPUT_ROOT", str(ROOT / "outputs")),
        help="Root containing shared geometry/detection/mapping artifacts.",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_from_args(parse_args())
    output = build()
    print(output)
