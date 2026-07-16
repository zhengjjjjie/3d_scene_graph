#!/usr/bin/env bash

# One-command, resumable ConceptGraphs scene-graph construction from an
# existing post-map. The API key is read interactively unless the caller has
# explicitly configured OPENAI_API_KEY or OPENAI_API_KEY_FILE.

set -Eeuo pipefail
set +x
umask 077
unset OPENAI_LOG

CG_ROOT="${CG_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
CG_DEPS="${CG_DEPS:-/data2/zhengjie/data/concept_graphs/python_packages}"
CG_PYTHON="${CG_PYTHON:-/data2/zhengjie/miniconda3/envs/svpp/bin/python}"
CG_SCRIPT="${CG_SCRIPT:-$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py}"
OUTPUT_SCRIPT="${OUTPUT_SCRIPT:-$CG_ROOT/conceptgraph/scenegraph/scenegraph_output.py}"
ATTRIBUTE_PROMPT="${ATTRIBUTE_PROMPT:-$CG_ROOT/conceptgraph/scenegraph/prompts/scene_graph_attributes.txt}"

OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.autodl.art/api/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
OPENAI_VISION_MODEL="${OPENAI_VISION_MODEL:-$OPENAI_MODEL}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}"

RUN_NAME="${RUN_NAME:-bedroom_4_CmEIg9gMI74}"
MAP_FILE="${MAP_FILE:-/data2/zhengjie/data/concept_graphs/outputs/bedroom_4_CmEIg9gMI74/pcd_saves/full_pcd_sam3_clip_overlap_maskconf0.95_simsum1.2_dbscan.1_sam3_clip_post.pkl.gz}"
RUN_ROOT="${RUN_ROOT:-$CG_ROOT/outputs/$RUN_NAME}"
SMOKE_CACHE="${SMOKE_CACHE:-$RUN_ROOT/smoke}"
OPENAI_CACHE="${OPENAI_CACHE:-$RUN_ROOT/scene_graph_openai}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
SMOKE_OBJECT_ID="${SMOKE_OBJECT_ID:-5}"
MAX_CAPTION_VIEWS="${MAX_CAPTION_VIEWS:-4}"
DEVICE="${DEVICE:-cuda:0}"

export CG_ROOT CG_DEPS CG_PYTHON CG_SCRIPT OUTPUT_SCRIPT ATTRIBUTE_PROMPT
export OPENAI_BASE_URL OPENAI_MODEL OPENAI_VISION_MODEL OPENAI_TIMEOUT OPENAI_MAX_RETRIES
export MAP_FILE RUN_ROOT SMOKE_CACHE OPENAI_CACHE LOG_DIR
export PYTHONPATH="$CG_DEPS/openai_py311:$CG_DEPS/mapping_py311:$CG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data2/zhengjie/miniconda3/envs/svpp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup_credential() {
  unset OPENAI_API_KEY
}
trap cleanup_credential EXIT
trap 'exit 130' HUP INT TERM

OPENAI_COMMON_ARGS=(
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-model "$OPENAI_MODEL"
  --openai-vision-model "$OPENAI_VISION_MODEL"
  --openai-timeout "$OPENAI_TIMEOUT"
  --openai-max-retries "$OPENAI_MAX_RETRIES"
)
OPENAI_ATTRIBUTE_ARGS=(
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-model "$OPENAI_MODEL"
  --openai-timeout "$OPENAI_TIMEOUT"
  --openai-max-retries "$OPENAI_MAX_RETRIES"
)

if [[ ! -f "$CG_PYTHON" ]]; then
  echo "Missing required file: $CG_PYTHON" >&2
  exit 2
fi

if [[ -n "${OPENAI_API_KEY_FILE:-}" ]]; then
  unset OPENAI_API_KEY
fi

if [[ -n "${OPENAI_API_KEY_FILE:-}" ]] && ! "$CG_PYTHON" -c '
import os, stat, sys
fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    mode = os.fstat(fd).st_mode
    value = os.read(fd, 65537).decode("utf-8").strip()
finally:
    os.close(fd)
valid = (
    stat.S_ISREG(mode)
    and not (stat.S_IMODE(mode) & 0o077)
    and bool(value)
    and len(value) <= 65536
    and not any(c.isspace() for c in value)
)
if not valid:
    raise SystemExit(1)
' "$OPENAI_API_KEY_FILE" >/dev/null 2>&1; then
  echo "已忽略无效的 OPENAI_API_KEY_FILE，改用隐藏输入。" >&2
  unset OPENAI_API_KEY_FILE OPENAI_API_KEY
fi

if [[ -n "${OPENAI_API_KEY_FILE:-}" ]]; then
  export OPENAI_API_KEY_FILE
  OPENAI_COMMON_ARGS+=(--openai-api-key-file "$OPENAI_API_KEY_FILE")
  OPENAI_ATTRIBUTE_ARGS+=(--openai-api-key-file "$OPENAI_API_KEY_FILE")
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp "只粘贴 API key，然后按回车: " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi

if [[ -z "${OPENAI_API_KEY_FILE:-}" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" || "$OPENAI_API_KEY" =~ [[:space:]] ]]; then
    echo "API key 为空或包含空白；请只粘贴 key 值，不要粘贴变量名或命令。" >&2
    exit 2
  fi
fi

for required_file in "$CG_PYTHON" "$CG_SCRIPT" "$OUTPUT_SCRIPT" "$ATTRIBUTE_PROMPT" "$MAP_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

MAP_OBJECT_COUNT=$(env -u OPENAI_API_KEY "$CG_PYTHON" - "$MAP_FILE" <<'PY'
import contextlib
import io
import sys
from types import SimpleNamespace
from conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map
from conceptgraph.slam.slam_classes import MapObjectList

scene_map = MapObjectList()
with contextlib.redirect_stdout(io.StringIO()):
    load_scene_map(SimpleNamespace(mapfile=sys.argv[1]), scene_map)
print(len(scene_map))
PY
)
if [[ ! "$MAP_OBJECT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Map contains no scene objects" >&2
  exit 2
fi
if [[ ! "$SMOKE_OBJECT_ID" =~ ^[0-9]+$ ]] || (( SMOKE_OBJECT_ID >= MAP_OBJECT_COUNT )); then
  echo "SMOKE_OBJECT_ID=$SMOKE_OBJECT_ID is invalid for $MAP_OBJECT_COUNT objects; using 0." >&2
  SMOKE_OBJECT_ID=0
fi
export MAP_OBJECT_COUNT SMOKE_OBJECT_ID

install -d -m 700 "$RUN_ROOT" "$SMOKE_CACHE" "$OPENAI_CACHE" "$LOG_DIR"

echo "[1/9] API preflight"
"$CG_PYTHON" - <<'PY' 2>&1 | tee "$LOG_DIR/00_api_preflight.log"
import os
import platform
from pathlib import Path
from conceptgraph.scenegraph.build_scenegraph_cfslam import make_openai_client

print("python:", platform.python_version())
print("conda:", Path(os.environ["CG_PYTHON"]).parent.parent)
print("base_url:", os.environ["OPENAI_BASE_URL"])
print("text_model:", os.environ["OPENAI_MODEL"])
print("vision_model:", os.environ["OPENAI_VISION_MODEL"])
client = make_openai_client(
    api_key_file=os.getenv("OPENAI_API_KEY_FILE") or None,
    base_url=os.environ["OPENAI_BASE_URL"],
    max_retries=int(os.environ["OPENAI_MAX_RETRIES"]),
)
print("available_models:")
try:
    models = client.models.list(timeout=float(os.environ["OPENAI_TIMEOUT"]))
    for model in models.data:
        print(" -", model.id)
    print("model_count:", len(models.data))
except Exception as exc:
    status = getattr(exc, "status_code", None)
    suffix = f" HTTP {status}" if isinstance(status, int) else ""
    print(" - model listing unavailable:", type(exc).__name__ + suffix)
print("api_preflight: COMPLETE")
PY

echo "[2/9] One-view vision smoke test"
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$SMOKE_CACHE" \
  --mapfile "$MAP_FILE" \
  --annot-inds "$SMOKE_OBJECT_ID" \
  --max-detections-per-object 1 \
  --masking-option red_outline \
  --openai-image-detail high \
  2>&1 | tee "$LOG_DIR/01_caption_smoke.log"

"$CG_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SMOKE_CACHE"]) / "cfslam_openai_caption_manifest.json"
value = json.loads(path.read_text())
assert value["state"] in {"partial", "complete"}, value["state"]
assert value["processed_object_ids"] == [int(os.environ["SMOKE_OBJECT_ID"])]
assert value["api_requests_this_run"] + value["view_cache_hits_this_run"] == 1
print("smoke manifest: PASS")
PY

echo "[3/9] Full multi-view captions"
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --max-detections-per-object "$MAX_CAPTION_VIEWS" \
  --masking-option red_outline \
  --openai-image-detail high \
  2>&1 | tee "$LOG_DIR/02_caption_full.log"

"$CG_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OPENAI_CACHE"])
manifest = json.loads((root / "cfslam_openai_caption_manifest.json").read_text())
captions = json.loads((root / "cfslam_openai_captions.json").read_text())
assert manifest["complete"] is True
assert len(captions) == manifest["num_scene_objects"]
assert all(item["captions"] for item in captions)
print("caption objects:", len(captions))
print("caption requests this run:", manifest["api_requests_this_run"])
print("caption cache hits this run:", manifest["view_cache_hits_this_run"])
PY

echo "[4/9] Node refinement"
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode refine-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device "$DEVICE" \
  2>&1 | tee "$LOG_DIR/03_refine_nodes.log"

echo "[5/9] ConceptGraphs relation edges"
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode build-scenegraph \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device "$DEVICE" \
  2>&1 | tee "$LOG_DIR/04_build_scenegraph.log"

echo "[6/9] Detailed node JSON"
"$CG_PYTHON" "$CG_SCRIPT" \
  --mode generate-scenegraph-json \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device "$DEVICE" \
  2>&1 | tee "$LOG_DIR/05_generate_nodes.log"

echo "[7/9] Generic property/state extraction"
"$CG_PYTHON" "$OUTPUT_SCRIPT" extract-attributes \
  "${OPENAI_ATTRIBUTE_ARGS[@]}" \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --captions-file "$OPENAI_CACHE/cfslam_openai_captions.json" \
  --prompt-file "$ATTRIBUTE_PROMPT" \
  --output-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --cache-dir "$OPENAI_CACHE/scene_graph_attribute_cache" \
  --manifest-file "$OPENAI_CACHE/scene_graph_attributes_manifest.json" \
  2>&1 | tee "$LOG_DIR/06_extract_attributes.log"

# No remaining stage needs the credential in its environment.
cleanup_credential

echo "[8/9] Sparse scene-graph format"
"$CG_PYTHON" "$OUTPUT_SCRIPT" format \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --attributes-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --edges-file "$OPENAI_CACHE/cfslam_scenegraph_edges.pkl" \
  --output-json "$OPENAI_CACHE/scene_graph.json" \
  --output-repr "$OPENAI_CACHE/scene_graph.txt" \
  --manifest-file "$OPENAI_CACHE/scene_graph_format_manifest.json" \
  2>&1 | tee "$LOG_DIR/07_format_scenegraph.log"

echo "[9/9] Final validation"
"$CG_PYTHON" - <<'PY' 2>&1 | tee "$LOG_DIR/08_validate.log"
import json
import os
from pathlib import Path

root = Path(os.environ["OPENAI_CACHE"])
graph = json.loads((root / "scene_graph.json").read_text())
nodes = json.loads((root / "scene_graph_nodes.json").read_text())
manifest = json.loads((root / "scene_graph_format_manifest.json").read_text())
assert isinstance(graph, dict)
assert graph, "scene graph must contain at least one retained node"
assert len(graph) == len(nodes) == manifest["node_count"]
assert manifest["complete"] is True
for source, fields in graph.items():
    prefix, separator, suffix = source.rpartition("_")
    assert prefix and separator and suffix.isascii() and suffix.isdigit(), source
    assert set(fields).issubset({"property", "state", "relation"})
    assert all(isinstance(values, list) for values in fields.values())
    for relation in fields.get("relation", []):
        predicate, target = relation.split(" ", 1)
        assert predicate in {"ON", "INSIDE"}
        assert target in graph and target != source
print("nodes:", len(graph))
print("relations:", sum(len(v.get("relation", [])) for v in graph.values()))
print("scene_graph.json:", root / "scene_graph.json")
print("scene_graph.txt:", root / "scene_graph.txt")
print("validation: PASS")
PY

find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -exec chmod 600 {} +
echo "Scene graph complete: $OPENAI_CACHE/scene_graph.json"
