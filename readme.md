# Bedroom RGB frames → OpenAI ConceptGraphs Scene Graph

本文件记录从已生成的 13-object post-map 开始，使用 OpenAI-compatible API 完成：

1. 单视角视觉 smoke test；
2. 全量多视角 caption；
3. 节点语义 refinement；
4. ConceptGraphs `on/in` 关系构图；
5. 通用 `property/state` 抽取；
6. 生成截图所示 sparse scene-graph 字典。

所有命令使用 `svpp`，日志和结果均写入：

~~~text
/data2/zhengjie/code/concept-graphs/outputs/bedroom_4_CmEIg9gMI74
~~~

`GPTPrompt.py` 保持仓库原样。`property/state` 不使用物体类别映射或 bedroom 专用规则，
而是通过可替换的外部 prompt 对每个保留节点推断。formatter 只进行通用 schema、ID 和
ConceptGraphs 关系方向转换，不合成 room 节点或 `INSIDE bedroom_*` 关系。

## 0. 最简单的完整执行方式

只运行下面两条命令：

~~~bash
cd /data2/zhengjie/code/concept-graphs
bash run_scene_graph.sh
~~~

看到 `只粘贴 API key，然后按回车:` 后，只粘贴 key 值本身并按回车。脚本会在
`svpp` 中自动依次执行视觉 smoke、全量 caption、refinement、关系、通用属性和最终
格式化；中断后再次运行同一命令会复用兼容缓存。

URL、模型、map 和输出目录仍可在执行前用环境变量覆盖，例如：

~~~bash
OPENAI_MODEL=my-text-model \
OPENAI_VISION_MODEL=my-vision-model \
bash run_scene_graph.sh
~~~

换场景时同时指定 map、输出名和一个有效的 smoke object ID：

~~~bash
MAP_FILE=/path/to/scene_post.pkl.gz \
RUN_NAME=my_scene \
SMOKE_OBJECT_ID=0 \
bash run_scene_graph.sh
~~~

## 1. 安全写入可替换的 API 凭证文件

不要把 API key 写进本文件、源码、命令参数或 shell history。
如果某个 key 曾经明文出现在聊天或日志中，建议在服务方控制台轮换。若明确接受继续
使用该 key 的风险，也仍应只通过下述私有文件传入；程序参数传的是文件路径，不是 key。

~~~bash
set -Eeuo pipefail
set +x
unset OPENAI_LOG
export OPENAI_API_KEY_FILE="${OPENAI_API_KEY_FILE:-/data2/zhengjie/data/concept_graphs/secrets/openai_api_key}"
install -d -m 700 "$(dirname "$OPENAI_API_KEY_FILE")"
umask 077
KEY_TMP=
cleanup_key_input() {
  unset OPENAI_API_KEY
  if [[ -n "${KEY_TMP:-}" ]]; then
    rm -f -- "$KEY_TMP"
  fi
}
trap cleanup_key_input EXIT
trap 'cleanup_key_input; exit 130' HUP INT TERM
read -rsp "OPENAI_API_KEY: " OPENAI_API_KEY; echo
KEY_TMP=$(mktemp "${OPENAI_API_KEY_FILE}.XXXXXX")
chmod 600 "$KEY_TMP"
printf '%s' "$OPENAI_API_KEY" > "$KEY_TMP"
mv -fT -- "$KEY_TMP" "$OPENAI_API_KEY_FILE"
KEY_TMP=
cleanup_key_input
trap - EXIT HUP INT TERM
~~~

检查文件只报告权限和是否非空，不打印内容：

~~~bash
test -s "$OPENAI_API_KEY_FILE"
test "$(stat -c '%a' "$OPENAI_API_KEY_FILE")" = 600
~~~

## 2. 公共环境变量

以下命令需要在 Bash 中执行，以便 `pipefail` 和 `PIPESTATUS` 行为明确。

~~~bash
set -Eeuo pipefail
set +x
umask 077
unset OPENAI_LOG

export CG_ROOT=/data2/zhengjie/code/concept-graphs
export CG_DEPS=/data2/zhengjie/data/concept_graphs/python_packages
export CG_PYTHON=/data2/zhengjie/miniconda3/envs/svpp/bin/python
export CG_SCRIPT="$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py"
export OUTPUT_SCRIPT="$CG_ROOT/conceptgraph/scenegraph/scenegraph_output.py"
export ATTRIBUTE_PROMPT="$CG_ROOT/conceptgraph/scenegraph/prompts/scene_graph_attributes.txt"

export PYTHONPATH="$CG_DEPS/openai_py311:$CG_DEPS/mapping_py311:$CG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data2/zhengjie/miniconda3/envs/svpp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 只修改这一组参数即可切换兼容服务、模型或凭证文件，无需改源码。
export OPENAI_API_KEY_FILE="${OPENAI_API_KEY_FILE:-/data2/zhengjie/data/concept_graphs/secrets/openai_api_key}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.autodl.art/api/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
export OPENAI_VISION_MODEL="${OPENAI_VISION_MODEL:-$OPENAI_MODEL}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}"

export MAP_FILE=/data2/zhengjie/data/concept_graphs/outputs/bedroom_4_CmEIg9gMI74/pcd_saves/full_pcd_sam3_clip_overlap_maskconf0.95_simsum1.2_dbscan.1_sam3_clip_post.pkl.gz
export RUN_ROOT="$CG_ROOT/outputs/bedroom_4_CmEIg9gMI74"
export SMOKE_CACHE="$RUN_ROOT/smoke"
export OPENAI_CACHE="$RUN_ROOT/scene_graph_openai"
export LOG_DIR="$RUN_ROOT/logs"

install -d -m 700 "$RUN_ROOT" "$SMOKE_CACHE" "$OPENAI_CACHE" "$LOG_DIR"
test -s "$OPENAI_API_KEY_FILE"
test "$(stat -c '%a' "$OPENAI_API_KEY_FILE")" = 600
unset OPENAI_API_KEY

# build_scenegraph_cfslam.py 的所有 OpenAI 阶段复用这一参数数组。
OPENAI_COMMON_ARGS=(
  --openai-api-key-file "$OPENAI_API_KEY_FILE"
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-model "$OPENAI_MODEL"
  --openai-vision-model "$OPENAI_VISION_MODEL"
  --openai-timeout "$OPENAI_TIMEOUT"
  --openai-max-retries "$OPENAI_MAX_RETRIES"
)
~~~

## 3. 本地预检与模型列表

~~~bash
"$CG_PYTHON" - <<'PY' 2>&1 | tee "$LOG_DIR/00_preflight.log"
import os
import platform
from pathlib import Path
from conceptgraph.scenegraph.build_scenegraph_cfslam import make_openai_client

required = [
    Path(os.environ["MAP_FILE"]),
    Path(os.environ["CG_SCRIPT"]),
    Path(os.environ["OUTPUT_SCRIPT"]),
    Path(os.environ["ATTRIBUTE_PROMPT"]),
]
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

print("python:", platform.python_version())
print("base_url:", os.environ["OPENAI_BASE_URL"])
print("text_model:", os.environ["OPENAI_MODEL"])
print("vision_model:", os.environ["OPENAI_VISION_MODEL"])

client = make_openai_client(
    api_key_file=os.environ["OPENAI_API_KEY_FILE"],
    base_url=os.environ["OPENAI_BASE_URL"],
    max_retries=int(os.environ["OPENAI_MAX_RETRIES"]),
)
print("available_models:")
try:
    for model in client.models.list(timeout=float(os.environ["OPENAI_TIMEOUT"])).data:
        print(" -", model.id)
except Exception as exc:
    # Some compatible providers implement /responses but not /models. Keep
    # this diagnostic non-fatal and do not print the SDK exception body.
    print(" - model listing unavailable:", type(exc).__name__)
PY
~~~

如果服务返回的真实模型 ID 不同，只修改 `OPENAI_MODEL` / `OPENAI_VISION_MODEL` 环境变量，
不要修改源码。模型列表存在不等于一定支持图片输入，因此仍必须执行下一步 smoke test。

## 4. 单对象视觉 smoke test

该步骤只处理 object 5 的一个视角，输出 partial manifest，不会被后续 refinement 当作
完整 caption。smoke 使用独立 cache，不污染正式结果。

~~~bash
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$SMOKE_CACHE" \
  --mapfile "$MAP_FILE" \
  --annot-inds 5 \
  --max-detections-per-object 1 \
  --masking-option red_outline \
  --openai-image-detail high \
  2>&1 | tee "$LOG_DIR/01_caption_smoke.log"

"$CG_PYTHON" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["SMOKE_CACHE"]) / "cfslam_openai_caption_manifest.json"
value = json.loads(path.read_text())
assert value["state"] == "partial"
assert value["api_requests_this_run"] + value["view_cache_hits_this_run"] == 1
print("smoke manifest: OK")
PY
~~~

## 5. 全量 OpenAI 视觉 caption

当前 post-map 有 13 个对象，默认每对象 4 个时间分散视角，首次完整运行最多 52 次视觉
请求。逐视角 cache 会在中断后复用，Base64 图片不会写盘。

~~~bash
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --max-detections-per-object 4 \
  --masking-option red_outline \
  --openai-image-detail high \
  2>&1 | tee "$LOG_DIR/02_caption_full.log"

"$CG_PYTHON" - <<'PY'
import json, os
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
~~~

## 6. 节点 caption refinement

该阶段保留仓库原始 `GPTPrompt.py` 语义。新增的 request identity/cache 只负责断点续跑和
防止旧模型、旧 caption 或旧 map 的 response 混入。

~~~bash
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode refine-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0 \
  2>&1 | tee "$LOG_DIR/03_refine_nodes.log"
~~~

## 7. 构造 ConceptGraphs 关系边

此阶段按官方 baseline 使用单向 `overlap[i,j]` 和原始 minimum-spanning-tree 权重行为，
同时使用原始 object ID 对齐 response、pruned map 和 edge。局部 overlap 通过数值等价的
FAISS 兼容实现计算，不依赖 `gradslam`。relation cache 会核对 query、prompt、模型、map
和 response 哈希。

~~~bash
"$CG_PYTHON" "$CG_SCRIPT" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode build-scenegraph \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0 \
  2>&1 | tee "$LOG_DIR/04_build_scenegraph.log"
~~~

## 8. 生成详细 node list

~~~bash
"$CG_PYTHON" "$CG_SCRIPT" \
  --mode generate-scenegraph-json \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0 \
  2>&1 | tee "$LOG_DIR/05_generate_nodes.log"
~~~

详细节点保存在 `scene_graph_nodes.json`，包含 ID、bbox、tag 和 caption。

## 9. 通用属性/状态抽取

`scenegraph_output.py` 不包含物体类别、房间类型或固定属性 ontology。语义由
`--prompt-file` 指定的外部文本控制；代码只校验 `property/state` 是字符串数组并统一为
uppercase snake case。状态无证据时 prompt 要求返回空数组。

~~~bash
"$CG_PYTHON" "$OUTPUT_SCRIPT" extract-attributes \
  --openai-api-key-file "$OPENAI_API_KEY_FILE" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --captions-file "$OPENAI_CACHE/cfslam_openai_captions.json" \
  --prompt-file "$ATTRIBUTE_PROMPT" \
  --output-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --cache-dir "$OPENAI_CACHE/scene_graph_attribute_cache" \
  --manifest-file "$OPENAI_CACHE/scene_graph_attributes_manifest.json" \
  --openai-model "$OPENAI_MODEL" \
  --openai-timeout "$OPENAI_TIMEOUT" \
  --openai-max-retries "$OPENAI_MAX_RETRIES" \
  2>&1 | tee "$LOG_DIR/06_extract_attributes.log"
~~~

## 10. 生成截图式 sparse scene graph

~~~bash
"$CG_PYTHON" "$OUTPUT_SCRIPT" format \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --attributes-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --edges-file "$OPENAI_CACHE/cfslam_scenegraph_edges.pkl" \
  --output-json "$OPENAI_CACHE/scene_graph.json" \
  --output-repr "$OPENAI_CACHE/scene_graph.txt" \
  --manifest-file "$OPENAI_CACHE/scene_graph_format_manifest.json" \
  2>&1 | tee "$LOG_DIR/07_format_scenegraph.log"
~~~

`scene_graph.json` 是标准 JSON（双引号）；`scene_graph.txt` 是与示例截图一致的 Python
字典显示形式（单引号）。两者内容相同。每个 key 为规范化 `object_tag` 加原始 object ID，
例如 `wall art` → `wall_art_6`。空的 `property/state/relation` 字段被省略。

结构关系只作以下确定性方向转换：

~~~text
a on b  → A: "ON B"
b on a  → B: "ON A"
a in b  → A: "INSIDE B"
b in a  → B: "INSIDE A"
~~~

不会根据目录名自动添加 `bedroom`、room node 或 `INSIDE` 关系。

## 11. 最终严格校验

~~~bash
"$CG_PYTHON" - <<'PY' 2>&1 | tee "$LOG_DIR/08_validate.log"
import json, os, re
from pathlib import Path

root = Path(os.environ["OPENAI_CACHE"])
graph = json.loads((root / "scene_graph.json").read_text())
nodes = json.loads((root / "scene_graph_nodes.json").read_text())
manifest = json.loads((root / "scene_graph_format_manifest.json").read_text())

assert isinstance(graph, dict)
assert len(graph) == len(nodes) == manifest["node_count"]
assert manifest["complete"] is True
for source, fields in graph.items():
    assert re.fullmatch(r"[a-z0-9_]+_[0-9]+", source), source
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
~~~

## 12. 可选：清除凭证

Python 只读取参数指向的凭证文件，不会修改或删除它。若后续还要运行，可保留该 0600
文件；要换 key，重新执行第 1 节即可原子替换。若不再使用，可手动删除：

~~~bash
rm -f "$OPENAI_API_KEY_FILE"
~~~

## 13. 主要结果与日志

~~~text
outputs/bedroom_4_CmEIg9gMI74/
├── logs/
│   ├── 00_preflight.log
│   ├── 01_caption_smoke.log
│   ├── 02_caption_full.log
│   ├── 03_refine_nodes.log
│   ├── 04_build_scenegraph.log
│   ├── 05_generate_nodes.log
│   ├── 06_extract_attributes.log
│   ├── 07_format_scenegraph.log
│   └── 08_validate.log
├── smoke/
└── scene_graph_openai/
    ├── cfslam_openai_captions.json
    ├── cfslam_openai_caption_manifest.json
    ├── cfslam_gpt-4_responses/
    ├── cfslam_object_relations.json
    ├── cfslam_scenegraph_edges.pkl
    ├── scene_graph_nodes.json
    ├── scene_graph_attributes.json
    ├── scene_graph.json
    ├── scene_graph.txt
    └── scene_graph_format_manifest.json
~~~

图片 crop、caption prompt、refinement prompt 和属性 prompt 会发送到配置的 AutoDL API。
`store=false` 不等同于代理或上游服务零日志、零留存，运行前应确认服务方的数据处理策略。
