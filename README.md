# 3D Scene Graph：Video2Mesh + GroundingDINO + SAM2

本仓库是在 [ConceptGraphs](https://github.com/concept-graphs/concept-graphs)
基础上扩展的可复现视频场景图流水线。输入普通 RGB 视频，依次完成：

```text
视频抽帧
  → COLMAP 稀疏重建
  → GroundingDINO 开放词汇检测
  → SAM2 多帧实例跟踪
  → 2D mask 实例消歧与质量门控
  → 3D 融合
  → ConceptGraphs 对象地图
  → 多视角 2D/3D 关系图
  → scene_graph.json
```

当前实现使用 **SAM2.1 Hiera Tiny**，不是 SAM3。外部仓库、模型和输入数据均按
只读资源使用；每次实验写入独立的 run 目录。

## 1. 已验证的软件组合

| 组件 | 版本 |
|---|---|
| 操作系统 | Linux x86_64 |
| `svpp` | Python 3.11.15、PyTorch 2.1.1、CUDA 12.1 |
| `groundingdino` | Python 3.10.18、PyTorch 1.13.1、CUDA 11.7 |
| `sam2` | Python 3.10.18、PyTorch 2.5.1、CUDA 12.1 |
| COLMAP | 4.1 CPU build |
| FFmpeg | Conda 环境内版本 |

三套环境分开是有意设计：GroundingDINO、SAM2 和 ConceptGraphs 的已验证
PyTorch 依赖不同。更换 CUDA/PyTorch 组合后，应重新执行第 6 节的 preflight。

流水线固定并验证以下外部仓库提交：

| 仓库 | Commit |
|---|---|
| Video2Mesh | `3ed5ece2974594c26498676e1276f168e6db8962` |
| GroundingDINO | `856dde20aee659246248e20734ef9ba5214f5e44` |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |

## 2. 推荐目录结构

默认配置假设四个仓库互为同级目录：

```text
workspace/
├── 3d_scene_graph/
├── Video2Mesh/
├── GroundingDINO/
└── sam2/
```

模型默认放在本仓库的 `models/`，结果默认放在 `runs/`。两者都已加入
`.gitignore`，不会上传到远程仓库。[`.env`](.env) 会随代码提交，当前内容是
本机可运行的路径参考；在其他机器克隆后必须按照第 5 节修改。目录不同时无需
修改 YAML，只需修改 `.env`。

## 3. 克隆固定版本

```bash
mkdir -p workspace
cd workspace

git clone https://github.com/zhengjjjjie/3d_scene_graph.git
git clone https://github.com/Interstellar6/Video2Mesh.git
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/facebookresearch/sam2.git

git -C Video2Mesh checkout 3ed5ece2974594c26498676e1276f168e6db8962
git -C GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44
git -C sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Preflight 要求这三个外部仓库的已跟踪文件保持干净，以免不同机器使用了不同
实现。安装产生的未跟踪构建文件不影响检查。

## 4. 创建三套 Conda 环境

建议使用 Miniconda/Mambaforge。环境名应保持为
`svpp`、`groundingdino`、`sam2`；若修改环境名，需要同时设置
`CG_CONDA_ENVS_ROOT` 并修改
`conceptgraph/configs/video2mesh_pipeline.yaml` 中对应解释器名称。

```bash
cd workspace/3d_scene_graph

conda env create -f environment.yml
conda env create -f environments/groundingdino.yml
conda env create -f environments/sam2.yml
```

安装本仓库以及两个外部 Python 包：

```bash
conda run -n svpp python -m pip install -e .
conda run -n groundingdino python -m pip install --no-build-isolation -e ../GroundingDINO
conda run -n sam2 python -m pip install --no-build-isolation -e ../sam2
```

Video2Mesh 通过固定仓库的 `PYTHONPATH` 调用，不需要复制或安装到本仓库。

如果只运行原始 ConceptGraphs RGB-D 流程，而不是本文的视频流水线，可能还需要
上游项目的 `gradslam`、`chamferdist`、Grounded-SAM 或 LLaVA 依赖；它们不是
当前 Video2Mesh/SAM2 路径的必需项。

## 5. 下载模型并配置本机路径

在仓库根目录执行：

```bash
MODEL_ROOT="$PWD/models"
mkdir -p \
  "$MODEL_ROOT/GroundingDINO/weights" \
  "$MODEL_ROOT/sam2/checkpoints" \
  "$MODEL_ROOT/huggingface"

curl -fL \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "$MODEL_ROOT/GroundingDINO/weights/groundingdino_swint_ogc.pth"

curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o "$MODEL_ROOT/sam2/checkpoints/sam2.1_hiera_tiny.pt"

MODEL_ROOT="$MODEL_ROOT" conda run -n svpp python -c \
  'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="openai/clip-vit-base-patch16", local_dir=os.path.join(os.environ["MODEL_ROOT"], "huggingface", "clip-vit-base-patch16"))'
```

校验关键模型文件：

```bash
sha256sum \
  models/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  models/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  models/huggingface/clip-vit-base-patch16/pytorch_model.bin
```

期望 SHA-256：

```text
3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69  sam2.1_hiera_tiny.pt
ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0  pytorch_model.bin
```

### 配置随仓库提交的 `.env`

仓库已经包含并跟踪 [`.env`](.env)，其中保存的是当前验证机器的路径和非敏感
模型服务参数。克隆到其他机器后，需要根据本机目录布局修改其中的绝对路径；
无本机路径的配置模板见 [`.env.example`](.env.example)。

路径变量含义如下：

| 变量 | 应指向的路径 | 必要内容或用途 |
| --- | --- | --- |
| `CG_WORKSPACE_ROOT` | 四个代码仓库的共同父目录 | 其下应有 `3d_scene_graph/`、`Video2Mesh/`、`GroundingDINO/` 和 `sam2/` |
| `CG_MODEL_ROOT` | 模型权重根目录 | 其下应有 `GroundingDINO/weights/groundingdino_swint_ogc.pth`、`sam2/checkpoints/sam2.1_hiera_tiny.pt` 和 `huggingface/clip-vit-base-patch16/pytorch_model.bin` |
| `CG_OUTPUT_BASE` | 可写的实验输出根目录 | 每次运行会在其下创建 `<scene-id>/<run-id>/`；它不是输入数据目录 |
| `CG_DEPENDENCY_ROOT` | 可写的依赖下载和 bootstrap 缓存目录 | 三个环境均已安装完成时可以为空；其中不需要放输入视频或原始数据 |
| `CG_CONDA_ROOT` | Miniconda 或 Anaconda 安装根目录 | 其下必须存在 `bin/conda` |
| `CG_CONDA_ENVS_ROOT` | Conda 环境的共同父目录 | 其下应有 `svpp/`、`groundingdino/` 和 `sam2/`；对应解释器为各目录下的 `bin/python` |
| `CG_DEPS` | 旧版序列化 map 的可选兼容包目录 | 使用时应包含 `openai_py311/` 和 `mapping_py311/`；相关包已直接安装到 `svpp` 时可设为空值 |

其中 `svpp` 环境还需要提供 `bin/colmap` 和 `bin/ffprobe`。输入视频路径不写入
`.env`，而是在运行时通过 `--video /absolute/path/to/video.mp4` 传入；
`scene-id` 和 `run-id` 也是实验名称，不是目录变量。

其余非路径变量含义如下：

| 变量 | 含义 |
| --- | --- |
| `OPENAI_BASE_URL` | OpenAI-compatible Responses API 的 HTTPS 根地址 |
| `OPENAI_MODEL` | 节点 refinement、关系判断和属性生成使用的文本模型 |
| `OPENAI_VISION_MODEL` | 多视角对象 caption 使用的视觉模型 |
| `OPENAI_TIMEOUT` | 单次 API 请求的超时秒数 |
| `OPENAI_MAX_RETRIES` | SDK 层自动重试次数 |

`.env` 会上传到远程仓库，因此绝对不要加入 `OPENAI_API_KEY`、访问令牌、密码
或其他凭据。`run_scene_graph.sh` 默认隐藏读取 API key；也可以在仓库外创建
权限为 `600` 的 key 文件，并在运行前设置 `OPENAI_API_KEY_FILE`。

手动运行 Python 入口前加载 `.env`：

```bash
set -a
source .env
set +a
conda activate svpp
```

## 6. 运行环境预检

长时间运行之前先检查仓库版本、模型 hash、三个解释器、CUDA、COLMAP、FFmpeg
和输入视频：

```bash
VIDEO=/absolute/path/to/video.mp4

python -m conceptgraph.scripts.run_video2mesh_pipeline preflight \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video "$VIDEO"
```

成功时 JSON 输出中的 `ok` 为 `true`。也可以只生成完整阶段命令而不执行：

```bash
OUTPUT_BASE="${CG_OUTPUT_BASE:-$PWD/runs}"

python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video "$VIDEO" \
  --scene-id my_scene \
  --output-base "$OUTPUT_BASE" \
  --run-id dry_run_check \
  --dry-run
```

## 7. 从视频构建 ConceptGraphs Map

每次全新实验使用一个从未用过的 `run-id`：

```bash
export VIDEO=video_data_path
export SCENE_ID=my_scene
export RUN_ID=run_001
export OUTPUT_BASE="${CG_OUTPUT_BASE:-$PWD/runs}"

python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video "$VIDEO" \
  --scene-id "$SCENE_ID" \
  --output-base "$OUTPUT_BASE" \
  --run-id "$RUN_ID"
```

通用视频默认最多均匀选择 200 帧。可使用以下参数覆盖：

```text
--start-frame N --end-frame N --stride N --max-frames N
--queries-file /absolute/path/to/object_queries.txt
```

仓库中的 `bedroom_validation` profile 只对应原实验视频的源帧
`2709, 2733, ..., 3429`：

```bash
python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video "$VIDEO" \
  --scene-id bedroom_4_CmEIg9gMI74 \
  --output-base "$OUTPUT_BASE" \
  --run-id bedroom_validation_001 \
  --profile bedroom_validation
```

中断后只能对同一个显式 `run-id` 使用 `--resume`。Runner 会核对输入、
配置、commit 和阶段产物 hash；不匹配时拒绝混用旧结果。

Map 输出位置：

```text
<output-base>/<scene-id>/<run-id>/
├── v2m_project/
│   ├── masks/
│   │   ├── 2d_raw/       # SAM2 原始轨迹，只读输入
│   │   ├── 2d/           # 实例消歧后的观测 mask
│   │   └── 2d_fusion/    # 仅供 3D 融合使用的 mask
│   ├── simulator_assets/
│   │   ├── identity_quality_report.json
│   │   └── mask_track_quality_report.json
│   └── logs/conceptgraphs_video2mesh/
└── conceptgraphs/
    ├── full_pcd_video2mesh_colmap_sam2.pkl.gz
    └── full_pcd_video2mesh_colmap_sam2.conversion.json
```

实例消歧首先以 complete-link 合并同类别重复轨迹和高置信跨类别重复轨迹，
随后把稳定、被完整实例包含的同类别局部碎片挂接到该实例，但不会用碎片桥接
两个相互独立的完整实例。`identity_quality_report.json` 记录原始轨迹到
canonical instance 的映射、重复簇、碎片挂接、跨类别合并和未决冲突；未决
身份冲突会在 3D 融合前阻断运行。覆盖率、mask 面积波动等分割质量问题保留为
非阻断 warning，供人工复核。

## 8. 从 Map 构建 scene graph

`run_scene_graph.sh` 会自动读取仓库根目录的 `.env`。确保当前 shell 使用
`svpp`：

```bash
conda activate svpp

SCENE_ID=my_scene \
RUN_NAME=run_001 \
./run_scene_graph.sh
```

也可以显式指定任意可信 Map 和结果目录：

```bash
RUN_ROOT=/absolute/path/to/run_root \
MAP_FILE=/absolute/path/to/full_pcd_video2mesh_colmap_sam2.pkl.gz \
./run_scene_graph.sh
```

最终结果：

```text
<run-root>/scene_graph_openai/scene_graph.json
<run-root>/scene_graph_openai/scene_graph.txt
<run-root>/scene_graph_openai/scene_graph_nodes.json
<run-root>/scene_graph_openai/cfslam_multiview_relation_evidence.json
<run-root>/logs/
```

若旧 Map 内部的 `color_path` 来自另一挂载点，可临时设置
`MAP_PATH_REMAP_FROM` 和 `MAP_PATH_REMAP_TO`。脚本只在
`<run-root>/runtime_inputs/` 创建派生 Map，不修改源 pickle。新实验不需要路径
重映射。

## 9. 数据安全和复现约束

- 输入视频、外部仓库和模型权重不会被流水线改写。
- 新 run 默认拒绝已存在的非空目录；重新实验应更换 `run-id`。
- `--resume` 不是覆盖开关，只会复用 hash 完全匹配的阶段。
- 模型、数据、run 输出、secret 和自包含 HTML 报告均被 Git 忽略。
- `.env` 会被 Git 跟踪，只允许保存路径和非敏感运行参数，禁止保存 API key。
- Pickle 只能加载自己生成或来源可信的文件。
- COLMAP 稀疏重建尺度是任意尺度，坐标不能直接解释为米。
- 没有有效 3D 点但具有足够多视图的对象会保留为
  `geometry_type=multiview_2d`，其运行时 `bbox=None`。

## 10. 测试

```bash
conda activate svpp
bash -n run_scene_graph.sh
python -m pytest -q \
  tests/test_video2mesh_integration.py \
  tests/test_multiview_relations_v2.py
```

测试不调用模型 API，也不会修改原始视频或外部仓库。

## 11. 常见问题

- `conceptgraphs_python_matches_caller`：当前命令不是从 `svpp` 执行。运行
  `conda activate svpp`，或检查 `CG_CONDA_ENVS_ROOT`。

- 模型或仓库 hash/commit 不匹配：使用第 3、5 节固定的版本。不要通过关闭
  preflight 混用模型结果。

- `normalize_mask_tracks` 报输出已存在：空的 Video2Mesh 初始化目录可以复用；
  非空目录会被保护。全新实验换 `run-id`，中断恢复使用 `--resume`。

- API 的 model listing 返回 404：某些 OpenAI-compatible 服务不实现模型枚举；
  该项本身可忽略，但 smoke test 必须真正返回视觉 caption。确认
  `OPENAI_BASE_URL` 和模型 ID 与服务一致。

- 关系阶段遇到 `bbox=None`：当前实现会跳过没有 3D geometry 的对象进行 3D
  overlap，同时保留其多视角 2D 关系证据。请确认使用的是本仓库测试通过的
  版本。

## 项目来源

原始项目与论文：

- [ConceptGraphs project page](https://concept-graphs.github.io/)
- [ConceptGraphs paper](https://arxiv.org/abs/2309.16650)
- [上游仓库](https://github.com/concept-graphs/concept-graphs)

代码许可见 [LICENSE](LICENSE)。Video2Mesh、GroundingDINO、SAM2 和模型权重
分别遵循其原仓库/模型页面的许可。
