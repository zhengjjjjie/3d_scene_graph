# 当前视频 3D Scene Graph 流程与技术设计

本文记录仓库截至 2026-08-03 的当前实现，用于解释从普通 RGB 视频到
`scene_graph.json` 的完整数据流、阶段边界、核心算法、产物契约和已知局限。

运行参数的唯一事实来源是
[`conceptgraph/configs/video2mesh_pipeline.yaml`](../conceptgraph/configs/video2mesh_pipeline.yaml)；
安装、模型下载和命令示例以仓库根目录的
[`README.md`](../README.md) 为准。本文侧重技术方案，不重复完整安装步骤。

## 1. 目标与边界

当前流水线的目标是：

1. 从普通 RGB 视频恢复一组注册相机和 COLMAP 稀疏点云。
2. 通过 GroundingDINO 和 SAM2 获得跨帧实例 mask。
3. 对重复轨迹、碎片轨迹和跨类别同实例轨迹进行消歧。
4. 将多视角 2D mask 投影到 COLMAP 稀疏点云，得到点索引形式的 3D mask。
5. 转换为 ConceptGraphs `MapObjectList`。
6. 结合多视角 2D、尺度无关 3D 证据和语言模型，生成节点、属性及
   `ON`/`INSIDE` 关系。

当前实现明确不包含：

- SAM3；当前分割后端是 SAM2.1 Hiera Tiny。
- TSDF、体素占据或稠密 RGB-D 融合。
- 完整物体表面重建或实例 mesh 重建。
- 3D Gaussian 语义训练。
- 从 SAM2 原始 logits 进行真正的软概率融合。
- 自动拆分已经被上游合成一个 object ID 的两个物理实例。
- `LEFT_OF`、`NEAR`、`BEHIND` 等通用关系；最终稀疏图目前只保留
  `ON` 和 `INSIDE`。

## 2. 总体架构

```text
输入 RGB 视频
  │
  ├─ Video2Mesh：抽帧
  ├─ COLMAP：相机位姿 + 稀疏点云
  ├─ GroundingDINO：开放词汇候选与实例锚点
  ├─ ConceptGraphs：prompt 归一化
  ├─ SAM2：逐实例视频 mask 跟踪
  ├─ ConceptGraphs：实例消歧、碎片挂接、质量门
  ├─ Video2Mesh：2D mask → 稀疏点级 3D mask
  ├─ ConceptGraphs adapter：MapObjectList + CLIP 特征
  │
  └─ run_scene_graph.sh
       ├─ 多视角视觉 caption
       ├─ 节点语义 refinement
       ├─ 多视角 2D/尺度无关 3D 关系候选
       ├─ 语言模型关系判定
       ├─ property/state 抽取
       └─ scene_graph.json / scene_graph.txt
```

流水线分成两个相互独立的阶段：

- **阶段 A：视频到对象 Map**  
  入口是 `python -m conceptgraph.scripts.run_video2mesh_pipeline run`。
- **阶段 B：对象 Map 到 scene graph**  
  入口是 `./run_scene_graph.sh`。

阶段 B 只读取阶段 A 的可信 `.pkl.gz` Map，不会重新运行检测、分割或 3D 融合。

## 3. 组件、环境与模型

| 组件 | Conda 环境 | 当前职责 |
|---|---|---|
| ConceptGraphs | `svpp` | 调度、归一化、消歧、质量门、Map 转换、scene graph |
| Video2Mesh | `groundingdino` | 抽帧、COLMAP 接口、质量报告、2D→3D mask 融合 |
| GroundingDINO | `groundingdino` | 开放词汇候选检测 |
| SAM2 | `sam2` | 视频实例 mask 传播 |
| COLMAP/FFmpeg | `svpp` 中的二进制 | 稀疏重建、视频探测 |
| CLIP ViT-B/16 | `svpp` | Map 中的对象图像/文本特征 |
| OpenAI-compatible API | 外部服务 | caption、节点 refinement、关系和属性抽取 |

三套 Python 环境是有意隔离的。Runner 会检查当前调用 Python 是否等于配置中的
`conceptgraphs_python`，避免从错误环境运行。

外部仓库固定提交为：

| 仓库 | Commit |
|---|---|
| Video2Mesh | `3ed5ece2974594c26498676e1276f168e6db8962` |
| GroundingDINO | `856dde20aee659246248e20734ef9ba5214f5e44` |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |

Preflight 同时验证仓库提交、解释器、CUDA 能力、COLMAP CLI、模型路径和关键模型
SHA-256。外部仓库、模型和输入视频作为只读输入使用。

## 4. 输入、运行隔离与输出根目录

阶段 A 的最小输入为：

- 一个可读取的 RGB 视频；
- `scene-id`；
- 一个全新的 `run-id`；
- 仓库内的 pipeline YAML；
- GroundingDINO、SAM2 和 CLIP 权重。

每次运行写入独立目录：

```text
<output-base>/<scene-id>/<run-id>/
```

新运行拒绝复用已存在的非空目录。`--resume` 不是覆盖开关，只复用满足以下条件的
已完成阶段：

- 输入视频 hash 相同；
- 配置和命令 hash 相同；
- 外部仓库提交与模型 hash 相同；
- 阶段 marker 存在；
- 阶段产物 hash 与 marker 完全一致。

任何无法验证的残留产物都会阻止安全 resume。

## 5. 阶段 A：视频到 ConceptGraphs Map

### 5.1 入口命令

```bash
python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/video.mp4 \
  --scene-id <scene-id> \
  --output-base "$CG_OUTPUT_BASE" \
  --run-id <new-run-id>
```

正式执行前可使用 `preflight` 和 `--dry-run`：

```bash
python -m conceptgraph.scripts.run_video2mesh_pipeline preflight \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/video.mp4

python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/video.mp4 \
  --scene-id <scene-id> \
  --output-base "$CG_OUTPUT_BASE" \
  --run-id <dry-run-id> \
  --dry-run
```

`--dry-run` 只生成精确命令计划，不创建 run 目录。

### 5.2 Runner 的 12 个阶段

| 顺序 | 阶段名 | 环境 | 主要输入 | 主要输出 |
|---:|---|---|---|---|
| 1 | `init` | `groundingdino` | 视频、scene ID | `v2m_project/manifest.json` |
| 2 | `extract_frames` | `groundingdino` | 视频、帧策略 | `scene/frames/`、`frames_manifest.json` |
| 3 | `run_colmap` | Python + `svpp` COLMAP | 抽取帧 | 相机、稀疏点云、COLMAP 报告 |
| 4 | `reconstruction_readiness` | `groundingdino` | 相机、点云 | 重建 readiness 报告 |
| 5 | `discover_object_prompts` | `groundingdino` | 图像、GroundingDINO | 原始检测、prompt、labels |
| 6 | `normalize_object_prompts` | `svpp` | GroundingDINO 结果 | canonical tracking prompts |
| 7 | `track_masks` | `sam2` | 图像、prompts、SAM2 | `masks/2d_raw/` |
| 8 | `normalize_mask_tracks` | `svpp` | 原始 SAM2 轨迹 | `masks/2d/`、`masks/2d_fusion/` |
| 9 | `identity_quality_report` | `svpp` | canonical 轨迹 | 实例质量报告与阻断门 |
| 10 | `mask_track_quality_report` | `groundingdino` | canonical 2D mask | 跟踪质量诊断 |
| 11 | `fuse_masks` | `groundingdino` | 相机、点云、`2d_fusion` | 点索引形式的 3D mask |
| 12 | `finalize_fusion_manifest` | `svpp` | 3D mask manifest | 下游观测路径与融合路径分离 |

12 个阶段成功后，入口脚本还会：

1. 严格验证 Video2Mesh project。
2. 加载本地 CLIP。
3. 转换为 ConceptGraphs Map。
4. 验证 Map、转换报告和 project 输入 hash 的一致性。

### 5.3 抽帧与 bedroom profile

通用视频默认在全视频中均匀选择最多 200 帧。

当前 `bedroom_validation` profile 固定选择：

```text
源帧：2709, 2733, ..., 3429
stride：24
总帧数：31
```

显式的 `--start-frame`、`--end-frame`、`--stride` 和 `--max-frames`
会覆盖 profile。

### 5.4 COLMAP 稀疏重建

当前设置：

- 相机模型：`PINHOLE`；
- 单相机内参；
- exhaustive matcher；
- CPU 特征提取/匹配；
- 稀疏重建；
- focal scale 初值为 `1.2`；
- 优化焦距，不优化主点和额外畸变参数。

输出包括：

```text
scene/cameras/camera_info.json
scene/reconstruction/point_cloud.ply
external/colmap/colmap_run_report.json
```

COLMAP 坐标尺度是任意尺度，不能直接解释为米。后续方法只在需要时使用相对场景
尺度，或明确记录 `coordinate_units=colmap_arbitrary`。

### 5.5 GroundingDINO 检测与 prompt 归一化

当前 GroundingDINO 主要参数：

| 参数 | 值 |
|---|---:|
| anchor frame count | 5 |
| box threshold | 0.28 |
| text threshold | 0.25 |
| max objects | 40 |
| NMS IoU | 0.65 |
| instance IoU | 0.18 |
| instance center distance | 0.75 |
| max instances per label | 8 |
| merge bed parts | false |
| single instance labels | 空 |

Runner 要求保留 raw detections。ConceptGraphs 随后执行对象粒度归一化：

- `bed`、`mattress`、`blanket`、`quilt`、`comforter`、`bedding`、
  `bed sheet`、`bed skirt`、`headboard` 被视为 bed family。
- 选择最高置信 bed-family detection 作为 `cg_bed` 跟踪 prompt，并保留所有
  bed-family source detection ID。
- `pillow`、`lamp`、`nightstand` 作为独立实例种子，最多从 2 个锚点帧、
  每类保留 8 个种子。
- 同一帧、同类别、bbox IoU 小于 `0.20` 的两个锚点被记录为
  `forbidden_merge_pairs`，防止两个空间分离的真实实例在跟踪后误合并。
- 其他类别沿用 GroundingDINO 已生成的对象候选。

归一化不会自动修复错误词元。例如 `##board` 不等于 `headboard`，不会自动进入
bed family。

### 5.6 SAM2 视频实例跟踪

每个 canonical prompt 独立运行 SAM2.1 Hiera Tiny：

- 设备：CUDA；
- autocast：开启；
- 视频帧和推理状态可 offload 到 CPU；
- SAM2 原始 mask 阈值为 `0.0`；
- 结果写入 `masks/2d_raw/<object-id>/<frame-id>.png`。

当前持久化 mask 是二值 `0/255` PNG，不保留 SAM2 的软 logits。原始轨迹目录是
不可变输入，后续所有合并都写入新的目录。

## 6. 实例消歧与轨迹归一化

实例消歧分成“严格重复合并”和“局部碎片挂接”两层。

### 6.1 轨迹对统计量

对于两个轨迹 \(A,B\)，只在二者 mask 都非空的共享帧计算：

- 共享非空帧数；
- 对较短轨迹的共享覆盖率；
- 每帧 mask IoU；
- median IoU；
- 高 IoU 帧比例；
- \(A\) 在 \(B\) 中的方向性 containment；
- \(B\) 在 \(A\) 中的方向性 containment；
- 两者的中位面积和面积比。

empty/empty 帧不进入 IoU 分母。

### 6.2 严格重复轨迹合并

同类别重复轨迹的默认条件为：

```text
共享非空帧 >= 5
较短轨迹共享覆盖率 >= 0.60
median IoU >= 0.85
至少 80% 共享帧的 IoU >= 0.75
```

跨类别轨迹只有在更严格条件下才允许合并：

```text
共享非空帧 >= 5
较短轨迹共享覆盖率 >= 0.60
median IoU >= 0.90
至少 80% 共享帧的 IoU >= 0.85
```

这用于处理 `counter` 与 `nightstand` 等语义名称不同、实际 mask 几乎完全相同的
情况。跨类别合并始终写入质量 warning 和 lineage，不静默隐藏。

聚类使用 complete-link：

> 一个新轨迹只有与簇内每个成员都满足重复条件时才能加入。

因此 `A≈B`、`B≈C`、但 `A≉C` 不会通过传递闭包把三个实例全部合并。

canonical track 按以下顺序确定：

1. 有效帧最多；
2. mask 面积变化最稳定；
3. detection confidence 最高；
4. object ID 字典序。

严格重复轨迹合并时，同帧 mask 取 union，并完整保留
`source_object_ids`、`source_labels` 和 source prompt/detection lineage。

### 6.3 同实例局部碎片挂接

严格重复合并后，对剩余同类别轨迹判断“小碎片是否属于一个更大的完整轨迹”。

默认条件：

```text
median containment >= 0.95
至少 75% 共享帧的 containment >= 0.90
碎片/父轨迹 median area ratio <= 0.75
共享非空帧 >= 5
较短轨迹共享覆盖率 >= 0.60
```

挂接方向必须是严格从较小轨迹到较大轨迹。若同一碎片同时满足两个独立完整实例，
未选中的有效候选会保留为 unresolved conflict，而不是静默选择。

碎片挂接不会用局部碎片扩大 canonical mask：

- canonical 已有非空 mask 的帧保持不变；
- 只在 canonical 缺帧或空帧时用碎片补齐。

这一策略避免把枕头局部误分割再次 union 到完整枕头上。

### 6.4 输出和 lineage

归一化输出：

```text
masks/2d/
├── <canonical-object-id>/<frame-id>.png
└── tracking_manifest.json
```

manifest 记录：

- raw track count；
- canonical track count；
- duplicate clusters；
- fragment candidates 和最终 attachments；
- raw source ID 到 canonical ID 的映射；
- pair metrics；
- forbidden overlap conflicts；
- unresolved candidates；
- 每个 canonical object 的所有 source IDs 和 source labels。

## 7. 质量检查与阻断策略

### 7.1 Identity quality gate

`identity_quality_report` 在 3D 融合前执行，输出：

```text
simulator_assets/identity_quality_report.json
```

以下问题作为 error，并令质量门失败：

- 缺失 identity resolution；
- 未解决的重复或碎片候选；
- forbidden pair 在多帧中仍高度重叠；
- 一个 source track 被分配给多个 canonical instance；
- 归一化 mask 文件缺失；
- 空轨迹。

以下问题作为 warning，不阻断 3D 融合：

- 帧数少于 3；
- 轨迹覆盖率低于 0.70；
- mask area CV 大于 1.0；
- 相邻有效帧 mask 面积变化超过 4 倍；
- 跨类别 identity merge。

当前配置 `fail_on_unresolved: true`。CLI 实际依据报告的整体 `ok` 状态退出，因此
任何上述 error 都会阻止进入 3D 融合。

### 7.2 Video2Mesh mask quality report

第二份报告位于：

```text
simulator_assets/mask_track_quality_report.json
```

它检查覆盖率、面积 CV 和 bbox center jump。当前 bbox center jump 是图像坐标
指标，对明显相机运动敏感，因此主要用于诊断。只要报告成功生成，该阶段本身不会
因为 warning 自动阻止后续融合。

`identity_quality_report.ok=true` 表示身份冲突已解决，不等价于
`quality_clean=true`，更不等价于每个 SAM2 mask 都具有正确语义。

## 8. 融合专用 2D mask

`masks/2d_fusion` 首先是 `masks/2d` 的逐文件副本。

当前唯一的结构化 carve 规则是 pillow/bed：

1. pillow 在 bed 中的 median containment 至少为 `0.80`；
2. pillow/bed 最大面积比不超过 `0.35`；
3. pillow mask 膨胀 2 像素；
4. 只从 bed 的融合专用 mask 中扣除膨胀后的 pillow；
5. pillow mask 和 `masks/2d` 中的 bed 观测 mask 不变。

这样做的目的，是在 3D 融合前减少 bed 与 pillow 对同一稀疏点的竞争，同时保留
原始观测用于 caption 和多视角关系判断。

## 9. 多视角 2D mask 到 3D mask

### 9.1 3D mask 的实际定义

当前所谓 3D mask 是全局 COLMAP 稀疏点云的点索引集合：

\[
\mathcal M_o^{3D}=\{i\mid X_i\text{ 被最终分配给对象 }o\}
\]

它不是体素 mask，也不是完整表面。

每个对象输出：

```text
masks/3d/<object-id>/
├── point_indices.npy
├── point_indices.json
├── point_probabilities.npz
└── point_probabilities_summary.json
```

### 9.2 3D 点投影

对 COLMAP 点 \(X_i\) 和帧 \(f\)，使用 world-to-camera 外参：

\[
X^c_{i,f}=T_f[X_i,1]^T
\]

使用 PINHOLE 内参投影：

\[
u=\left\lfloor f_xX/Z+c_x \right\rfloor,\qquad
v=\left\lfloor f_yY/Z+c_y \right\rfloor
\]

只保留 \(Z>0\) 且投影位于图像范围内的点。投影坐标使用 `floor` 到单个像素，
没有双线性采样、点 footprint 或 mask 边缘距离权重。

### 9.3 稀疏 Z-buffer 遮挡过滤

对投影到同一像素的稀疏点，取最近深度 \(z_{\min}\)。点满足：

\[
z_i\le z_{\min}+\max(0.05,\;0.03z_{\min})
\]

才视为可见。

当前参数：

```yaml
occlusion_filter: true
depth_tolerance: 0.05
relative_depth_tolerance: 0.03
```

该 Z-buffer 来自稀疏点云自身，不是稠密深度图。若某个前景表面没有 COLMAP 点，
它无法为后方点提供遮挡。

### 9.4 Probability 模式在当前输入下的真实行为

当前配置：

```yaml
mode: probability
min_probability: 0.5
min_votes: 1
```

Video2Mesh 将灰度值转换为：

\[
p_{o,f,i}=\frac{M_{o,f}(u_i,v_i)}{255}
\]

并累计：

\[
V_{o,i}=\sum_f
\mathbf 1[\text{visible}_{f,i}]
\mathbf 1[p_{o,f,i}\ge0.5]
\]

候选点条件：

\[
V_{o,i}\ge1
\]

由于当前持久化 SAM2 mask 严格为 `0/255`，所以
\(p\in\{0,1\}\)。因此当前 probability mode 在选择层面退化为二值投票：

> 一个可见视角命中一次，就足以使该点成为该对象的候选点。

当前没有：

- 对可见但落在 mask 外的帧累计负证据；
- 用 `positive_views / visible_views` 作为支持率；
- 使用 SAM2 原始 soft logits；
- 按视角、深度、入射角或轨迹质量加权。

`probability_mean` 只在正概率观测之间求均值。对二值 mask，非空对象的
`probability_mean` 和 `probability_max` 通常均为 1。

### 9.5 对象间逐点互斥

当前 `exclusive_objects: true`，每个 3D 点最多属于一个对象。

如果一个点同时属于多个候选对象，按以下顺序选 winner：

1. 可选的人工 category priority；
2. 最大 projected probability；
3. 正命中视角数；
4. 角色优先级：`thing > surface_or_fixture > unknown > background`；
5. 候选点更少的对象；
6. object ID 字典序。

当前没有配置人工 category priority，且二值 mask 使最大 probability 通常同为
1，所以正命中视角数通常是主要判据。

互斥分配保证下游不会出现两个对象共享同一 point index，但也可能显著压缩小物体、
背景或分割不稳定对象的 3D 几何。

### 9.6 Fusion manifest finalization

实际融合输入是：

```text
masks/2d_fusion
```

融合结束后，`finalize_fusion_manifest`：

- 将 `fusion_input_mask_root` 记录为 `masks/2d_fusion`；
- 将下游 `mask_root` 恢复为 `masks/2d`；
- 在 frame score 中同时保留 fusion mask 与 observation mask 路径。

因此：

- 3D point indices 来自 carve 后的融合 mask；
- caption、Map 中的多视角 mask 和关系证据使用未 carve 的 canonical observation
  mask。

## 10. Video2Mesh project 到 ConceptGraphs Map

Adapter 只通过磁盘产物契约读取 Video2Mesh，不导入或修改其源码。

### 10.1 严格校验

转换前验证：

- 相机和帧 manifest 完整；
- 每个 frame ID 唯一且有注册相机；
- PLY 点为有限的 Nx3 坐标并带 RGB；
- 2D mask 是与图像同尺寸的二值 `0/255`；
- point indices 是唯一、合法的一维整数；
- 对象 3D mask 之间互斥；
- probability sidecar 与 point indices 对齐；
- manifest 中声明的路径指向 canonical 文件；
- 所有输入树的 SHA-256。

### 10.2 对象保留规则

当前规则：

| 对象类型 | 最低有效 caption views |
|---|---:|
| 前景 | 2 |
| 背景 | 1 |

背景类别仅为：

```text
wall, floor, ceiling
```

若对象没有 3D 点：

- 至少 3 个非空 2D views 时，保留为 `geometry_type=multiview_2d`；
- 否则拒绝该对象。

有点对象标记为 `geometry_type=colmap_3d`。其 3D bbox 是所选稀疏点的轴对齐
bounding box，没有连通域过滤、离群点清理或鲁棒分位数裁剪。

### 10.3 CLIP 特征

对 caption-valid views：

1. 根据 mask bbox 裁剪 RGB 图像，默认 padding 20 像素；
2. 使用本地 `clip-vit-base-patch16` 提取每个 crop 的图像特征；
3. 归一化后求平均，再归一化为对象 `clip_ft`；
4. 对 class name 提取并缓存文本特征 `text_ft`。

Map 最终包含：

- `objects`：前景对象；
- `bg_objects`：wall/floor/ceiling；
- 多视角 mask、bbox、frame ID 和原视频帧号；
- 稀疏 3D points、colors、point indices 和 AABB；
- CLIP image/text features；
- source object/detection/prompt lineage；
- runner fingerprint、输入 hash 和外部版本 provenance。

主要输出：

```text
conceptgraphs/full_pcd_video2mesh_colmap_sam2.pkl.gz
conceptgraphs/full_pcd_video2mesh_colmap_sam2.conversion.json
```

## 11. 阶段 B：ConceptGraphs Map 到 Scene Graph

### 11.1 入口

```bash
RUN_ROOT=/absolute/path/to/<run-root> \
RELATION_MODE=multiview-2d-3d \
./run_scene_graph.sh
```

`run_scene_graph.sh` 默认：

```text
MAP_FILE=<run-root>/conceptgraphs/full_pcd_video2mesh_colmap_sam2.pkl.gz
OPENAI_CACHE=<run-root>/scene_graph_openai
RELATION_MODE=multiview-2d-3d
MAX_RELATION_CANDIDATES=100
MAX_CAPTION_VIEWS=4
DEVICE=cuda:0
```

API key 通过隐藏输入、已存在的环境变量，或权限为 `600` 的外部 key 文件读取。
key 不写入日志、`.env` 或结果 manifest。

### 11.2 九个 scene graph 阶段

| 顺序 | 阶段 | 技术作用 |
|---:|---|---|
| 1 | API preflight | 检查 API 客户端、base URL 和模型配置 |
| 2 | one-view smoke test | 对一个对象做一次视觉 caption，验证真实视觉请求 |
| 3 | full multi-view captions | 每对象最多 4 个视角，red outline，高细节视觉请求 |
| 4 | node refinement | 聚合多视角 caption，生成 object tag、summary、possible tags |
| 5 | relation edges | 生成候选对、计算证据、请求关系判定 |
| 6 | detailed node JSON | 输出完整节点 JSON |
| 7 | property/state extraction | 从节点和 caption 抽取通用 property/state |
| 8 | sparse format | 合并节点、属性和边，生成最终稀疏图 |
| 9 | final validation | 校验节点数、目标存在性和 relation schema |

caption、node refinement、关系和属性请求都有独立 cache/manifest。缓存身份包含
输入、模型、prompt、mask/camera hash 等信息，只有完全匹配时才复用。

## 12. 默认多视角 2D/3D 关系方法

`run_scene_graph.sh` 默认使用 `multiview-2d-3d`，而不是旧的
`legacy-3d-mst`。

### 12.1 候选对输入

候选构造同时读取：

- canonical 多视角 2D mask；
- COLMAP 稀疏对象点；
- 每帧相机位姿；
- foreground/background 标记；
- parent candidate 和 parent object lineage；
- refinement 后的对象 tag 和 caption。

所有对象对都会先计算证据。background/background 对被禁止进入关系候选，避免
wall/floor/ceiling 之间产生大量无意义边。

### 12.2 相机 pose 聚类

不能把相邻视频帧视为完全独立证据。共享帧先按相机 pose 聚类：

- 视角方向差不超过 5°；
- 相机中心距离不超过场景对角线的 3%。

每个 pose cluster 内先以多数票和中位数聚合，再跨 cluster 统计支持度，减少连续
相似视角的重复计票。

### 12.3 多视角 `ON` 证据

对方向 `source on target`，每个共享帧检查：

- source bbox 中心是否位于 target bbox 中心上方；
- 水平投影是否有足够重叠；
- source 底部与 target 顶部的归一化 gap 是否接近接触。

默认关键阈值：

```text
至少 3 个 pose clusters
above vote >= 0.75
support vote >= 0.50
median horizontal support >= 0.25
normalized gap ∈ [-0.25, 0.10]
```

包含背景对象的 pair 更严格：

```text
至少 5 个 pose clusters
support vote >= 0.60
```

### 12.4 多视角 `INSIDE` 证据

对 `small in large`，每帧检查：

- small bbox 至少 90% 位于 large bbox；
- small mask 至少 80% 位于 large mask 的填充凸包；
- small 中心位于 large 凸包；
- small/large 面积比不超过 0.50。

跨 pose cluster 的支持率至少为 0.60 才成为候选。

### 12.5 尺度无关 3D 邻近证据

由于 COLMAP 没有米制尺度，3D 邻近半径不使用固定米数：

1. 分别估计两个对象内部点的中位最近邻间距；
2. 取较大的间距；
3. 乘以 `2.5` 得到 pair-specific radius；
4. 计算两个方向上落入该半径的点比例；
5. 任一方向比例至少为 `0.02` 时，提供 3D candidate evidence。

每个对象最多采样 5,000 点。

### 12.6 Parent hint

prompt normalization 和 pillow/bed carve 产生的 parent lineage 只用于提高候选召回：

```text
parent_hint_recall_only
```

它不会强制语言模型输出 `ON` 或 `INSIDE`。

### 12.7 语言模型关系判定

只有满足至少一种 2D、3D 或 parent-hint 召回条件的 pair 才调用文本模型。模型
接收：

- 两个节点的 tag、caption、可能类别；
- geometry type；
- bbox center/extent（若有 3D）；
- point count；
- 压缩后的 2D/3D 定量证据。

关系请求不发送图像。允许结果严格限定为：

```text
a on b
b on a
a in b
b in a
none of these
```

若候选超过 `MAX_RELATION_CANDIDATES=100`，流水线输出诊断并停止，不会静默截断。
混合关系模式不使用旧版 MST 强制连通。

## 13. 最终输出

完整 run 目录的关键结构：

```text
<run-root>/
├── v2m_project/
│   ├── manifest.json
│   ├── scene/
│   │   ├── frames/
│   │   ├── frames_manifest.json
│   │   ├── cameras/camera_info.json
│   │   └── reconstruction/point_cloud.ply
│   ├── masks/
│   │   ├── object_prompts_groundingdino.json
│   │   ├── object_prompts_normalized.json
│   │   ├── object_labels.json
│   │   ├── 2d_raw/
│   │   ├── 2d/
│   │   ├── 2d_fusion/
│   │   └── 3d/
│   ├── simulator_assets/
│   │   ├── reconstruction_readiness_report.json
│   │   ├── identity_quality_report.json
│   │   └── mask_track_quality_report.json
│   └── logs/conceptgraphs_video2mesh/
├── conceptgraphs/
│   ├── full_pcd_video2mesh_colmap_sam2.pkl.gz
│   └── full_pcd_video2mesh_colmap_sam2.conversion.json
├── scene_graph_openai/
│   ├── cfslam_openai_captions.json
│   ├── cfslam_gpt-4_responses.pkl
│   ├── cfslam_multiview_relation_evidence.json
│   ├── cfslam_object_relations.json
│   ├── cfslam_scenegraph_edges.pkl
│   ├── scene_graph_nodes.json
│   ├── scene_graph_attributes.json
│   ├── scene_graph.json
│   └── scene_graph.txt
└── logs/
```

最终 `scene_graph.json`：

- key 是带实例后缀的节点名；
- 每个节点最多包含 `property`、`state`、`relation`；
- `relation` 当前只允许 `ON <target>` 或 `INSIDE <target>`；
- 每个 target 必须存在且不能指向自身。

## 14. 当前 bedroom 验证基线

`bedroom_v2m_20260731_identity_v2` 是当前实现的一次参考运行，不是硬编码测试输入。

阶段 A 的关键结果：

```text
31 个注册视角
38 条原始 SAM2 轨迹
21 个 canonical 实例
10 个重复轨迹簇
2 个 fragment attachments
1 个跨类别 merge
0 个 unresolved candidates
0 个 forbidden overlap conflicts
18 个非阻断质量 warnings
```

目标实例数：

```text
lamp：2
nightstand：2
pillow：3
```

3D 融合：

```text
COLMAP 稀疏点：12,517
2D mask records：465
互斥前 object-point memberships：17,931
最终唯一归属点：12,302
被互斥删除的 memberships：5,629
```

转换结果：

```text
17 个前景对象
3 个背景对象
1 个仅有单帧、无 3D 点的 floor 候选被拒绝
```

该结果通过 identity gate 和 Map 验证，但 `quality_clean=false`。例如一个
nightstand、两个 pillow、window 等轨迹仍存在明显面积跳变。因此“实例数量正确”
不能替代 mask 语义与形状质量检查。

## 15. 已知问题与技术风险

### 15.1 分割语义错误不会被 3D 融合修复

如果一个 `##board` mask 实际来自 pillow，3D 融合只会把该错误 mask 命中的点
赋给 `##board`。当前没有图像-类别一致性模型参与 pre-fusion quality gate。

### 15.2 已合并实例无法在 3D 阶段拆分

如果两个 window 或两个 wall art 在上游已经共用一个 canonical ID，融合只会生成
一个联合 point set。类别粒度和实例粒度必须在检测、prompt 或 track 阶段解决。

### 15.3 当前 probability fusion 实际是二值 union-like voting

`min_votes=1` 且没有负证据意味着单帧假阳性可能进入 3D candidate。soft logits、
可见视角分母和支持率尚未进入决策。

### 15.4 稀疏 Z-buffer 对遮挡建模有限

只有投影到同一离散像素的 COLMAP 点能够互相比较深度。纹理弱表面没有稀疏点时，
不会阻挡后方点。

### 15.5 互斥分配可能显著改变几何

当多个对象共享候选点时，winner-takes-all 会压缩小对象、背景或低覆盖对象。
最终 AABB 和中心可能由很少的残余点决定，进而影响 3D relation evidence。

### 15.6 没有 3D 几何清理

当前没有：

- 点云连通域过滤；
- statistical/radius outlier removal；
- 主体 component 选择；
- robust/quantile bbox；
- object-specific surface completion。

少量远端错误点可能显著改变 AABB。

### 15.7 关系模型只看到摘要证据

关系语言模型不接收原图，只接收 caption 和定量证据。错误 caption、错误实例
lineage、稀疏 3D 点或不稳定 2D mask 都可能导致错误关系。

### 15.8 COLMAP 任意尺度

3D 点坐标不能解释为米。当前邻近关系采用内部点间距归一化，但融合的绝对
`depth_tolerance=0.05` 仍会受重建尺度影响，relative tolerance 只能部分缓解。

## 16. 复现与安全原则

- 原始视频、模型和外部仓库不被修改。
- 每次完整实验使用新的 `run-id`。
- `--resume` 只接受经过 hash 验证的阶段。
- 派生 Map 路径重映射只写入 `<run-root>/runtime_inputs/`。
- `.env` 只能保存路径和非敏感参数，不能保存 API key。
- API key 通过隐藏输入或仓库外权限为 `600` 的文件提供。
- 所有 pickle 只能加载自己生成或来源可信的文件。
- 质量报告、runner manifest、conversion report 和 API cache manifest 应与
  最终结果一起保留，用于追踪参数与输入。

## 17. 主要代码入口

| 功能 | 文件 |
|---|---|
| Pipeline CLI | `conceptgraph/scripts/run_video2mesh_pipeline.py` |
| Runner、preflight、resume | `conceptgraph/integrations/video2mesh/runner.py` |
| Prompt/track 归一化、实例消歧、质量门 | `conceptgraph/integrations/video2mesh/object_normalization.py` |
| Video2Mesh project → Map | `conceptgraph/integrations/video2mesh/adapter.py` |
| Scene graph 主逻辑 | `conceptgraph/scenegraph/build_scenegraph_cfslam.py` |
| 多视角 2D/3D 关系证据 | `conceptgraph/scenegraph/multiview_relations.py` |
| 属性与最终格式 | `conceptgraph/scenegraph/scenegraph_output.py` |
| Scene graph 一键脚本 | `run_scene_graph.sh` |
| 当前配置 | `conceptgraph/configs/video2mesh_pipeline.yaml` |
