# ConceptGraphs Scene Graph 生成 Pipeline

本文结合当前仓库 **main@93277a0** 与论文
[ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)，
总结从 posed RGB-D 数据生成 3D scene graph 的完整流程，并说明只有 RGB 视频时如何补齐深度、相机内参和位姿。

## 1. 核心结论

ConceptGraphs 不是从单帧图像直接预测 scene graph，而是分成两个主要阶段：

1. 增量构建 object-centric 3D map；
2. 在最终 object map 上离线生成节点描述和对象关系。

论文/原仓库的语言链路概括如下；当前可执行版本在第 7 节将 LLaVA/GPT-4 替换为
参数化的 OpenAI-compatible 视觉/文本模型，同时保留相同的节点与关系阶段。

~~~text
带位姿 RGB-D
→ 2D mask / detection + CLIP 特征
→ mask 反投影为 3D observation
→ 跨帧关联与对象融合
→ 3D object map
→ LLaVA 多视角描述
→ GPT-4 汇总节点标签
→ 空间邻接图 + MST
→ GPT-4 判断对象关系
→ 3D scene graph
~~~

当前仓库使用离线 batch pipeline。虽然 3D object map 按帧增量更新，但 2D detection、object mapping、caption 和 edge generation 是分阶段执行的。

## 2. 输入要求

每一帧需要：

\[
I_t = \langle I_t^{rgb}, I_t^{depth}, K, T_t \rangle
\]

其中：

- RGB 图像；
- 与 RGB 像素对齐的 Z-depth；
- 相机内参 \(K=(f_x,f_y,c_x,c_y)\)；
- 4×4 camera-to-world pose \(T_t\)。

当前代码不估计相机轨迹，而是直接读取 dataset poses。因此，只有 RGB 图像和独立 depth 仍然不够；相机内参和逐帧位姿同样是必需输入。

## 3. 生成 2D object observations

主入口：

~~~bash
python scripts/generate_gsa_results.py ...
~~~

仓库支持论文中的两种前端。

### 3.1 ConceptGraphs（CG）

使用 class-agnostic SAM：

~~~text
RGB → SAM automatic mask generator → class-agnostic masks
~~~

命令中使用 class_set=none。

### 3.2 ConceptGraphs-Detector（CG-D）

~~~text
RGB
→ RAM/Tag2Text 生成当前帧候选类别
→ GroundingDINO 检测 bounding boxes
→ NMS
→ bounding box prompt SAM
→ instance masks
~~~

wall、floor 和 ceiling 可作为背景类别单独处理。

### 3.3 区域特征

对每个 mask 裁剪图像区域，并使用 OpenCLIP ViT-H/14 提取：

- image feature；
- detection class 对应的 text feature；
- mask、2D bbox、confidence 和 class id。

每帧输出：

~~~text
gsa_detections_<variant>/<frame>.pkl.gz
~~~

pickle 中主要包含：

~~~text
xyxy
confidence
class_id
mask
classes
image_crops
image_feats
text_feats
~~~

## 4. 2D mask 提升为 3D observation

核心函数：

~~~text
conceptgraph/slam/utils.py::gobs_to_detection_list()
~~~

处理步骤：

1. 根据 mask 面积、confidence、bbox 面积过滤低质量 detection；
2. 从较大的 enclosing mask 中减去内部前景 mask；
3. 使用 depth 和相机内参反投影；
4. 使用 camera-to-world pose 变换到全局坐标；
5. 体素降采样；
6. 使用 DBSCAN 保留主要点云簇；
7. 计算 3D oriented 或 axis-aligned bounding box。

反投影公式为：

\[
x = (u-c_x)d/f_x,\qquad
y = (v-c_y)d/f_y,\qquad
z = d
\]

每个 3D observation 包含：

~~~text
3D point cloud
3D bounding box
CLIP image/text feature
image_idx
color_path
mask
2D bbox
confidence
class
pixel_area
n_points
~~~

论文与代码的默认 voxel size 均为 0.025 m。

## 5. 跨帧 object association

对当前帧的 \(M\) 个 observations 与已有 \(N\) 个 map objects 计算 \(M\times N\) 相似度矩阵。

### 5.1 论文中的相似度

几何相似度：

\[
\phi_{\mathrm{geo}}(i,j)
=
\frac{
|\{p\in P_i:\operatorname{NN}(p,P_j)<\delta_{\mathrm{nn}}\}|
}{|P_i|}
\]

它表示新 observation 中有多少比例的点，能在已有 object 点云中找到距离小于阈值的最近邻。

语义相似度：

\[
\phi_{\mathrm{sem}}(i,j)
=
\frac{\cos(f_i,f_j)+1}{2}
\]

总分：

\[
\phi(i,j)=\phi_{\mathrm{geo}}(i,j)+\phi_{\mathrm{sem}}(i,j)
\]

论文参数：

- nearest-neighbor threshold：2.5 cm；
- association threshold：1.1。

### 5.2 当前代码实现

推荐配置：

~~~text
spatial_sim_type=overlap
match_method=sim_sum
~~~

几何部分：

1. 用 3D bbox IoU 排除完全不相交的 pair；
2. 用 FAISS 最近邻计算新 observation 点云相对 map object 的 overlap ratio。

视觉部分使用原始 CLIP cosine similarity，没有执行论文中的 \((\cos+1)/2\) 映射。

实际聚合公式：

\[
S=(1+\mathrm{phys\_bias})S_{\mathrm{geo}}
 +(1-\mathrm{phys\_bias})S_{\mathrm{visual}}
\]

README 推荐 phys_bias=0，因此仍为等权相加，但阈值不可直接与论文的 1.1 比较；README 示例通常使用 sim_threshold=1.2。

每个 detection 独立选择最高分 object：

- 最大相似度低于阈值：创建新 object；
- 否则：融合进 argmax object。

当前实现不是一对一 Hungarian assignment。同一帧的多个 detections 可以被并入同一个 map object。

## 6. Object fusion 与 post-processing

匹配成功后：

- 对点云求并集；
- voxel downsample；
- 重新计算 3D bbox；
- image/text feature 按 observation 数量加权平均并重新归一化；
- 累加图像路径、mask、confidence 和 detection history。

CG-D 中，wall、floor、ceiling 不经过普通关联，而是按类别直接融合成背景 object。

序列结束后执行：

~~~text
DBSCAN denoise
→ 删除点数或观测次数不足的 object
→ 根据点云 overlap + image similarity + text similarity 合并重复 object
~~~

输出：

~~~text
pcd_saves/full_pcd_<variant>_<suffix>.pkl.gz
pcd_saves/full_pcd_<variant>_<suffix>_post.pkl.gz
~~~

推荐使用带 **_post** 后缀的结果进入 scene graph 阶段。

## 7. 生成 scene graph 节点语义

依次运行：

~~~bash
python conceptgraph/scenegraph/build_scenegraph_cfslam.py \
  --mode extract-node-captions ...

python conceptgraph/scenegraph/build_scenegraph_cfslam.py \
  --mode refine-node-captions ...
~~~

### 7.1 OpenAI 视觉多视角 caption

当前修改后的代码对每个 object：

1. 过滤缺图、无效 bbox、过小 mask 和低 mask/bbox fill 的 observation；
2. 同一视频帧只保留质量最高的 observation；
3. 沿时间序列分箱，默认选择最多 4 个时间分散视角；
4. 动态扩展 crop，并按 `masking_option` 用红色轮廓、黑背景或原图标记目标；
5. 在内存中编码 JPEG data URL，送入可配置的 OpenAI-compatible Responses 视觉模型；
6. 按 map、图像 payload、模型、URL 和 prompt 指纹逐视角缓存。

由此得到同一 object 的多个粗 caption。历史 `cfslam_llava_captions.json` 不会被创建或
覆盖；完整结果写入 `cfslam_openai_captions.json`。

### 7.2 OpenAI-compatible caption refinement

文本模型使用仓库原始 `GPTPrompt.py`，将多视角 caption 汇总为：

~~~json
{
  "summary": "...",
  "possible_tags": ["..."],
  "object_tag": "..."
}
~~~

互相冲突或无法识别的 object 会被标记为 invalid。URL、文本模型、超时、重试和私有
key 文件路径均为运行参数，不绑定具体兼容服务。

随后删除：

- invalid/fail 节点；
- 没有 caption response 的节点；
- 观测次数小于 min_views_per_object 的节点。

带语言属性的节点保存到：

~~~text
sg_cache/map/scene_map_cfslam_pruned.pkl.gz
~~~

## 8. 生成 scene graph edges

执行：

~~~bash
python conceptgraph/scenegraph/build_scenegraph_cfslam.py \
  --mode build-scenegraph ...
~~~

当前实现：

1. 计算对象两两的点云 nearest-neighbor overlap；
2. bbox 完全不相交的 pair 被排除；
3. 保留 overlap > 0.01 的 pair；
4. 形成稀疏邻接图并计算 connected components；
5. 对每个连通分量调用 SciPy minimum_spanning_tree；
6. 按官方代码使用单向 `overlap[i,j]` 作为 SciPy minimum spanning tree 权重，
   对每条 MST edge 调用配置的文本模型判断关系。

LLM 输入：

~~~text
object_tag
3D bbox center
3D bbox extent
~~~

关系类别被限制为：

~~~text
a on b
b on a
a in b
b in a
none of these
~~~

配置的文本模型同时输出 reason。none of these 不进入最终 edge list。

最终表示：

~~~text
Node = 3D geometry + fused feature + object tag/caption
Edge = source object + target object + spatial relation
~~~

## 9. Scene graph 输出文件

~~~text
sg_cache/
├── cfslam_openai_captions.json
├── cfslam_openai_caption_manifest.json
├── cfslam_gpt-4_responses/<id>.json
├── map/scene_map_cfslam_pruned.pkl.gz
├── cfslam_object_relation_queries.json
├── cfslam_object_relations.json
├── cfslam_scenegraph_edges.pkl
├── scene_graph_nodes.json
├── scene_graph_attributes.json
├── scene_graph.json
└── scene_graph.txt
~~~

注意：

- `generate-scenegraph-json` 先生成详细 `scene_graph_nodes.json`；
- 通用 attribute 阶段由外部 prompt 生成 `property/state`，不使用类别硬编码；
- formatter 将节点、属性和 `cfslam_scenegraph_edges.pkl` 合并为最终 sparse
  `scene_graph.json` / `scene_graph.txt`，其中包含 `ON` / `INSIDE` 关系。

## 10. 论文与当前代码的主要差异

| 环节 | 论文 | 当前代码 |
|---|---|---|
| 语义相似度 | \((\cos+1)/2\) | 原始 cosine |
| 关联阈值 | 1.1 | README 推荐 1.2 |
| 代表视角 | 最佳 10 视角，按无噪点贡献数 | 默认 4 个时间分散视角，分箱后按可见质量选择 |
| 候选边 | 3D bbox IoU | 点云 NN overlap；bbox IoU 仅用于排除 |
| 关系输入 | caption + 3D location | object_tag + bbox center/extent |
| 关系词表 | 论文描述为可扩展 | 代码固定为 on/in/none |
| 图输出 | 逻辑上统一的 \(M=(O,E)\) | 保留详细中间文件，并额外输出合并后的 sparse dict |

实现注意事项：

1. 当前 baseline 保留官方的单向 overlap 和原始 minimum-spanning-tree 权重行为；
   仅在 prune、response 和 edge 文件之间保留原 object ID，避免过滤后 ID 错位。
2. mapping 默认配置写的是 match_method=sep_thresh，但当前 aggregate_similarities() 只实现 sim_sum；运行时必须像 README 一样显式覆盖。
3. 当前 streamlined detection 的输出目录和 Hydra 配置未与主 mapping 自动接通；复现论文时应使用经典入口。

## 11. 只有 RGB 视频时如何获得深度

只有单目 RGB 无法唯一恢复真实绝对尺度，只能通过传感器、多视图几何或学习模型估计深度。对于 ConceptGraphs，还必须同时补齐 K 和 camera-to-world pose。

### 11.1 推荐：MapAnything

[MapAnything](https://github.com/facebookresearch/map-anything) 可以从一组 RGB 帧联合输出：

~~~python
pred["depth_z"]       # 相机光轴方向的 metric Z-depth
pred["intrinsics"]    # 3x3 K
pred["camera_poses"]  # OpenCV camera-to-world, 4x4
pred["conf"]          # per-pixel confidence
~~~

这些字段与 ConceptGraphs 的输入接口最接近。

推荐链路：

~~~text
RGB frames
→ 抽帧并保持足够视角重叠
→ MapAnything
→ depth_z + K + camera_poses
→ 转为 depth PNG + dataconfig + traj.txt
→ ConceptGraphs
~~~

学习模型输出的 metric scale 仍是估计值，不等价于深度传感器测量。最好使用场景中一个已知长度校正整体尺度。

### 11.2 长视频：Metric Video Depth Anything + pose estimation

[Metric Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) 适合生成时序一致的 metric video depth：

~~~bash
python3 run.py \
  --input_video input.mp4 \
  --output_dir outputs \
  --encoder vitb \
  --metric \
  --save_npz
~~~

读取：

~~~python
depths = np.load("outputs/input_depths.npz")["depths"]
~~~

该模型不输出 pose，因此还需要 COLMAP、visual SLAM 或其他多视图重建方法。

单目 SfM/SLAM 的 translation 通常只有任意尺度。将其与 metric depth 结合时，必须：

1. 估计统一 scale \(s\)；
2. 同时缩放 depth 和 pose translation；
3. 不能只缩放 depth。

[COLMAP](https://colmap.readthedocs.io/en/latest/tutorial.html) 也可以通过 SfM+MVS 同时生成相互一致的 camera poses 和 dense depth，但单目结果通常仍需要真实尺度锚点。

### 11.3 如果可以重新采集

最稳妥的是直接使用：

- Intel RealSense；
- Azure Kinect；
- iPhone/iPad LiDAR + Record3D；
- 已标定双目相机。

这些设备能直接提供同步、米制、像素对齐的 RGB-D。传感器深度通常比单目预测更适合跨帧 3D object fusion。

## 12. 将预测深度转为仓库格式

最简单的是使用 Replica-like 布局：

~~~text
ROOT/SCENE/
├── results/
│   ├── frame000000.jpg
│   ├── depth000000.png
│   ├── frame000001.jpg
│   └── depth000001.png
└── traj.txt
~~~

将 meter depth 保存成 16-bit millimeter PNG：

~~~python
depth_m[~valid_mask] = 0
depth_mm = np.clip(
    np.rint(depth_m * 1000.0), 0, 65535
).astype(np.uint16)

cv2.imwrite("depth000000.png", depth_mm)
~~~

dataconfig：

~~~yaml
dataset_name: replica

camera_params:
  image_height: H
  image_width: W
  fx: ...
  fy: ...
  cx: ...
  cy: ...
  png_depth_scale: 1000.0
  crop_edge: 0
~~~

traj.txt 每帧一行，共 16 个 row-major float，表示 4×4 camera-to-world pose。

必须保证：

- depth 是 Z-depth，而不是 inverse depth、disparity 或 ray distance；
- 不要把彩色 depth visualization 当作原始 depth；
- 无效像素保存为 0；
- RGB、depth、pose 数量与顺序严格一致；
- RGB 和 depth 分辨率、视场及像素坐标对齐；
- pose translation 和 depth 使用同一尺度，建议米；
- 视频来自固定焦距相机时使用同一个 K；
- 若使用 COLMAP 的 world-to-camera extrinsic，需要先求逆得到 camera-to-world。

## 13. 推荐执行路径

~~~text
短/中等静态视频：
RGB frames → MapAnything → depth_z + K + cam2world → ConceptGraphs

较长视频：
RGB video → Metric Video Depth Anything
          + SfM/SLAM pose
          → metric scale alignment
          → ConceptGraphs

可重新采集：
RGB-D sensor / LiDAR / stereo → aligned RGB-D + K + pose → ConceptGraphs
~~~

## 14. 当前仓库主入口

~~~text
conceptgraph/scripts/generate_gsa_results.py
→ conceptgraph/slam/cfslam_pipeline_batch.py
→ conceptgraph/scenegraph/build_scenegraph_cfslam.py --mode extract-node-captions
→ conceptgraph/scenegraph/build_scenegraph_cfslam.py --mode refine-node-captions
→ conceptgraph/scenegraph/build_scenegraph_cfslam.py --mode build-scenegraph
→ conceptgraph/scenegraph/build_scenegraph_cfslam.py --mode generate-scenegraph-json
→ conceptgraph/scenegraph/scenegraph_output.py extract-attributes
→ conceptgraph/scenegraph/scenegraph_output.py format
~~~

相关文档和代码：

- [README.md](README.md)
- [2D detection](conceptgraph/scripts/generate_gsa_results.py)
- [3D mapping](conceptgraph/slam/cfslam_pipeline_batch.py)
- [Mapping utilities](conceptgraph/slam/utils.py)
- [Association](conceptgraph/slam/mapping.py)
- [Scene graph generation](conceptgraph/scenegraph/build_scenegraph_cfslam.py)
- [Dataset loaders](conceptgraph/dataset/datasets_common.py)
- [ConceptGraphs paper](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)
- [MapAnything](https://github.com/facebookresearch/map-anything)
- [Metric Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [COLMAP](https://colmap.readthedocs.io/en/latest/tutorial.html)

## 15. 当前 bedroom 数据的本地实测状态（2026-07-15）

本节记录实际路径、正式派生运行和验证结果。没有修改 31 张原始视频帧或模型权重。
RGB-D、detections 和 object map 派生数据位于
`/data2/zhengjie/data/concept_graphs/outputs`；本次 OpenAI 日志和最终 scene graph 按用户
要求位于 `/data2/zhengjie/code/concept-graphs/outputs`。为实现可恢复的 OpenAI 视觉与
通用 sparse 输出，本节后半部分记录了有限的 ConceptGraphs 源码扩展。

### 15.1 路径与运行时边界

~~~text
ConceptGraphs:
  /data2/zhengjie/code/concept-graphs

输入 RGB（31 张，1280x720）:
  /data2/zhengjie/code/concept-graphs/bedroom_4_CmEIg9gMI74/images

MapAnything 源码:
  /data2/zhengjie/data/concept_graphs/map-anything
MapAnything 权重:
  /data2/zhengjie/data/concept_graphs/models/map-anything/model.safetensors

SAM3 native 权重:
  /data2/zhengjie/data/TAB/sam3/sam3.pt
SAM3 Hugging Face safetensors:
  /data2/zhengjie/data/TAB/sam3/model.safetensors
SAM3 源码:
  /data2/zhengjie/data/concept_graphs/sam3

CLIP:
  /data2/zhengjie/data/huggingface/clip-vit-base-patch16

隔离 Python 依赖:
  /data2/zhengjie/data/concept_graphs/python_packages/vision
  /data2/zhengjie/data/concept_graphs/python_packages/mapping_py311
TorchHub 缓存:
  /data2/zhengjie/data/concept_graphs/torch
~~~

没有向 `svpp` 或 `tab` Conda 环境安装、升级或卸载包。MapAnything、SAM3 和 CLIP
使用 `/data2/zhengjie/miniconda3/envs/tab/bin/python`（Torch 2.4.1+cu121）；
ConceptGraphs mapping 使用 `/data2/zhengjie/miniconda3/envs/svpp/bin/python`
（Torch 2.1.1+cu121）。额外 Python 包均通过 `PYTHONPATH` 从上述隔离目录读取。

### 15.2 MapAnything 实测

本地模型可成功加载：

~~~text
参数量:       1,228,491,222
加载时间:     约 35.9 s
模型显存:     约 4.62 GiB
~~~

MapAnything checkpoint 已包含完整 DINOv2 encoder 权重，不需要再下载
4.23 GB 的 `dinov2_vitg14_pretrain.pth`。初始化时必须增加：

~~~text
model.encoder.uses_torch_hub=false
+model.encoder.torch_hub_pretrained=false
~~~

其中第二项前面的 `+` 是 Hydra 添加新字段的语法。`TORCH_HOME` 必须指向：

~~~bash
export TORCH_HOME=/data2/zhengjie/data/concept_graphs/torch
~~~

三帧 Images-only 冒烟测试成功：

~~~text
输入预处理尺寸: 518x294 (W x H)
推理时间:       约 3.88 s / 3 frames
峰值显存:       约 7.18 GiB
有效深度比例:   约 96%--97%
深度范围:       约 0.67--4.82 m
~~~

输出包含 `depth_z`、`depth_along_ray`、`intrinsics`、`camera_poses`、
`conf` 和 `mask`，pose 是 OpenCV/RDF 坐标系下的 camera-to-world 4x4 矩阵。

固定相机 COLMAP 解给出的原始内参为：

~~~text
fx = fy = 628.4598283557966
cx = 640
cy = 360
~~~

经过 MapAnything 的共同 crop/resize 后，每帧输入 K 一致：

~~~text
fx = fy = 256.62109375
cx = 258.70416259765625
cy = 146.70416259765625
~~~

需要注意：给定 K 只作为模型条件，MapAnything 仍会预测自己的 ray directions，
所以输出 `intrinsics` 不会原样等于输入 K。ConceptGraphs 的 Replica-like loader
只支持一个固定 K。为了使固定 K、depth 和反投影严格一致，推荐用给定 K 生成
单位相机射线 `r`，再由 MapAnything 的沿射线深度计算：

\[
P_{cam}=d_{ray}r,\qquad Z=d_{ray}r_z
\]

把该 Z 保存为 uint16 millimeter PNG，并在 dataconfig 中使用上面的固定预处理 K。

31 帧正式派生结果：

~~~text
输出分辨率:              518x294
MapAnything 推理时间:    6.47 s / 31 frames
峰值显存:                8.789 GiB
有效深度像素:            92.76%--97.32%
相机轨迹长度:            3.4569 m
相邻帧对称 NN 中位误差:  各帧对的中位数 2.72 cm，最大 3.69 cm
10 cm 内点重叠率:        中位数 93.25%
~~~

输出位于：

~~~text
/data2/zhengjie/data/concept_graphs/outputs/bedroom_4_CmEIg9gMI74/
├── results/frame000000.jpg ... frame000030.jpg
├── results/depth000000.png ... depth000030.png
├── traj.txt
├── dataconfig.yaml
├── geometry_manifest.json
└── .geometry_complete
~~~

31 张 RGB、depth、pose 数量一致；depth PNG 是 uint16 millimeter Z-depth；pose
旋转矩阵行列式接近 1，最大正交误差为 $2.26\times10^{-7}$。

### 15.3 SAM3 class-agnostic masks 实测

Native `sam3.pt` 可离线加载，参数量约 840.5 M、模型显存约 3.33 GiB。
但官方 native API 没有现成的 `SamAutomaticMaskGenerator`；它主要提供文本、点和框提示。

本地 Hugging Face `model.safetensors` 可以配合 Transformers `mask-generation`
管线生成 dense point-grid masks，但不能直接调用
`AutoModelForMaskGeneration.from_pretrained()`：SAM3 完整视频 checkpoint 带有嵌套前缀，
直接加载会造成大量 missing/unexpected keys。正确的 tracker 映射是：

~~~text
detector_model.vision_encoder.backbone.* -> vision_encoder.backbone.*
tracker_neck.*                           -> vision_encoder.neck.*
tracker_model.*                          -> *
~~~

映射后 tracker 所需 685 个张量全部存在，shape 全部一致，strict load 的
missing/unexpected keys 均为 0。注意不能把 detector neck 当成 tracker neck。

正式派生采用 FP32 AMG：

~~~text
points_per_crop:       32 (32x32 grid)
points_per_batch:      64
pred_iou_thresh:       0.88
stability_score_thresh: 0.95
crops_n_layers:        0
总 masks:              414
每帧 masks:            6--19，中位数 14
31 帧推理时间:          84.58 s
峰值显存:              约 5.60 GiB
mask score 范围:       0.8805--0.9901
~~~

BF16 在当前 Torch/Torchvision 组合的 NMS 中会出现 boxes/scores dtype 不一致，
因此使用 FP32。Transformers 5.4 的通用 mask-generation processor 对非方形图像
存在 point grid 与内部方形 resize 不一致的问题：直接送入 518x294 时，Y 方向的
point grid 只覆盖约 57.4%，而 `crops_n_layers=1` 还会在 stack masks 时 shape 不一致。
本次无侵入处理为：

1. 只在内存中把 518x294 RGB 双线性 warp 到 518x518；
2. 在 518x518 上运行 `crops_n_layers=0` 的 AMG；
3. 把 bool mask 最近邻缩放回 518x294；
4. 在原对齐尺寸上重算 bbox，过滤面积小于 100 像素的 mask；
5. 不写入方形 RGB，不修改原帧。

因此最终 detection mask 与 ConceptGraphs 使用的 518x294 RGB/depth 严格同尺寸。
输出位于：

~~~text
gsa_detections_sam3_clip/frame000000.pkl.gz ... frame000030.pkl.gz
gsa_vis_sam3_clip/frame000000.jpg ... frame000030.jpg
gsa_classes_sam3_clip.json
detections_manifest.json
.detections_complete
~~~

31 个 pickle 均通过 schema、dtype、shape、bbox-mask 一致性和 512 维特征归一化检查；
压缩后合计 13.692 MiB。

### 15.4 本地 CLIP 实测

`openai/clip-vit-base-patch16` 目录只有旧式 `pytorch_model.bin`。
本地 SHA256：

~~~text
ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0
~~~

它与 Hugging Face 官方仓库公布的 SHA256 一致。Transformers 5.4 会拒绝用
Torch 2.4.1 经 `from_pretrained()` 加载 pickle 权重；在哈希确认后，使用
`torch.load(..., weights_only=True)` 读取 state dict，再 strict-compatible 地加载
`CLIPModel`。实测 missing keys 为 0，仅多出两个可重建的 `position_ids` buffer。

单图图像和文本特征均成功输出归一化 512 维向量，显存约 0.57 GiB。
后续每个 SAM3 mask crop 和类别文本都必须使用同一个 ViT-B/16 模型，保证
`image_feats` 与 `text_feats` 维数和特征空间一致。它不是原仓库默认的
OpenCLIP ViT-H/14，因此需要在仓库外生成兼容的 detection pickle，不能直接调用
原始 `generate_gsa_results.py` 并期待它自动使用该 HF CLIP。

### 15.5 ConceptGraphs mapping 实测

正式 mapping 使用 `svpp`，但不修改该 Conda 环境。纯 Python 依赖从
`mapping_py311` 通过 `PYTHONPATH` 读取。`torch` 先导入时会优先加载系统旧
`libstdc++.so.6.0.30`，导致 FAISS 1.10.0 缺 `CXXABI_1.3.15`；`svpp` 自带的
`libstdc++.so.6.0.34` 已满足要求，因此只对目标进程使用：

~~~bash
LD_LIBRARY_PATH="/data2/zhengjie/miniconda3/envs/svpp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="/data2/zhengjie/data/concept_graphs/python_packages/mapping_py311:/data2/zhengjie/code/concept-graphs" \
/data2/zhengjie/miniconda3/envs/svpp/bin/python ...
~~~

这不会修改 activate 脚本或任何动态库。`torch + faiss`、`IndexFlatL2.add/search`
均已验证成功。

当前 checkout 不能原样直接启动 mapping，存在以下兼容问题：

1. `MapObjectList(device=cfg.device)`，但该类没有接受 `device` 的构造函数；
2. 以模块方式调用时 Hydra 找不到没有 `__init__.py` 的配置包；
3. `svpp` 缺 GradSLAM 和 supervision，但本次执行路径只需要 GradSLAM 的
   `scale_intrinsics()` 和 `relative_transformation()`，且 `vis_render=false`；
4. `spatial_sim_type=overlap` 的 bbox 候选预筛依赖缺失的 PyTorch3D。

按“不修改代码”要求，本次只在目标 Python 进程内：

- 为 `MapObjectList` 提供兼容的 list constructor；
- 直接加载原 `base.yaml` 并调用 Hydra wrapper 下的原函数主体；
- 提供本次执行路径需要的最小 GradSLAM/supervision import compatibility；
- 用 AABB IoU 代替 PyTorch3D OBB IoU做候选预筛。

最后一项只影响“是否需要进一步计算”的 bbox 粗筛；实际 `overlap` 分数仍由原代码的
FAISS point nearest-neighbor 实现计算，association、fusion、DBSCAN、filter 和 merge
均使用仓库原函数。

实际参数：

~~~text
start=0 end=-1 stride=1
gsa_variant=sam3_clip
spatial_sim_type=overlap
match_method=sim_sum
sim_threshold=1.2
mask_conf_threshold=0.95
max_bbox_area_ratio=0.5
dbscan_eps=0.1
class_agnostic=true
skip_bg=true
merge_interval=20
merge_visual_sim_thresh=0.8
merge_text_sim_thresh=0.8
vis_render=false
save_objects_all_frames=false
~~~

31 帧 mapping 用时约 12 秒。最终 post-processing 从 31 个候选对象过滤、合并为
13 个对象，共 60,675 个降采样点；每个对象包含 4--31 次跨帧 observations，所有点、
bbox 和 512 维 CLIP feature 均 finite，feature norm 为 1。

~~~text
pcd_saves/
├── full_pcd_sam3_clip_overlap_maskconf0.95_simsum1.2_dbscan.1_sam3_clip.pkl.gz
└── full_pcd_sam3_clip_overlap_maskconf0.95_simsum1.2_dbscan.1_sam3_clip_post.pkl.gz
mapping_manifest.json
.mapping_complete
~~~

进入 scene graph 的是第二个 `_post.pkl.gz`。

### 15.6 离线 Scene Graph 实测

当前本地没有 LLaVA checkpoint，也没有使用 OpenAI API；原
`build_scenegraph_cfslam.py` 还会在顶层导入当前环境缺失的 `rich`、`tyro`、
`transformers` 和 `openai`。因此没有伪装成论文原版语言阶段，而是生成了明确标注
provenance 的离线图：

1. 从 `_post.pkl.gz` 读取 13 个融合对象及其归一化 `clip_ft`；
2. 使用同一个本地 CLIP ViT-B/16，对 ScanNet200 和 bedroom 补充词表做 zero-shot；
3. 用多视角 mask montage、平面方向和 gravity 对低 margin 标签消歧；
4. 从 31 个 c2w 的 camera +Y 轴中位方向估计 gravity，得到
   `up=[0.041214,-0.999130,0.006421]`；
5. 生成 `on` 兼容边，并另外生成 `above`、`next_to`、`same_surface_as` 增强边。

节点结果：

| ID | 标签 | observations | 说明 |
|---:|---|---:|---|
| 0 | ceiling | 12 | 天花板片段 |
| 1 | floor | 15 | 床侧大块木地板 |
| 2 | wall | 4 | 近处墙/门框表面 |
| 3 | window | 13 | 右墙第一扇窗 |
| 4 | window | 6 | 右墙第二扇窗 |
| 5 | pillow | 29 | 中央白枕头 |
| 6 | wall art | 26 | 右侧墙画 |
| 7 | wall art | 31 | 左侧墙画 |
| 8 | pillow | 13 | 左侧装饰枕 |
| 9 | pillow | 12 | 右侧装饰枕 |
| 10 | floor | 10 | 门边木地板片段 |
| 11 | bed | 12 | 可见床/床品表面 |
| 12 | wall | 9 | 门边墙面片段 |

CLIP cosine 仅是排序分数，不是概率；结构面 top-1/top-2 margin 很小，因此
`possible_tags` 和 `semantic_confidence` 均保留在增强 JSON 中。ID 没有二次 prune，
所以 object map、节点 JSON、relation JSON 和可视化使用同一组连续 ID 0--12。

原 scene graph 脚本只对 point overlap > 0.01 的对象构建 MST；当前 fragment map 中
只有 3 组 pillow pair 超过该阈值，严格照搬会得到几乎无边的图。为让结果可用，增强图
增加了保守的 gravity/场景关系，同时把非论文关系单独标识：

~~~text
ConceptGraphs 兼容 on 边: 4
  pillow 5 -> bed 11
  pillow 8 -> bed 11
  pillow 9 -> bed 11
  bed 11    -> floor 1

增强边总数: 10
  on + above + next_to + same_surface_as
~~~

三个 pillow 到 bed 可见表面的 gravity-aligned gap 分别为 1.0、2.6、1.9 cm。
bed 节点主要是上表面，所以 bed-to-floor 的直接点接触没有被 mask 捕获，该边标为
medium confidence。

最终输出：

~~~text
scene_graph/
├── scene_graph.json                 # 原脚本节点字段兼容，13 nodes
├── cfslam_scenegraph_edges.pkl      # 原 tuple 格式，4 on edges
├── cfslam_scenegraph_edges.json     # 上述边的可读 JSON
├── cfslam_object_relations.json     # on 关系及 reason
├── scene_graph_enriched.json        # 13 nodes + 10 edges + provenance/限制
├── scene_graph_manifest.json        # 数量、校验状态与 SHA256
├── object_montage.jpg               # 逐对象 mask/crop 复查图
├── scene_graph_topdown.png          # 顶视节点与边可视化
└── .scene_graph_complete
~~~

所有 JSON 已通过解析和 ID/边界检查；pickle 与 JSON edge tuple 完全一致。

### 15.7 结果边界与可选下一步

15.6 记录的是一套可用的离线 Scene Graph；15.8 起新增了可配置的 OpenAI-compatible
语言链路。它们都不是论文 LLaVA+GPT-4 阶段的逐位复现。若要严格复现论文语言链路，
仍需：

1. 下载与当前仓库接口兼容的 LLaVA checkpoint；
2. 为每个 object 生成多视角 caption；
3. 提供 OpenAI API 或自行部署等价的 JSON-constrained LLM；
4. 运行 caption refinement 和原 on/in relation prompt；
5. 恢复论文的视角排序、具体 LLaVA/GPT-4 checkpoint 与 prompt/eval 条件。当前 baseline
   已恢复官方 MST 权重行为，同时保留 pruned/original ID 对齐和 response 校验。

在当前 31 帧和 mask 质量下，优先改进方向是为 SAM3 增加开放词汇 prompt/detector，
提高 lamp、nightstand、door 等完整实例的召回率，而不是仅调低三维过滤阈值；否则会
增加墙面/地板碎片并使 relation candidate graph 更不稳定。

### 15.8 AutoDL OpenAI 兼容接口配置（`svpp`）

已修改 `conceptgraph/scenegraph/build_scenegraph_cfslam.py` 的 API 调用层：

- `OPENAI_BASE_URL` 默认是 `https://www.autodl.art/api/v1`；
- 密钥可由 `--openai-api-key-file` / `OPENAI_API_KEY_FILE` 指向的 0600 文件读取，未指定
  文件时才回退到 `OPENAI_API_KEY`；密钥内容不会写入源码、Markdown 或输出目录；
- `OPENAI_MODEL` 默认是 `gpt-5.5`，可在运行前覆盖；
- `OPENAI_TIMEOUT` 默认是 120 秒，避免较慢的推理请求沿用原代码的 25 秒超时；
- `OPENAI_MAX_RETRIES` 默认是 0，避免兼容代理自动重复产生视觉计费；需要自动重试时可显式增大；
- `OPENAI_BASE_URL` 必须是无用户名、密码、query 和 fragment 的 HTTPS URL；
- URL、文本/视觉模型、超时和重试均有对应 CLI 参数，参数优先于环境变量默认值；
- OpenAI 系列使用 [AutoDL 配置说明](https://autodl.art/docs/cherry_studio/) 推荐的
  Responses API，SDK 会请求
  `https://www.autodl.art/api/v1/responses`；
- 缺少密钥时立即报出明确错误，不会发送匿名请求。

依赖使用 `svpp` 的 Python 3.11 安装到仓库外的隔离目录，不修改 Conda 环境本身：

~~~text
/data2/zhengjie/data/concept_graphs/python_packages/openai_py311
  openai 2.37.0
  transformers 4.31.0
  rich 13.5.3
  tyro 1.0.15
  numpy 1.24.0
~~~

`tyro` 使用 1.0.15 是因为仓库原锁定的 0.5.9 与当前 Python 3.11.15 的
`argparse` 不兼容。NumPy 固定为 1.24.0，与 `svpp` 的 SciPy 1.10.1 一致，避免
隔离目录中的 NumPy 2.x 覆盖 Conda 版本。

每次打开新终端，先设置非敏感运行参数：

~~~bash
export CG_ROOT=/data2/zhengjie/code/concept-graphs
export CG_DEPS=/data2/zhengjie/data/concept_graphs/python_packages
unset OPENAI_LOG
export PYTHONPATH="$CG_DEPS/openai_py311:$CG_DEPS/mapping_py311:$CG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data2/zhengjie/miniconda3/envs/svpp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENAI_BASE_URL=https://www.autodl.art/api/v1
export OPENAI_MODEL=gpt-5.5
export OPENAI_VISION_MODEL=gpt-5.5
export OPENAI_TIMEOUT=120
export OPENAI_MAX_RETRIES=0
export OPENAI_API_KEY_FILE=/data2/zhengjie/data/concept_graphs/secrets/openai_api_key
~~~

密钥不要放进脚本、明文命令参数、命令历史或 `.env`。在 Bash 中用静默输入写入
0600 文件；以后换 key 时重复执行这一段即可：

~~~bash
set -Eeuo pipefail
set +x
unset OPENAI_LOG
umask 077
install -d -m 700 "$(dirname "$OPENAI_API_KEY_FILE")"
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

可以先只查询模型列表验证鉴权；下列代码不会打印密钥：

~~~bash
/data2/zhengjie/miniconda3/envs/svpp/bin/python - <<'PY'
import os
from conceptgraph.scenegraph.build_scenegraph_cfslam import make_openai_client

client = make_openai_client(
    api_key_file=os.environ["OPENAI_API_KEY_FILE"],
    base_url=os.environ["OPENAI_BASE_URL"],
    max_retries=int(os.environ["OPENAI_MAX_RETRIES"]),
)
for model in client.models.list(timeout=float(os.environ["OPENAI_TIMEOUT"])).data:
    print(model.id)
PY
~~~

如果账户返回的文本或视觉模型 ID 不同，按列表原样更新 `OPENAI_MODEL` 和
`OPENAI_VISION_MODEL`。建议使用新的 cache
目录，避免覆盖 15.6 中已经生成的离线结果：

~~~bash
export RUN_ROOT=/data2/zhengjie/code/concept-graphs/outputs/bedroom_4_CmEIg9gMI74
export SMOKE_CACHE="$RUN_ROOT/smoke"
export OPENAI_CACHE="$RUN_ROOT/scene_graph_openai"
export MAP_FILE=/data2/zhengjie/data/concept_graphs/outputs/bedroom_4_CmEIg9gMI74/pcd_saves/full_pcd_sam3_clip_overlap_maskconf0.95_simsum1.2_dbscan.1_sam3_clip_post.pkl.gz
install -d -m 700 "$SMOKE_CACHE" "$OPENAI_CACHE"
OPENAI_COMMON_ARGS=(
  --openai-api-key-file "$OPENAI_API_KEY_FILE"
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-model "$OPENAI_MODEL"
  --openai-vision-model "$OPENAI_VISION_MODEL"
  --openai-timeout "$OPENAI_TIMEOUT"
  --openai-max-retries "$OPENAI_MAX_RETRIES"
)

# 可选：只处理 object 5 的一个视角，先验证视觉模型与计费链路。
# 此命令只写 partial 文件，不会伪装成完整 caption 集。
/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$SMOKE_CACHE" \
  --mapfile "$MAP_FILE" \
  --annot-inds 5 \
  --max-detections-per-object 1 \
  --masking-option red_outline

# 正式生成全部 13 个对象的多视角 caption；默认每对象 4 个视角。
/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode extract-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --max-detections-per-object 4 \
  --masking-option red_outline \
  --openai-image-detail high

/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode refine-node-captions \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0

/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py" \
  "${OPENAI_COMMON_ARGS[@]}" \
  --mode build-scenegraph \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0

/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/build_scenegraph_cfslam.py" \
  --mode generate-scenegraph-json \
  --cachedir "$OPENAI_CACHE" \
  --mapfile "$MAP_FILE" \
  --device cuda:0

# 通用 property/state 抽取；语义由外部 prompt 控制，不在 Python 中写类别规则。
/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/scenegraph_output.py" extract-attributes \
  --openai-api-key-file "$OPENAI_API_KEY_FILE" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --openai-timeout "$OPENAI_TIMEOUT" \
  --openai-max-retries "$OPENAI_MAX_RETRIES" \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --captions-file "$OPENAI_CACHE/cfslam_openai_captions.json" \
  --prompt-file "$CG_ROOT/conceptgraph/scenegraph/prompts/scene_graph_attributes.txt" \
  --output-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --cache-dir "$OPENAI_CACHE/scene_graph_attribute_cache" \
  --manifest-file "$OPENAI_CACHE/scene_graph_attributes_manifest.json"

# 合并 nodes、attributes 和 ConceptGraphs on/in edges 为截图式 sparse dict。
/data2/zhengjie/miniconda3/envs/svpp/bin/python \
  "$CG_ROOT/conceptgraph/scenegraph/scenegraph_output.py" format \
  --nodes-file "$OPENAI_CACHE/scene_graph_nodes.json" \
  --attributes-file "$OPENAI_CACHE/scene_graph_attributes.json" \
  --edges-file "$OPENAI_CACHE/cfslam_scenegraph_edges.pkl" \
  --output-json "$OPENAI_CACHE/scene_graph.json" \
  --output-repr "$OPENAI_CACHE/scene_graph.txt" \
  --manifest-file "$OPENAI_CACHE/scene_graph_format_manifest.json"
~~~

### 15.9 OpenAI 视觉 caption 阶段

`extract-node-captions` 已从本地 LLaVA 改为 OpenAI Responses 视觉输入。实现遵循
[OpenAI Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
中的 Base64 data URL 格式：请求 content 包含 `input_text` 和 `input_image`，图片
detail 默认使用 `high`。`OPENAI_VISION_MODEL` 可以与后续文本模型单独配置，未设置时
回退到 `OPENAI_MODEL`。

当前 bedroom post-map 有 13 个对象、192 条 observation。caption 阶段会：

1. 丢弃缺图、无效 bbox、mask 面积小于 100 像素或 mask/bbox fill 小于 0.1 的视角；
2. 对同帧 observation 去重；
3. 沿相机时间序列分成 4 段，每段选择 `confidence × mask_area` 最大的视角；
4. 按 bbox 加动态 padding，并用红色轮廓标出目标；
5. 将 JPEG crop 只在内存中编码为 Base64，每个视角请求一句简短英文 caption；
6. 每个视角立即原子写入缓存，中断后重跑不会重复请求已完成且参数一致的视角。

Responses 请求显式设置 `store=false`，用于关闭 Responses 应用状态存储；参见
[OpenAI 数据控制说明](https://developers.openai.com/api/docs/guides/your-data)。这不等同于
代理或上游服务“零日志、零留存”，AutoDL 的实际数据保留策略需要向服务方确认。Base64
图片和 API key 不会写入本地 checkpoint；view checkpoint 只保存请求指纹、原图路径和
返回 caption。JSON/checkpoint/pickle/debug PNG 在创建时即使用 `0600`，OpenAI caption、
view、debug 和 refinement response 子目录为 `0700`。API 异常日志会省略请求体和响应体。

本地 debug crop 默认关闭；需要人工复查时才在 extract 命令后增加
`--save-caption-debug`。refinement 的 prompt/response 也默认不打印到 stdout，需要调试时
才增加 `--print-openai-responses`。

实测筛选后 13 个对象都能稳定得到 4 个视角，所以完整首次运行有 52 个逻辑视觉请求；
默认 `OPENAI_MAX_RETRIES=0` 时至多尝试 52 次 HTTP 请求。修改视角数或启用重试会改变
实际尝试数。图片 crop、caption prompt 和后续 refinement prompt 会发送到配置的 AutoDL
API 服务；原始帧、mask、模型和已有离线 Scene Graph 均不会被覆盖。

输出结构：

~~~text
scene_graph_openai/
├── cfslam_captions_openai/
│   ├── views/                         # 每视角请求与 caption checkpoint
│   └── 0.json ... 12.json            # 每对象聚合 checkpoint
├── cfslam_captions_openai_debug/      # 可选；红框 crop + caption 复查图
├── cfslam_openai_captions_partial.json
├── cfslam_openai_captions.json        # 完整 canonical 输入
└── cfslam_openai_caption_manifest.json
~~~

只有全部对象都有非空且兼容的 checkpoint 时才会生成 canonical caption 文件。OpenAI
阶段不会创建或覆盖历史 `cfslam_llava_captions.json`，已有真实 LLaVA 结果会原样保留；
更新后的 refinement 直接读取 canonical OpenAI 文件。使用
`--annot-inds` 做单对象 smoke test 时只生成 partial 文件和 manifest，避免后续
`refine-node-captions` 误把部分结果当作完整场景。refinement 会优先读取
`cfslam_openai_captions.json`，并验证 manifest 为 complete 且 map 路径、大小、mtime、
SHA256 与当前 `--mapfile` 一致；不存在 OpenAI manifest 时才回退到旧 LLaVA 文件名。

已完成的无网络验证包括：真实 map 的 52 个 view selection、JPEG/Base64 可解码、
SDK 向 `/api/v1/responses` 发送的多模态 JSON、13 对象完整输出 schema、单对象 partial
保护、refinement 消费，以及第二次运行 52/52 cache hit。没有使用真实密钥，也没有
发出线上请求。
