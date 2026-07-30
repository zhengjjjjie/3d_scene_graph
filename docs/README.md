# 3D Scene Graph 运行说明

## 使用当前已经构建完成的 Map

当前 Video2Mesh Map 已构建并验证完成，不需要重新运行 Video2Mesh。执行下面的命令继续构造 scene graph：

```bash
cd /data/zhengjie/code/3d_scene_graph
conda activate svpp
./run_scene_graph.sh
```

按照终端提示输入 API key。脚本会复用已有的 caption 和节点精炼缓存，并使用 `multiview-2d-3d` 关系模式。

最终输出文件：

```text
/data/zhengjie/data/concept_graphs/video2mesh_runs/bedroom_4_CmEIg9gMI74/bedroom_v2m_20260729_clean1/scene_graph_openai/scene_graph.json
/data/zhengjie/data/concept_graphs/video2mesh_runs/bedroom_4_CmEIg9gMI74/bedroom_v2m_20260729_clean1/scene_graph_openai/scene_graph.txt
```

当前成功的 ConceptGraphs Map：

```text
/data/zhengjie/data/concept_graphs/video2mesh_runs/bedroom_4_CmEIg9gMI74/bedroom_v2m_20260729_clean1/conceptgraphs/full_pcd_video2mesh_colmap_sam2.pkl.gz
```

## 从视频开始一次全新实验

每次全新实验必须使用一个没有使用过的 `run-id`：

```bash
cd /data/zhengjie/code/3d_scene_graph
conda activate svpp

python -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --video /data/zhengjie/data/svpp/bedroom_4_CmEIg9gMI74/video.mp4 \
  --scene-id bedroom_4_CmEIg9gMI74 \
  --output-base /data/zhengjie/data/concept_graphs/video2mesh_runs \
  --run-id bedroom_v2m_NEW_RUN_ID \
  --profile bedroom_validation
```

Video2Mesh 和 ConceptGraphs Map 构建完成后，使用相同的 run 名称构造 scene graph：

```bash
RUN_NAME=bedroom_v2m_NEW_RUN_ID ./run_scene_graph.sh
```

实验输出目录结构为：

```text
/data/zhengjie/data/concept_graphs/video2mesh_runs/
└── bedroom_4_CmEIg9gMI74/
    └── bedroom_v2m_NEW_RUN_ID/
        ├── v2m_project/
        ├── conceptgraphs/
        ├── scene_graph_openai/
        └── logs/
```

不要删除或覆盖已有 run。需要重新开始实验时，应更换 `--run-id`。
