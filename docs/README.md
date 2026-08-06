# 文档索引

端到端安装、模型下载、环境配置和运行命令统一维护在仓库根目录的
[README](../README.md)。

补充文档：

- [当前整体流程与技术设计](current_pipeline_technical_design.md)：从视频、
  实例消歧、稀疏点级 3D mask 融合到多视角 scene graph 的当前完整实现。
- [Video2Mesh 集成设计](video2mesh_integration.md)：阶段边界、产物契约、
  resume 安全性和混合 2D/3D 关系图。
- [精简检测流程](archive/streamlined_detection.md)：原始 ConceptGraphs
  检测流程的补充说明。
- [历史 Pipeline 记录](archive/pipeline_experiment_notes.md)：早期实验记录，
  仅供追溯，不作为当前安装或运行入口。

以根目录 README 和
`conceptgraph/configs/video2mesh_pipeline.yaml` 为当前版本的唯一运行依据。
