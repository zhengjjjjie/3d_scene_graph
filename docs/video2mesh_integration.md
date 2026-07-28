# Video2Mesh 2D/3D mask integration

This integration replaces the per-frame mask association and RGB-D
backprojection front end with the following file-backed pipeline:

1. Video2Mesh extracts uniformly selected video frames.
2. COLMAP reconstructs a sparse point cloud and registered cameras.
3. GroundingDINO retains raw detections (up to 40 candidates) without merging
   bed parts.
4. ConceptGraphs normalizes whole-object prompts while keeping pillow, lamp,
   and nightstand instance seeds separate.
5. SAM2 writes immutable raw tracks to `masks/2d_raw`.
6. ConceptGraphs complete-link merges duplicate tracks into `masks/2d` and
   writes pillow-carved, fusion-only bed masks to `masks/2d_fusion`.
7. Video2Mesh projects the fusion masks into the sparse point cloud.
8. The adapter converts those sets into a normal ConceptGraphs
   `MapObjectList` pickle, including local CLIP image and text features.

The Video2Mesh and GroundingDINO repositories are called as external tools;
their source is not copied into ConceptGraphs.  Commit pins and the discovered
local paths are recorded in
`conceptgraph/configs/video2mesh_pipeline.yaml`.

## Data safety and output layout

The command never writes into the source-video directory or a pre-existing
ConceptGraphs scene/output directory. Each run receives a dedicated new
directory:

```text
<output-base>/<scene-id>/<run-id>/
├── v2m_project/       # Video2Mesh project and original 2D/3D mask artifacts
│   └── logs/conceptgraphs_video2mesh/  # run manifest, stage logs and markers
└── conceptgraphs/     # compatible MapObjectList pickle and conversion report
```

An existing run directory is rejected.  `--resume` is accepted only with an
explicit `--run-id`; completed stages are reused only when the recorded input,
configuration, version fingerprint, and stage artifact SHA-256 values still
match. A converted map is reused only when its conversion report, pickle hash,
runner fingerprint, and current Video2Mesh input hashes also match. It never
means "overwrite" or "clear output".

COLMAP's sparse reconstruction has an arbitrary scale.  The resulting map is
compatible with the existing caption and scene-graph readers, but its
coordinates must not be interpreted as metres.

## Setup and preflight

This checkout is configured to use the manually provisioned SAM2 installation:

```text
source:     /data2/zhengjie/File/sam2
Python:     /data2/zhengjie/miniconda3/envs/sam2/bin/python
checkpoint: /data2/zhengjie/File/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

Do not run `bootstrap` for this manually managed layout. The command deliberately
refuses to take ownership of a pre-existing manual environment. The bootstrap
implementation remains available for a separately configured, fully isolated
dependency root.

Set the ConceptGraphs command interpreter:

```bash
CG_PYTHON=/data2/zhengjie/miniconda3/envs/groundingdino/bin/python
```

The command examples below assume this variable remains set in the same shell.
Preflight verifies the pinned SAM2 revision, Python/PyTorch/OpenCV runtime,
CUDA availability, and the configured SAM2.1 tiny checkpoint size and SHA-256.
Conversion likewise verifies the configured local CLIP `pytorch_model.bin`
SHA-256 before loading it.

Preflight also verifies the pinned GroundingDINO configuration and checkpoint
hashes. These model hashes are part of the resume fingerprint, so outputs from
different model bytes cannot be mixed in one resumed run.

The configured COLMAP binary is capability-probed as well. The pinned
Video2Mesh revision emits the legacy `SiftExtraction/SiftMatching.use_gpu`
switches; for COLMAP 4.x the ConceptGraphs-owned compatibility wrapper rewrites
only those two option names to `FeatureExtraction/FeatureMatching.use_gpu`.
The real binary path, version, SHA-256, wrapper SHA-256, and selected option
mapping are recorded in the run fingerprint. Video2Mesh itself is not modified.

Check repositories, commits, interpreters, checkpoints, COLMAP and an input
video before starting a long run:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline preflight \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/input.mp4
```

All commands print a JSON report and return non-zero when a required contract
is not satisfied. `run --dry-run` is the exception: it returns success when
command planning succeeds and reports environment readiness separately as
`ready_to_run`; inspect `runner.preflight.errors` when that value is false.

## Run a scene

For a general video, the default frame policy is Video2Mesh's Quick policy:
sample at most 200 frames uniformly from the full video.

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/input.mp4 \
  --scene-id living_room \
  --output-base /data2/zhengjie/data/concept_graphs/video2mesh_runs
```

Before allocating GPU time, inspect the exact stage commands:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/input.mp4 \
  --scene-id living_room \
  --output-base /data2/zhengjie/data/concept_graphs/video2mesh_runs \
  --run-id dry-run-review \
  --dry-run
```

The pinned Video2Mesh vocabulary is used when no query override is provided.
For a general scene, a comma-separated text file or a Video2Mesh-compatible
JSON query file can be supplied:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/input.mp4 \
  --scene-id laboratory \
  --output-base /data2/zhengjie/data/concept_graphs/video2mesh_runs \
  --queries-file /absolute/path/to/laboratory_queries.txt
```

The bedroom comparison profile corresponds to source frames
`2709, 2733, ..., 3429`:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline run \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --video /absolute/path/to/bedroom.mp4 \
  --scene-id bedroom_4_CmEIg9gMI74 \
  --output-base /data2/zhengjie/data/concept_graphs/video2mesh_runs \
  --profile bedroom_validation
```

The equivalent explicit flags are
`--start-frame 2709 --end-frame 3429 --stride 24 --max-frames 31`; explicit
frame flags override values from the selected profile.

## Convert or validate existing artifacts

Conversion can be repeated from a complete, read-only Video2Mesh project
without rerunning reconstruction or tracking. The output name must end in
`.pkl.gz`:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline convert \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --project-root /absolute/path/to/v2m_project \
  --output /absolute/path/to/full_pcd_video2mesh_colmap_sam2.pkl.gz
```

Validate either the Video2Mesh project, the converted map, or both:

```bash
$CG_PYTHON -m conceptgraph.scripts.run_video2mesh_pipeline validate \
  --config conceptgraph/configs/video2mesh_pipeline.yaml \
  --project-root /absolute/path/to/v2m_project \
  --map /absolute/path/to/full_pcd_video2mesh_colmap_sam2.pkl.gz
```

As with every Python pickle, only pass a map produced by a trusted run to
`--map`.

Objects with no fused COLMAP points remain usable when they have at least
three non-empty views. They are serialized as `geometry_type=multiview_2d`
with empty `(0, 3)` point/color/bbox arrays, empty `point_indices`, and
`bbox=None` at runtime; no origin placeholder point is created.

## Hybrid relation graph

Use the new mode for a Video2Mesh map and keep the same value for captioning,
refinement, relation construction, and node JSON generation:

```bash
RELATION_MODE=multiview-2d-3d \
MAP_FILE=/absolute/path/to/full_pcd_video2mesh_colmap_sam2.pkl.gz \
RUN_NAME=bedroom_sam2_run_colmap41_relations_v2 \
./run_scene_graph.sh
```

This mode loads both foreground and background objects, unions multi-view 2D
and scale-independent 3D candidates, and bypasses the MST. It writes
`cfslam_multiview_relation_evidence.json` and independently cached pair
results. More than 100 candidates is an explicit diagnostic failure rather
than a silent truncation. The default `legacy-3d-mst` mode is unchanged.
