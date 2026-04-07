# cuvis-ai-ultralytics

`cuvis-ai-ultralytics` packages the cuvis.ai plugin nodes that wrap the
Ultralytics YOLO26 runtime for use inside `NodeRegistry` pipelines.

The repository vendors the upstream `ultralytics` package alongside three
plugin-facing node classes:

- `cuvis_ai_ultralytics.node.YOLOPreprocess`
- `cuvis_ai_ultralytics.node.YOLO26Detection`
- `cuvis_ai_ultralytics.node.YOLOPostprocess`

## Release Manifest

Use the released plugin from `cuvis-ai-tracking` with a selective manifest:

```yaml
plugins:
  ultralytics:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_ultralytics.node.YOLO26Detection
      - cuvis_ai_ultralytics.node.YOLOPostprocess
      - cuvis_ai_ultralytics.node.YOLOPreprocess
```

## Local Development Manifest

Use a local checkout while iterating on the plugin:

```yaml
plugins:
  ultralytics:
    path: "../../../../cuvis-ai-ultralytics/ultralytics-init"
    provides:
      - cuvis_ai_ultralytics.node.YOLO26Detection
      - cuvis_ai_ultralytics.node.YOLOPostprocess
      - cuvis_ai_ultralytics.node.YOLOPreprocess
```

## Run Through cuvis-ai-tracking

From `cuvis-ai-tracking`, this plugin is exercised by the YOLO + DeepEIoU
tracking example:

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --video-path "D:\experiments\20260331\video_creation\tristimulus\XMR_50mm_ObjectTracking\20260331\12_39_03\Auto_002.mp4" `
  --no-reid `
  --output-dir "D:\experiments\20260407\deepeiou" `
  --out-basename "v0_1_0_smoke" `
  --end-frame 60
```

## Local Validation

```powershell
uv run --no-sources --extra dev pytest tests/cuvis_ai_ultralytics -v
uv run --no-sources --extra dev ruff format --check cuvis_ai_ultralytics tests/cuvis_ai_ultralytics
uv run --no-sources --extra dev ruff check cuvis_ai_ultralytics tests/cuvis_ai_ultralytics
uv build --no-sources
uv run --no-sources --with twine twine check dist/*
```

## Notes

- CI and release workflows use `uv ... --no-sources` so they ignore the local
  editable `cuvis-ai-core` override and validate publishable metadata.
- The package keeps the upstream `ultralytics` runtime in-tree so the plugin can
  be installed directly from the GitHub release tag referenced by cuvis.ai
  manifests.
- This repository distributes Ultralytics-derived code under `AGPL-3.0`. See
  [LICENSE](LICENSE).
