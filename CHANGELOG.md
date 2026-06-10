# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.2 - 2026-06-10

- Require `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2` (inherits the upstream security floors transitively).
- Relaxed the `export` extra's `numpy<2.0.0` cap to `numpy>=2.4.1` so it resolves with cuvis-ai-core (which requires numpy 2); TensorFlow 2.21+ supports numpy 2.x.
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the PyPI/TestPyPI release workflow; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests.
- Stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.1 - 2026-04-29

- Annotated `YOLO26Detection` with `_category = NodeCategory.MODEL` and `_tags = {RGB, IMAGE, DETECTION, BBOX, INFERENCE, LEARNABLE, BATCHED, TORCH}`; `YOLOPreprocess` with `_category = TRANSFORM` and `_tags = {RGB, IMAGE, PREPROCESSING, TORCH}`; `YOLOPostprocess` with `_category = TRANSFORM` and `_tags = {BBOX, DETECTION, POSTPROCESSING, TORCH}`.
- Added `cuvis-ai-schemas>=0.4.0` to dependencies (`NodeCategory` / `NodeTag` enums live there).
- Stripped `hash` fields from `torch` / `torchvision` wheel entries in `uv.lock`.

## 0.1.0 - 2026-04-07

- Added `cuvis_ai_ultralytics` plugin package with `YOLO26Detection`, `YOLOPreprocess`, and `YOLOPostprocess` node classes.
- Added plugin scaffolding with `pyproject.toml`, `setuptools-scm` versioning, and `.gitignore`.
- Added CI (`ci.yml`) and tag-driven GitHub release (`release.yml`) workflows.
- Added security scanning job (pip-audit, detect-secrets, bandit) to release workflow.
- Restructured README into user-facing, technical, and original upstream docs.
- Extracted `YOLOPreprocess` node from inline preprocessing for composable pipelines.
