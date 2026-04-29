# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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
