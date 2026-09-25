# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Security: a pip-audit step joins the CI security job (there was none) and the locked environment is refreshed for the open advisories: aiohttp 3.14.3, anyio 4.14.2, click 8.5.0, cryptography 50.0.1, gitpython 3.1.62, hydra-core 1.3.7, keras 3.15.1, mako 1.4.3, mlflow 3.16.1, msgpack 1.2.2, onnx 1.23.0 (the macOS entry stays at 1.17.0 under the export extra's cap), pillow 12.3.0, pip 26.2.1, pyasn1 0.6.4, pymdown-extensions 12.1, pytest 9.1.1, pytorch-lightning 2.6.6, soupsieve 2.10, sqlparse 0.6.0, starlette 1.7.0, tornado 6.5.10. Ignored with a comment in CI: torch CVE-2025-3000 (fixed in 2.13.0, the pinned line stays) and setuptools PYSEC-2026-3447 (capped at 81 by the export and solutions extras; macOS sdist builds only).

## 0.1.4 - 2026-07-17

- Raised the `cuvis-ai-schemas` floor to 0.8.0 and `cuvis-ai-core` to 0.11.0, adopting the released cuvis-ai-next framework versions.
- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry (a machine-specific path must not ship in a release).

- CI: add a detect-secrets secret-scan job (git-tracked files only).

## 0.1.3 - 2026-06-23

- Require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions.

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
