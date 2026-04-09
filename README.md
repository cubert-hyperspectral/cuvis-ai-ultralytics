![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# CUVIS.AI Ultralytics

This repository provides a port of Ultralytics YOLO26 as a cuvis.ai plugin, enabling object detection and preprocessing/postprocessing pipelines. It is maintained by Cubert GmbH as part of the cuvis.ai ecosystem.

## Platform

cuvis.ai is split across multiple repositories:

| Repository | Role |
|---|---|
| [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) | Framework — base `Node` class, pipeline orchestration, services, and plugin system |
| [cuvis-ai-schemas](https://github.com/cubert-hyperspectral/cuvis-ai-schemas) | Shared schema definitions and generated types |
| [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) | Node catalog and end-user pipeline examples |
| **cuvis-ai-ultralytics** (this repo) | Ultralytics plugin — cuvis.ai nodes for YOLO26 object detection |

## Nodes

| Node | Description |
|---|---|
| `YOLOPreprocess` | Letterbox resize/pad and RGB-to-BGR channel flip for YOLO input |
| `YOLO26Detection` | YOLO26 object detection inference |
| `YOLOPostprocess` | NMS filtering and coordinate rescaling to original frame dimensions |

## Quick Start

For local development in this repository:

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git
cd cuvis-ai-ultralytics
uv sync --all-extras
```

For cuvis.ai usage examples, see the YOLO + DeepEIoU tracking pipelines in [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai/tree/main/examples/object_tracking/deepeiou).

For the original upstream Ultralytics README, see [README_ORIGINAL.md](README_ORIGINAL.md). For plugin-specific technical details, see [README_TECHNICAL.md](README_TECHNICAL.md).

## Links

- **Documentation:** https://docs.cuvis.ai/latest/
- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Original Ultralytics README:** [README_ORIGINAL.md](README_ORIGINAL.md)
- **Technical README:** [README_TECHNICAL.md](README_TECHNICAL.md)

---

See [LICENSE](LICENSE) for repository licensing details.
