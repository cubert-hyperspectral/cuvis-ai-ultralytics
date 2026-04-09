"""cuvis.ai plugin package for Ultralytics YOLO26 nodes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuvis-ai-ultralytics")
except PackageNotFoundError:
    __version__ = "dev"


def register_all_nodes() -> int:
    """Register all plugin nodes in the cuvis.ai NodeRegistry."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    return registry.auto_register_package("cuvis_ai_ultralytics.node")


__all__ = ["__version__", "register_all_nodes"]
