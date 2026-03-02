from __future__ import annotations

import sys
import types

import torch

from cuvis_ai_ultralytics.node import YOLO26Detection


def _install_mock_ultralytics(monkeypatch):
    loaded: dict[str, str] = {}

    class DummyTorchModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.rand((x.shape[0], 84, 20), dtype=torch.float32, device=x.device)

    class DummyYOLO:
        def __init__(self, model_path: str) -> None:
            loaded["model_path"] = model_path
            self.model = DummyTorchModel()

    mock_module = types.ModuleType("ultralytics")
    mock_module.YOLO = DummyYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mock_module)
    return loaded


def test_import() -> None:
    assert YOLO26Detection.__name__ == "YOLO26Detection"


def test_eager_model_load(monkeypatch) -> None:
    loaded = _install_mock_ultralytics(monkeypatch)
    node = YOLO26Detection(model_path="yolo26n.pt")

    assert loaded["model_path"] == "yolo26n.pt"
    assert node.model_path == "yolo26n.pt"


def test_forward_raw_tensor_output(monkeypatch) -> None:
    _install_mock_ultralytics(monkeypatch)
    node = YOLO26Detection(model_path="yolo26n.pt")

    # Batch of 2 frames to validate native batching support.
    rgb_image = torch.rand((2, 32, 64, 3), dtype=torch.float32)
    outputs = node.forward(rgb_image=rgb_image)

    assert set(outputs.keys()) == {"raw_preds", "model_input_hw", "orig_hw"}
    assert isinstance(outputs["raw_preds"], torch.Tensor)
    assert outputs["raw_preds"].dtype == torch.float32
    assert outputs["model_input_hw"].shape == (2, 2)
    assert outputs["orig_hw"].shape == (2, 2)
    assert outputs["model_input_hw"].tolist() == [[32, 64], [32, 64]]
    assert outputs["orig_hw"].tolist() == [[32, 64], [32, 64]]
