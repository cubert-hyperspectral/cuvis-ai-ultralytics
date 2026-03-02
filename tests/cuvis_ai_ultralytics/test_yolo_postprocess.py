from __future__ import annotations

import sys
import types

import torch

from cuvis_ai_ultralytics.node import YOLOPostprocess


def _install_mock_ultralytics_utils(monkeypatch, nms_impl, scale_impl) -> None:
    ultralytics_mod = types.ModuleType("ultralytics")
    utils_mod = types.ModuleType("ultralytics.utils")
    utils_mod.nms = types.SimpleNamespace(non_max_suppression=nms_impl)
    utils_mod.ops = types.SimpleNamespace(scale_boxes=scale_impl)
    ultralytics_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils_mod)


def test_import() -> None:
    assert YOLOPostprocess.__name__ == "YOLOPostprocess"


def test_postprocess_outputs(monkeypatch) -> None:
    det = torch.tensor([[10.0, 20.0, 30.0, 40.0, 0.95, 3.0]], dtype=torch.float32)

    def nms_impl(*args, **kwargs):
        # Return one detection per sample for a batch of size 2
        return [det, det]

    def scale_impl(_in_shape, boxes, _out_shape):
        return boxes + 1.0

    _install_mock_ultralytics_utils(monkeypatch, nms_impl, scale_impl)

    node = YOLOPostprocess()
    outputs = node.forward(
        raw_preds=torch.rand((2, 84, 100), dtype=torch.float32),
        model_input_hw=torch.tensor([[320, 640]], dtype=torch.int64),
        orig_hw=torch.tensor([[160, 320]], dtype=torch.int64),
    )

    assert outputs["bboxes"].shape == (2, 1, 4)
    assert outputs["category_ids"].shape == (2, 1)
    assert outputs["confidences"].shape == (2, 1)
    assert outputs["category_ids"].dtype == torch.int64
    assert outputs["confidences"].dtype == torch.float32
    torch.testing.assert_close(outputs["bboxes"][0, 0], torch.tensor([11.0, 21.0, 31.0, 41.0]))
    torch.testing.assert_close(outputs["bboxes"][1, 0], torch.tensor([11.0, 21.0, 31.0, 41.0]))


def test_empty_detections(monkeypatch) -> None:
    def nms_impl(*args, **kwargs):
        return [torch.empty((0, 6), dtype=torch.float32), torch.empty((0, 6), dtype=torch.float32)]

    def scale_impl(_in_shape, boxes, _out_shape):
        return boxes

    _install_mock_ultralytics_utils(monkeypatch, nms_impl, scale_impl)

    node = YOLOPostprocess()
    outputs = node.forward(
        raw_preds=torch.rand((2, 84, 100), dtype=torch.float32),
        model_input_hw=torch.tensor([[320, 640]], dtype=torch.int64),
        orig_hw=torch.tensor([[160, 320]], dtype=torch.int64),
    )

    assert outputs["bboxes"].shape == (2, 0, 4)
    assert outputs["category_ids"].shape == (2, 0)
    assert outputs["confidences"].shape == (2, 0)


def test_scale_boxes_path(monkeypatch) -> None:
    calls: list[tuple[tuple[int, int], tuple[int, int, int]]] = []

    def nms_impl(*args, **kwargs):
        det = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 2.0]], dtype=torch.float32)
        return [det, det]

    def scale_impl(in_shape, boxes, out_shape):
        calls.append((in_shape, out_shape))
        return boxes

    _install_mock_ultralytics_utils(monkeypatch, nms_impl, scale_impl)

    node = YOLOPostprocess()
    _ = node.forward(
        raw_preds=torch.rand((2, 84, 100), dtype=torch.float32),
        model_input_hw=torch.tensor([[640, 640], [320, 320]], dtype=torch.int64),
        orig_hw=torch.tensor([[480, 848], [200, 300]], dtype=torch.int64),
    )

    assert calls == [((640, 640), (480, 848, 3)), ((320, 320), (200, 300, 3))]
