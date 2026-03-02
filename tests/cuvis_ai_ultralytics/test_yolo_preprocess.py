from __future__ import annotations

import torch

from cuvis_ai_ultralytics.node import YOLOPreprocess


def test_import() -> None:
    assert YOLOPreprocess.__name__ == "YOLOPreprocess"


def test_output_keys() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert set(out.keys()) == {"preprocessed", "model_input_hw", "orig_hw"}


def test_output_channel_first_bgr() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((2, 64, 128, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    # Channel-first: [B, 3, H', W']
    assert out["preprocessed"].ndim == 4
    assert out["preprocessed"].shape[1] == 3


def test_output_stride_aligned() -> None:
    node = YOLOPreprocess(stride=32)
    # 50x70 → must be padded to nearest multiple of 32 (64x96).
    rgb = torch.rand((1, 50, 70, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    _, _, h, w = out["preprocessed"].shape
    assert h % 32 == 0
    assert w % 32 == 0


def test_orig_hw_matches_input() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((3, 48, 96, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["orig_hw"].shape == (3, 2)
    assert out["orig_hw"][0].tolist() == [48, 96]
    assert out["orig_hw"][1].tolist() == [48, 96]


def test_model_input_hw_matches_preprocessed() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((2, 50, 70, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    _, _, h, w = out["preprocessed"].shape
    assert out["model_input_hw"].shape == (2, 2)
    assert out["model_input_hw"][0].tolist() == [h, w]


def test_already_stride_aligned_no_padding() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((1, 64, 128, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (1, 3, 64, 128)
    assert out["orig_hw"][0].tolist() == [64, 128]
    assert out["model_input_hw"][0].tolist() == [64, 128]


def test_output_dtype_float32() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((1, 32, 32, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].dtype == torch.float32
    assert out["model_input_hw"].dtype == torch.int64
    assert out["orig_hw"].dtype == torch.int64
