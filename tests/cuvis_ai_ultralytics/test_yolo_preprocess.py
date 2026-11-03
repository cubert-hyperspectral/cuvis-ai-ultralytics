from __future__ import annotations

import torch

from cuvis_ai_ultralytics.node import YOLOPreprocess

GREY = 114.0 / 255.0


def test_import() -> None:
    assert YOLOPreprocess.__name__ == "YOLOPreprocess"


def test_output_keys() -> None:
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert set(out.keys()) == {"preprocessed", "model_input_hw", "orig_hw"}


def test_output_channel_first_rgb() -> None:
    # Ultralytics models take RGB: the predictor turns cv2 BGR frames into RGB right before the
    # model, so the node keeps the channel order of its RGB input and only moves it channel-first.
    node = YOLOPreprocess(stride=32)
    rgb = torch.rand((2, 64, 128, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (2, 3, 64, 128)
    assert torch.equal(out["preprocessed"], rgb.permute(0, 3, 1, 2))


def test_channel_order_survives_resize_and_padding() -> None:
    node = YOLOPreprocess(stride=32)
    red = torch.zeros((1, 50, 70, 3), dtype=torch.float32)
    red[..., 0] = 1.0
    out = node.forward(rgb_image=red)["preprocessed"]
    assert out[:, 0].max().item() == 1.0
    assert out[:, 1].max().item() < 0.5  # at most the grey padding (114/255)
    assert out[:, 2].max().item() < 0.5


def test_output_stride_aligned() -> None:
    node = YOLOPreprocess(stride=32)
    # 50x70 must be padded to the nearest multiple of 32 (64x96).
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


def test_imgsz_defaults_to_native_resolution() -> None:
    node = YOLOPreprocess(stride=32)
    assert node.imgsz is None
    rgb = torch.rand((1, 64, 128, 3), dtype=torch.float32)
    assert node.forward(rgb_image=rgb)["preprocessed"].shape == (1, 3, 64, 128)


def test_imgsz_scales_the_long_side_then_pads_to_the_stride() -> None:
    # 50x70 at imgsz 64: the long side becomes 64, the short side 46, padded up to 64.
    node = YOLOPreprocess(stride=32, imgsz=64)
    rgb = torch.rand((1, 50, 70, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (1, 3, 64, 64)
    assert out["model_input_hw"][0].tolist() == [64, 64]
    assert out["orig_hw"][0].tolist() == [50, 70]


def test_imgsz_keeps_the_rectangle_like_the_ultralytics_letterbox() -> None:
    # 300x640 at imgsz 640 stays 640 wide and is padded only up to the next stride multiple
    # in height (320), the shape ultralytics' LetterBox produces with auto=True.
    node = YOLOPreprocess(stride=32, imgsz=640)
    rgb = torch.rand((1, 300, 640, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (1, 3, 320, 640)
    assert out["model_input_hw"][0].tolist() == [320, 640]


def test_imgsz_upscales_a_small_image() -> None:
    # 50x70 at imgsz 128: the long side becomes 128, the short side 91, padded up to 96.
    node = YOLOPreprocess(stride=32, imgsz=128)
    rgb = torch.rand((1, 50, 70, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (1, 3, 96, 128)


def test_imgsz_padding_is_centred_grey() -> None:
    node = YOLOPreprocess(stride=32, imgsz=640)
    rgb = torch.full((1, 300, 640, 3), 0.5, dtype=torch.float32)
    out = node.forward(rgb_image=rgb)["preprocessed"]
    assert torch.allclose(out[0, :, :10, :], torch.full_like(out[0, :, :10, :], GREY))
    assert torch.allclose(out[0, :, 310:, :], torch.full_like(out[0, :, 310:, :], GREY))
    assert torch.allclose(out[0, :, 10:310, :], torch.full_like(out[0, :, 10:310, :], 0.5))


def test_imgsz_is_a_serialised_hparam() -> None:
    node = YOLOPreprocess(stride=32, imgsz=640)
    assert node.imgsz == 640
    assert node.hparams["imgsz"] == 640
    assert YOLOPreprocess(stride=32).hparams["imgsz"] is None


def test_imgsz_is_rounded_up_to_the_stride() -> None:
    # ultralytics snaps imgsz to a stride multiple before it is used (check_imgsz); without it the
    # long side would be padded too and YOLOPostprocess would rebuild a different gain from the
    # padded shape than the one the resize used.
    node = YOLOPreprocess(stride=32, imgsz=100)
    assert node.imgsz == 128
    assert node.hparams["imgsz"] == 128
    rgb = torch.rand((1, 100, 200, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["model_input_hw"][0].tolist() == [64, 128]


def test_imgsz_long_side_is_never_padded() -> None:
    node = YOLOPreprocess(stride=32, imgsz=160)
    for h, w in ((93, 131), (131, 93), (160, 160), (17, 400)):
        out = node.forward(rgb_image=torch.rand((1, h, w, 3), dtype=torch.float32))
        model_h, model_w = out["model_input_hw"][0].tolist()
        assert max(model_h, model_w) == 160, (h, w, model_h, model_w)


def test_imgsz_path_handles_a_batch() -> None:
    node = YOLOPreprocess(stride=32, imgsz=64)
    rgb = torch.rand((2, 50, 70, 3), dtype=torch.float32)
    out = node.forward(rgb_image=rgb)
    assert out["preprocessed"].shape == (2, 3, 64, 64)
    assert out["model_input_hw"].tolist() == [[64, 64], [64, 64]]
    assert out["orig_hw"].tolist() == [[50, 70], [50, 70]]
