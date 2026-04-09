from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


class YOLOPreprocess(Node):
    """Convert RGB images to stride-aligned channel-first BGR tensors for YOLO.

    Handles the full letterbox pipeline: HWC RGB [0,1] → CHW BGR [0,1],
    proportional resize to fill a stride-aligned canvas, grey-padding at
    114/255 on the remaining borders.  Emits ``orig_hw`` and
    ``model_input_hw`` so that ``YOLOPostprocess`` can scale boxes back to
    source coordinates without any knowledge of this node's internals.

    Parameters
    ----------
    stride:
        Stride multiple used by the paired YOLO backbone (default 32).
        Pass ``YOLO26Detection(...).stride`` to keep the two nodes in sync.
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in [0, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "preprocessed": PortSpec(
            dtype=torch.float32,
            shape=(-1, 3, -1, -1),
            description="Channel-first BGR [B, 3, H', W'] stride-aligned, in [0, 1]",
        ),
        "model_input_hw": PortSpec(
            dtype=torch.int64,
            shape=(-1, 2),
            description="Padded model input [H', W'] per sample",
        ),
        "orig_hw": PortSpec(
            dtype=torch.int64,
            shape=(-1, 2),
            description="Original image [H, W] per sample before preprocessing",
        ),
    }

    def __init__(self, stride: int = 32, **kwargs: Any) -> None:
        self.stride = max(int(stride), 1)
        super().__init__(stride=stride, **kwargs)

    def forward(self, rgb_image: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Preprocess a batch of RGB images for YOLO inference."""
        if rgb_image.ndim != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(f"YOLOPreprocess expects rgb_image [B, H, W, 3], got {tuple(rgb_image.shape)}")

        batch = int(rgb_image.shape[0])
        device = rgb_image.device
        in_h, in_w = int(rgb_image.shape[1]), int(rgb_image.shape[2])

        orig_hw = torch.tensor([[in_h, in_w]] * batch, dtype=torch.int64, device=device)

        # HWC RGB → CHW BGR; values stay in [0, 1].
        x = rgb_image[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()

        # Letterbox: proportional resize then grey-pad to stride multiple.
        target_h = int(math.ceil(in_h / self.stride) * self.stride)
        target_w = int(math.ceil(in_w / self.stride) * self.stride)

        if target_h != in_h or target_w != in_w:
            gain = min(target_h / in_h, target_w / in_w)
            resized_h = max(int(round(in_h * gain)), 1)
            resized_w = max(int(round(in_w * gain)), 1)
            if resized_h != in_h or resized_w != in_w:
                x = F.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

            pad_h = target_h - resized_h
            pad_w = target_w - resized_w
            top = int(round(pad_h / 2 - 0.1))
            bottom = pad_h - top
            left = int(round(pad_w / 2 - 0.1))
            right = pad_w - left
            if top or bottom or left or right:
                x = F.pad(x, (left, right, top, bottom), value=114.0 / 255.0)

        model_input_hw = torch.tensor([[x.shape[2], x.shape[3]]] * batch, dtype=torch.int64, device=device)

        return {
            "preprocessed": x.float(),
            "model_input_hw": model_input_hw,
            "orig_hw": orig_hw,
        }
