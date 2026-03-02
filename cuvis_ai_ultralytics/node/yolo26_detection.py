from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


class YOLO26Detection(Node):
    """Run YOLO26 forward pass and expose raw prediction tensors."""

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in [0, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "raw_preds": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Raw YOLO prediction tensor",
        ),
        "model_input_hw": PortSpec(
            dtype=torch.int64,
            shape=(-1, 2),
            description="Model input H,W for scale-back",
        ),
        "orig_hw": PortSpec(
            dtype=torch.int64,
            shape=(-1, 2),
            description="Original image H,W before model forward",
        ),
    }

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        half_precision: bool = False,
        **kwargs: Any,
    ) -> None:
        from ultralytics import YOLO

        super().__init__(model_path=model_path, half_precision=half_precision, **kwargs)

        self.model_path = model_path
        self.half_precision = bool(half_precision)

        logger.info("Loading YOLO model: {}", self.model_path)
        # Store only the inner nn.Module, not the YOLO wrapper.
        # Ultralytics overrides nn.Module.train() to launch a training run, so
        # registering the wrapper as a submodule would cause predictor.eval() to
        # try starting a COCO training job.  Extracting .model gives us the real
        # PyTorch module so that .to(device) / .eval() propagate correctly.
        _yolo_wrapper = YOLO(self.model_path)
        self.yolo_model = _yolo_wrapper.model
        if self.half_precision:
            self.yolo_model.half()

        # Cache stride once; Ultralytics models expose stride as tensor/list/int.
        stride = getattr(self.yolo_model, "stride", 32)
        if isinstance(stride, torch.Tensor):
            self._stride_int = int(stride.max().item())
        elif isinstance(stride, (list, tuple)):
            self._stride_int = int(max(stride))
        else:
            self._stride_int = int(stride)
        self._stride_int = max(self._stride_int, 1)

        logger.info("YOLO model loaded successfully")

    def forward(self, rgb_image: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Execute raw model forward pass on an arbitrary batch size."""
        if rgb_image.ndim != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"YOLO26Detection expects rgb_image shape [B, H, W, 3], got {tuple(rgb_image.shape)}"
            )

        batch = int(rgb_image.shape[0])
        device = rgb_image.device

        # [B, 2] original H,W per frame
        orig_hw = torch.stack(
            [
                torch.tensor(
                    [rgb_image[i].shape[0], rgb_image[i].shape[1]],
                    dtype=torch.int64,
                    device=device,
                )
                for i in range(batch)
            ],
            dim=0,
        )

        # Ultralytics expects channel-first BGR tensors.
        x = rgb_image[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()

        # Raw model forward requires stride-compatible H/W or concat layers can fail
        # for odd downsample/upsample paths (e.g., 1000x1080 -> 64 vs 63 feature map).
        in_h, in_w = int(x.shape[2]), int(x.shape[3])
        target_h = int(math.ceil(in_h / self._stride_int) * self._stride_int)
        target_w = int(math.ceil(in_w / self._stride_int) * self._stride_int)

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

        x = (x.half() if self.half_precision else x.float()).contiguous()

        preds = self.yolo_model(x)
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"Expected tensor raw predictions, got {type(raw)}")

        raw = raw.float()
        model_input_hw = torch.tensor(
            [x.shape[2], x.shape[3]], dtype=torch.int64, device=raw.device
        ).unsqueeze(0).repeat(batch, 1)

        return {
            "raw_preds": raw,
            "model_input_hw": model_input_hw,
            "orig_hw": orig_hw,
        }
