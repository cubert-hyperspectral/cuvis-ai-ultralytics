from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


class YOLOPreprocess(Node):
    """Convert RGB images to stride-aligned channel-first tensors for YOLO.

    Handles the full letterbox pipeline: HWC RGB [0,1] to CHW RGB [0,1],
    proportional resize, grey-padding at 114/255 on the remaining borders.
    The channel order is kept: ultralytics models take RGB, and the library's
    own predictor turns cv2 BGR frames into RGB right before the model. Emits
    ``orig_hw`` and ``model_input_hw`` so that ``YOLOPostprocess`` can scale
    boxes back to source coordinates without any knowledge of this node's
    internals.

    Parameters
    ----------
    stride:
        Stride multiple used by the paired YOLO backbone (default 32).
        Pass ``YOLO26Detection(...).stride`` to keep the two nodes in sync.
    imgsz:
        Long side of the model input in pixels, the ``imgsz`` of an ultralytics
        predict call (640 by default there). The image is scaled so its long
        side equals ``imgsz`` and the short side is padded up to the next
        stride multiple, the rectangle ultralytics' LetterBox produces with
        ``auto=True``. A value that is not a multiple of ``stride`` is rounded
        up to the next multiple, as ultralytics' ``check_imgsz`` does, so the
        long side is never padded and ``YOLOPostprocess`` rebuilds the same
        gain the resize used. ``None`` (the default) keeps the image at its own
        resolution and only pads it to the stride multiple.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.RGB,
            NodeTag.IMAGE,
            NodeTag.PREPROCESSING,
            NodeTag.TORCH,
        }
    )

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
            description="Channel-first RGB [B, 3, H', W'] stride-aligned, in [0, 1]",
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

    def __init__(self, stride: int = 32, imgsz: int | None = None, **kwargs: Any) -> None:
        self.stride = max(int(stride), 1)
        if imgsz is not None:
            if int(imgsz) < 1:
                raise ValueError(f"YOLOPreprocess imgsz must be a positive number of pixels or None, got {imgsz}")
            rounded = self._ceil_to_stride(int(imgsz))
            if rounded != int(imgsz):
                logger.warning(
                    "YOLOPreprocess imgsz={} is not a multiple of stride {}; using {} as ultralytics does",
                    imgsz,
                    self.stride,
                    rounded,
                )
            imgsz = rounded
        self.imgsz = imgsz
        super().__init__(stride=stride, imgsz=imgsz, **kwargs)

    def forward(self, rgb_image: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Preprocess a batch of RGB images for YOLO inference."""
        if rgb_image.ndim != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(f"YOLOPreprocess expects rgb_image [B, H, W, 3], got {tuple(rgb_image.shape)}")

        batch = int(rgb_image.shape[0])
        device = rgb_image.device
        in_h, in_w = int(rgb_image.shape[1]), int(rgb_image.shape[2])

        orig_hw = torch.tensor([[in_h, in_w]] * batch, dtype=torch.int64, device=device)

        # HWC RGB to CHW RGB; values stay in [0, 1]. No channel swap: the model takes RGB.
        x = rgb_image.permute(0, 3, 1, 2).contiguous()

        # Letterbox: proportional resize then grey-pad to a stride multiple.
        if self.imgsz is None:
            target_h = self._ceil_to_stride(in_h)
            target_w = self._ceil_to_stride(in_w)
            gain = min(target_h / in_h, target_w / in_w)
            resized_h = max(int(round(in_h * gain)), 1)
            resized_w = max(int(round(in_w * gain)), 1)
        else:
            gain = self.imgsz / max(in_h, in_w)
            resized_h = max(int(round(in_h * gain)), 1)
            resized_w = max(int(round(in_w * gain)), 1)
            target_h = self._ceil_to_stride(resized_h)
            target_w = self._ceil_to_stride(resized_w)

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

    def _ceil_to_stride(self, size: int) -> int:
        return int(math.ceil(size / self.stride) * self.stride)
