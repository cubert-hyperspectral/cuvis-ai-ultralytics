from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


class YOLO26Detection(Node):
    """Run YOLO26 raw tensor inference on a stride-aligned CHW BGR batch.

    Expects input from :class:`YOLOPreprocess` — a channel-first BGR tensor
    whose spatial dimensions are already stride-aligned and padded.  The node
    holds only the inner ``nn.Module`` extracted from the Ultralytics wrapper so
    that ``pipeline.to(device)`` propagates weights correctly without triggering
    Ultralytics' training-mode side-effects.

    Parameters
    ----------
    model_path:
        YOLO weight file or model ID (e.g. ``"yolo26n.pt"``).
    half_precision:
        Cast model weights and input to FP16 before inference.
    """

    INPUT_SPECS = {
        "preprocessed": PortSpec(
            dtype=torch.float32,
            shape=(-1, 3, -1, -1),
            description="Channel-first BGR [B, 3, H', W'] from YOLOPreprocess",
        ),
    }

    OUTPUT_SPECS = {
        "raw_preds": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Raw YOLO prediction tensor",
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

        # Expose stride so callers can construct a matching YOLOPreprocess:
        #   pre = YOLOPreprocess(stride=det.stride)
        stride = getattr(self.yolo_model, "stride", 32)
        if isinstance(stride, torch.Tensor):
            self.stride = int(stride.max().item())
        elif isinstance(stride, (list, tuple)):
            self.stride = int(max(stride))
        else:
            self.stride = int(stride)
        self.stride = max(self.stride, 1)

        logger.info("YOLO model loaded (stride={})", self.stride)

    def forward(self, preprocessed: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Execute raw model forward on a stride-aligned CHW BGR batch."""
        if preprocessed.ndim != 4 or preprocessed.shape[1] != 3:
            raise ValueError(
                f"YOLO26Detection expects preprocessed [B, 3, H, W], got {tuple(preprocessed.shape)}"
            )

        x = preprocessed.half() if self.half_precision else preprocessed.float()
        preds = self.yolo_model(x)
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"Expected tensor raw predictions, got {type(raw)}")

        return {"raw_preds": raw.float()}
