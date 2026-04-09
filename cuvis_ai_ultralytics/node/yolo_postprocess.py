from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


class YOLOPostprocess(Node):
    """Ultralytics NMS + box scaling postprocess for YOLO raw tensors."""

    INPUT_SPECS = {
        "raw_preds": PortSpec(dtype=torch.float32, shape=(-1, -1, -1), description="Raw YOLO tensor"),
        "model_input_hw": PortSpec(dtype=torch.int64, shape=(-1, 2), description="Model H,W"),
        "orig_hw": PortSpec(dtype=torch.int64, shape=(-1, 2), description="Original H,W"),
    }

    OUTPUT_SPECS = {
        "bboxes": PortSpec(dtype=torch.float32, shape=(-1, -1, 4), description="Final boxes [B, N, 4]"),
        "category_ids": PortSpec(dtype=torch.int64, shape=(-1, -1), description="Class ids [B, N]"),
        "confidences": PortSpec(dtype=torch.float32, shape=(-1, -1), description="Scores [B, N]"),
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        agnostic_nms: bool = False,
        classes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.agnostic_nms = bool(agnostic_nms)
        self.classes = classes
        super().__init__(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            agnostic_nms=agnostic_nms,
            classes=classes,
            **kwargs,
        )

    def forward(
        self,
        raw_preds: torch.Tensor,
        model_input_hw: torch.Tensor,
        orig_hw: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Apply Ultralytics non-max suppression and scale boxes back to source size."""
        from ultralytics.utils import nms, ops

        raw_preds_f32 = raw_preds.float()
        detections = nms.non_max_suppression(
            raw_preds_f32,
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_detections,
        )

        batch = int(raw_preds_f32.shape[0])
        device = raw_preds_f32.device

        # Allow model_input_hw/orig_hw provided as single row (broadcast) or per-sample.
        if model_input_hw.shape[0] not in {1, batch}:
            raise ValueError(f"model_input_hw batch mismatch: got {model_input_hw.shape[0]} rows for batch {batch}")
        if orig_hw.shape[0] not in {1, batch}:
            raise ValueError(f"orig_hw batch mismatch: got {orig_hw.shape[0]} rows for batch {batch}")

        bboxes_list: list[torch.Tensor] = []
        cat_list: list[torch.Tensor] = []
        conf_list: list[torch.Tensor] = []
        max_len = 0

        for i in range(batch):
            det = detections[i] if i < len(detections) else torch.empty((0, 6), device=device)
            if det is None or det.numel() == 0:
                boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
                categories = torch.empty((0,), device=device, dtype=torch.int64)
                confs = torch.empty((0,), device=device, dtype=torch.float32)
            else:
                boxes = det[:, :4]
                in_h, in_w = [int(v) for v in model_input_hw[min(i, model_input_hw.shape[0] - 1)].tolist()]
                out_h, out_w = [int(v) for v in orig_hw[min(i, orig_hw.shape[0] - 1)].tolist()]
                boxes = ops.scale_boxes((in_h, in_w), boxes, (out_h, out_w, 3))
                categories = det[:, 5].long()
                confs = det[:, 4].float()

            max_len = max(max_len, int(boxes.shape[0]))
            bboxes_list.append(boxes)
            cat_list.append(categories)
            conf_list.append(confs)

        if max_len == 0:
            return {
                "bboxes": torch.empty((batch, 0, 4), dtype=torch.float32, device=device),
                "category_ids": torch.empty((batch, 0), dtype=torch.int64, device=device),
                "confidences": torch.empty((batch, 0), dtype=torch.float32, device=device),
            }

        bboxes_out = torch.zeros((batch, max_len, 4), dtype=torch.float32, device=device)
        cats_out = torch.full((batch, max_len), -1, dtype=torch.int64, device=device)
        conf_out = torch.full((batch, max_len), -1.0, dtype=torch.float32, device=device)

        for i in range(batch):
            n = int(bboxes_list[i].shape[0])
            if n > 0:
                bboxes_out[i, :n] = bboxes_list[i].float()
                cats_out[i, :n] = cat_list[i]
                conf_out[i, :n] = conf_list[i]

        return {
            "bboxes": bboxes_out,
            "category_ids": cats_out,
            "confidences": conf_out,
        }
