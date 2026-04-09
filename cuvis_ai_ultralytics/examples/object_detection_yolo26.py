"""Standalone YOLO26 object detection on MP4 video (no cuvis.ai node graph).

Pipeline:
1. Read input video frames
2. Run YOLO26 prediction per frame
3. Render bbox overlays
4. Save overlay video + JSON detections
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from pathlib import Path
from typing import Any

import click
import cv2
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_classes(classes: str | None) -> list[int] | None:
    if classes is None or not classes.strip():
        return None
    try:
        parsed = [int(x.strip()) for x in classes.split(",") if x.strip()]
    except ValueError as exc:
        raise click.BadParameter("--classes must be a comma-separated list of integers.") from exc
    return parsed if parsed else None


def _id_to_color(class_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color.
    r = (37 * class_id + 23) % 256
    g = (17 * class_id + 101) % 256
    b = (29 * class_id + 53) % 256
    return int(b), int(g), int(r)


def _class_name_for(
    class_id: int,
    names: dict[int, str] | list[str] | tuple[str, ...] | None,
) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _draw_detections(
    frame_bgr_u8: Any,
    boxes_xyxy: torch.Tensor,
    class_ids: torch.Tensor,
    confidences: torch.Tensor,
    names: dict[int, str] | list[str] | tuple[str, ...] | None,
    line_thickness: int,
    draw_labels: bool,
) -> Any:
    """Draw detections on BGR uint8 frame and return rendered BGR frame."""
    rendered = frame_bgr_u8.copy()
    n = int(boxes_xyxy.shape[0])
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i].round().tolist()]
        class_id = int(class_ids[i].item())
        conf = float(confidences[i].item())
        color = _id_to_color(class_id)

        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness=max(1, int(line_thickness)))

        if draw_labels:
            class_name = _class_name_for(class_id, names)
            label = f"{class_name}:{conf:.2f}"
            y = max(0, y1 - 6)
            cv2.putText(
                rendered,
                label,
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return rendered


def _read_video_meta(video_path: str) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for metadata: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, total_frames


def _resolve_output_dir(video_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (REPO_ROOT / "outputs" / video_path.stem).resolve()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to MP4 video file.",
)
@click.option("--start-frame", type=int, default=0, show_default=True, help="First frame to process.")
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Last frame to process (-1 = end of video).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory to write results. Default: D:/code-repos/cuvis-ai-ultralytics/ultralytics-init/outputs/{video_name}"
    ),
)
@click.option("--model-path", type=str, default="yolo26n.pt", show_default=True)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Detection confidence threshold.",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.7,
    show_default=True,
    help="NMS IoU threshold.",
)
@click.option("--max-detections", type=int, default=300, show_default=True)
@click.option(
    "--classes",
    type=str,
    default=None,
    help="Optional class filter as comma-separated IDs (e.g. '0,1,2').",
)
@click.option("--agnostic-nms", is_flag=True, default=False, help="Enable class-agnostic NMS.")
@click.option("--line-thickness", type=int, default=2, show_default=True)
@click.option("--draw-labels/--no-draw-labels", default=True, show_default=True)
@click.option("--bf16", is_flag=True, default=False, help="Enable bfloat16 autocast.")
@click.option("--compile", "compile_model", is_flag=True, default=False, help="Enable torch.compile.")
def main(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_dir: Path | None,
    model_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    classes: str | None,
    agnostic_nms: bool,
    line_thickness: int,
    draw_labels: bool,
    bf16: bool,
    compile_model: bool,
) -> None:
    """Run standalone YOLO26 detection on MP4 video frames."""
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be >= 0.")
    if end_frame >= 0 and end_frame < start_frame:
        raise click.BadParameter("--end-frame must be -1 or >= --start-frame.")
    if max_detections <= 0:
        raise click.BadParameter("--max-detections must be > 0.")
    if line_thickness <= 0:
        raise click.BadParameter("--line-thickness must be > 0.")
    for opt_name, value in (
        ("--confidence-threshold", confidence_threshold),
        ("--iou-threshold", iou_threshold),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise click.BadParameter(f"{opt_name} must be in [0, 1].")

    classes_filter = _parse_classes(classes)

    width, height, source_fps, total_frames = _read_video_meta(str(video_path))
    if width <= 0 or height <= 0:
        raise click.ClickException(f"Invalid video dimensions for {video_path}.")
    if total_frames > 0 and start_frame >= total_frames:
        raise click.ClickException(f"--start-frame {start_frame} is out of range for {total_frames} source frames.")
    if source_fps <= 0:
        source_fps = 30.0

    resolved_end = end_frame
    if total_frames > 0:
        max_idx = total_frames - 1
        if resolved_end < 0:
            resolved_end = max_idx
        elif resolved_end > max_idx:
            logger.warning(
                "--end-frame {} exceeds source max frame {}, clamping.",
                resolved_end,
                max_idx,
            )
            resolved_end = max_idx

    resolved_output_dir = _resolve_output_dir(video_path, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = resolved_output_dir / "yolo26_overlay.mp4"
    output_json_path = output_video_path.with_suffix(".json")

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if bf16 and device != "cuda":
        logger.warning("bf16 requested on non-CUDA device; disabling autocast.")
        bf16 = False

    logger.info("Video: {}", video_path)
    logger.info("Start frame: {} | End frame: {}", start_frame, resolved_end)
    logger.info("Video size: {}x{} | FPS: {:.3f}", width, height, source_fps)
    if total_frames > 0:
        logger.info("Source frames: {}", total_frames)
    logger.info("YOLO model: {}", model_path)
    logger.info("Device: {}", device)
    logger.info("Output dir: {}", resolved_output_dir)

    yolo = YOLO(model_path)
    model_names = yolo.names
    yolo.model.to(device)
    yolo.model.eval()
    if compile_model:
        try:
            yolo.model = torch.compile(yolo.model, mode="reduce-overhead")  # type: ignore[assignment]
            logger.info("torch.compile enabled")
        except Exception as exc:
            logger.warning("torch.compile failed, continuing without compile: {}", exc)

    writer: cv2.VideoWriter | None = None
    detections_payload: list[dict[str, Any]] = []
    processed = 0
    last_source_frame_idx: int | None = None
    results = yolo(
        source=str(video_path),
        stream=True,
        conf=float(confidence_threshold),
        iou=float(iou_threshold),
        classes=classes_filter,
        agnostic_nms=bool(agnostic_nms),
        max_det=int(max_detections),
        device=device,
        verbose=False,
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    try:
        with torch.no_grad(), amp_ctx:
            for source_frame_idx, pred in enumerate(results):
                if source_frame_idx < start_frame:
                    continue
                if resolved_end >= 0 and source_frame_idx > resolved_end:
                    break

                frame_bgr = pred.orig_img
                if frame_bgr is None:
                    continue
                pred_boxes = pred.boxes

                if pred_boxes is not None and pred_boxes.xyxy is not None and pred_boxes.xyxy.numel() > 0:
                    boxes_cpu = pred_boxes.xyxy.to(dtype=torch.float32, device="cpu")
                    confidences = pred_boxes.conf.to(dtype=torch.float32, device="cpu")
                    class_ids = pred_boxes.cls.to(dtype=torch.int64, device="cpu")
                else:
                    boxes_cpu = torch.empty((0, 4), dtype=torch.float32)
                    confidences = torch.empty((0,), dtype=torch.float32)
                    class_ids = torch.empty((0,), dtype=torch.int64)

                rendered = _draw_detections(
                    frame_bgr_u8=frame_bgr,
                    boxes_xyxy=boxes_cpu,
                    class_ids=class_ids,
                    confidences=confidences,
                    names=model_names,
                    line_thickness=line_thickness,
                    draw_labels=draw_labels,
                )

                if writer is None:
                    h, w = int(rendered.shape[0]), int(rendered.shape[1])
                    writer = cv2.VideoWriter(
                        str(output_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer at {output_video_path}")
                writer.write(rendered)

                frame_detections: list[dict[str, Any]] = []
                for i in range(int(boxes_cpu.shape[0])):
                    cls_id = int(class_ids[i].item())
                    frame_detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": _class_name_for(cls_id, model_names),
                            "confidence": float(confidences[i].item()),
                            "bbox_xyxy": [float(v) for v in boxes_cpu[i].tolist()],
                        }
                    )

                detections_payload.append(
                    {
                        "frame_index": int(processed),
                        "source_frame_idx": int(source_frame_idx),
                        "num_detections": len(frame_detections),
                        "detections": frame_detections,
                    }
                )

                processed += 1
                last_source_frame_idx = int(source_frame_idx)
                if processed % 50 == 0:
                    logger.info("Processed {} frames (latest detections: {})", processed, len(frame_detections))
    finally:
        if writer is not None:
            writer.release()

    if processed <= 0:
        raise click.ClickException("No frames were processed. Check frame range and video content.")

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "info": {
                    "description": "YOLO26 video object detection output",
                    "source_video_path": str(video_path),
                    "source_start_frame": int(start_frame),
                    "source_end_frame": int(
                        last_source_frame_idx if last_source_frame_idx is not None else start_frame - 1
                    ),
                    "num_frames_processed": int(processed),
                    "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                },
                "config": {
                    "model_path": model_path,
                    "confidence_threshold": float(confidence_threshold),
                    "iou_threshold": float(iou_threshold),
                    "max_detections": int(max_detections),
                    "classes_filter": classes_filter,
                    "agnostic_nms": bool(agnostic_nms),
                    "line_thickness": int(line_thickness),
                    "draw_labels": bool(draw_labels),
                },
                "frames": detections_payload,
            },
            f,
            indent=2,
        )

    if not output_video_path.exists():
        raise RuntimeError(f"Expected overlay video was not created: {output_video_path}")
    if not output_json_path.exists():
        raise RuntimeError(f"Expected detections JSON was not created: {output_json_path}")

    logger.success("YOLO26 standalone video detection complete")
    logger.info("Overlay video: {}", output_video_path)
    logger.info("Detections JSON: {}", output_json_path)


if __name__ == "__main__":
    main()
