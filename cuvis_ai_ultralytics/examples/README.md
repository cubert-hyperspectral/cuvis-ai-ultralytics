# cuvis_ai_ultralytics Examples

This folder contains plugin-level example scripts.

## object_detection_yolo26.py

Standalone YOLO26 object detection on MP4 video without using cuvis.ai Node plugins.

Outputs:
- `yolo26_overlay.mp4`
- `detections.json`

### Run (PowerShell)

Known-working local command (run from `cuvis-ai-sam3` workspace):

```powershell
Set-Location D:\code-repos\cuvis-ai\cuvis-ai-sam3
$env:PYTHONPATH = "D:\code-repos\cuvis-ai-ultralytics\ultralytics-init"

uv run python D:\code-repos\cuvis-ai-ultralytics\ultralytics-init\cuvis_ai_ultralytics\examples\object_detection_yolo26.py `
  --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.mp4" `
  --start-frame 0 `
  --end-frame 49 `
  --model-path yolo26n.pt `
  --output-dir "D:\experiments\yolo26\yolo_output"
```

PowerShell note:
- The backtick (`` ` ``) must be the very last character on each continued line (no trailing spaces).
- Do not put the backtick on its own line.

### Useful options

- `--start-frame 0`
- `--end-frame -1`
- `--confidence-threshold 0.5`
- `--iou-threshold 0.7`
- `--max-detections 300`
- `--classes 0,1,2`
- `--agnostic-nms`
- `--bf16`
- `--compile`
