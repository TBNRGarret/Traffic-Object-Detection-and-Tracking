## Cursor Cloud specific instructions

This is a **Python ML/CV pipeline** for traffic object detection and tracking using YOLOv11 (Ultralytics). There are no web services, databases, or Docker containers — only standalone Python scripts.

### Dependencies

Install with: `pip install ultralytics onnxruntime opencv-python-headless numpy onnx onnxslim`

This pulls in PyTorch, torchvision, and all other transitive dependencies automatically.

### Key model weights (committed to repo)

- `yolo11n.pt` — base YOLOv11n pretrained weights (COCO)
- `runs/detect/train4/weights/best.pt` — fine-tuned model (Vietnam traffic: motorbike, car, bus, truck)
- `runs/detect/train4/weights/best.onnx` — ONNX export of the fine-tuned model

### Running scripts

All scripts have **hardcoded Windows paths** (`D:\FPT\...`). To run any script on Linux, paths must be adjusted. The scripts that can run without external datasets (using existing weights):

| Script | What it does | Needs external data? |
|---|---|---|
| `benchmark_onnx.py` | ONNX latency benchmark | No (uses dummy input) |
| `export_onnx.py` | Export best.pt to ONNX | No |
| `onnx_testing.py` | Predict with ONNX model | Yes (needs a test image) |
| `tracking_yolo.py` | ByteTrack on video | Yes (needs video file) |
| `yolo.py` | Pretrain on UA-DETRAC | Yes (needs dataset) |
| `vnyolo.py` | Fine-tune on VN data | Yes (needs dataset) |
| `framecutter.py` | Extract video frames | Yes (needs video files) |
| `scriptprelabel.py` | Pre-label images | Yes (needs frames) |

### Gotchas

- Scripts default to `device=0` (GPU). On CPU-only environments, override with `device='cpu'`.
- The `ultralytics` package auto-installs missing optional deps (onnx, onnxslim) on first use; this may trigger warnings about restarting the runtime — safe to ignore.
- No `requirements.txt` exists in the repo; dependencies are implicit from imports.
- No linter or test framework is configured. Validation is done by running the scripts directly.
- `.gitignore` excludes `*.txt` and `*.jpg`/`*.png` files, so label and image files won't appear in git status.
