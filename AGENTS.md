# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a Python-based traffic object detection and tracking system built on YOLOv11 (Ultralytics). The codebase consists of standalone Python scripts for training, tracking, ONNX export, benchmarking, and data utilities. There is no web server, no test suite, and no linting configuration in the current repo.

### Key Dependencies

All installed via pip (no `requirements.txt` in the repo): `ultralytics`, `opencv-python-headless`, `onnxruntime`, `numpy`, `torch`, `torchvision`.

### Running Scripts

- **All scripts use hardcoded Windows paths** (`D:\FPT\...`). They will fail at runtime unless those paths are edited or the planned config refactoring (described in `.kiro/specs/`) is implemented.
- The pre-trained base model `yolo11n.pt` is checked into the repo root and can be loaded directly: `YOLO('yolo11n.pt')`.
- Fine-tuned weights are expected at `runs/detect/train4/weights/best.pt` but are gitignored and not present in the repo.

### Quick Verification

```bash
# Verify dependencies
python3 -c "from ultralytics import YOLO; m = YOLO('yolo11n.pt'); print('OK')"

# Run inference on a test image (CPU)
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
results = model.predict(source='https://ultralytics.com/images/bus.jpg', imgsz=640, device='cpu')
print(f'Detections: {len(results[0].boxes)}')
"
```

### Gotchas

- No `requirements.txt` exists in the repo. Dependencies must be installed manually (`pip install ultralytics opencv-python-headless onnxruntime numpy`).
- The `Dockerfile` has a broken CMD (`CMD honga 'my.example:app`) and references a missing `requirements.txt`. Docker builds will fail.
- `compose.yaml` just wraps the broken Dockerfile; `docker compose up` will not work.
- No automated tests or linting configuration exist in the repo.
- ONNX export creates `yolo11n.onnx` in the working directory; clean it up if it's not wanted in commits.
- Scripts reference `device=0` (GPU); use `device='cpu'` when no GPU is available.
