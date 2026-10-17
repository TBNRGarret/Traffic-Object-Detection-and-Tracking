"""
Export trained YOLO PyTorch weights to ONNX format
===================================================
Usage:
  python export_onnx.py
  python export_onnx.py --model "runs/detect/vn_direct/weights/best.pt"
"""
import os
import argparse
from ultralytics import YOLO


def get_default_model():
    candidates = [
        os.path.join("runs", "detect", "vn_direct", "weights", "best.pt"),
        os.path.join("weights", "best_yolo_vn.pt"),
        os.path.join("weights", "best_colab_old.pt"),
        "yolo11n.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, default=get_default_model(), help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic axes")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model")
    args = parser.parse_args()

    print(f"📦 Exporting model: {args.model} -> ONNX")
    model = YOLO(args.model)
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
    print(f"✅ ONNX model saved to: {exported_path}")


if __name__ == "__main__":
    main()