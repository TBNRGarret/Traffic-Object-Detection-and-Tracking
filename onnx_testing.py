"""
Test YOLO ONNX inference on a sample frame or image
===================================================
Usage:
  python onnx_testing.py
  python onnx_testing.py --model "runs/detect/vn_direct/weights/best.onnx" --source "path/to/image.jpg"
"""
import os
import argparse
from ultralytics import YOLO


def find_default_onnx():
    candidates = [
        os.path.join("runs", "detect", "vn_direct", "weights", "best.onnx"),
        os.path.join("weights", "best_yolo_vn.onnx"),
        "yolo11n.onnx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--model", type=str, default=find_default_onnx(), help="Path to ONNX model")
    parser.add_argument("--source", type=str, default="", help="Path to test image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    if not args.model or not os.path.exists(args.model):
        print("⚠️ No ONNX model found. Please run export_onnx.py first!")
        return

    # If no source provided, look for any sample image in dataset
    if not args.source:
        sample_dir = os.path.join("dataset", "images", "val")
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
            if sample_files:
                args.source = os.path.join(sample_dir, sample_files[0])

    if not args.source or not os.path.exists(args.source):
        print("⚠️ No test image found. Please pass --source <image_path>")
        return

    print(f"🔍 Testing ONNX model: {args.model}")
    print(f"🖼️ Test image:        {args.source}")
    model = YOLO(args.model)
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=True)
    print(f"✅ Prediction complete. Output saved to runs/detect/predict*/")


if __name__ == "__main__":
    main()