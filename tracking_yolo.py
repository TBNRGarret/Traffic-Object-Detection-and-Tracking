"""
Traffic Object Detection & Tracking with YOLO + ByteTrack
==========================================================
Usage:
  python tracking_yolo.py
  python tracking_yolo.py --source "../VN-Dataset/vietnam traffic vids/chuongduonghn.mp4"
  python tracking_yolo.py --model "weights/best_yolo_vn.pt" --conf 0.3
"""
import os
import argparse
import torch
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
    parser = argparse.ArgumentParser(description="Vehicle tracking on Vietnam traffic video")
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help="Path to YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join("dataset", "videos", "vietnam_traffic_vids", "road1tdhhn.mp4"),
        help="Path to input video or webcam stream",
    )
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Inference device (0, cpu)",
    )
    parser.add_argument("--save", action="store_true", default=True, help="Save tracking video")
    parser.add_argument("--show-conf", action="store_true", default=False, help="Display confidence score")
    args = parser.parse_args()

    print(f"🚗 Loading model: {args.model}")
    print(f"📹 Source video:  {args.source}")
    print(f"⚡ Device:        {args.device}")

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source video not found: {args.source}")

    model = YOLO(args.model)

    for _ in model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        stream=True,
        show_labels=True,
        show_conf=args.show_conf,
    ):
        pass

    print("\n✅ Tracking finished. Result saved to runs/detect/track*/")


if __name__ == "__main__":
    main()
