"""
Auto pre-label extracted frames using trained YOLO model
========================================================
Usage:
  python scriptprelabel.py
  python scriptprelabel.py --model "weights/best_yolo_vn.pt" --images "frames" --labels "prelabels"
"""
import os
import argparse
from ultralytics import YOLO


def get_default_model():
    candidates = [
        os.path.join("runs", "detect", "vn_direct", "weights", "best.pt"),
        os.path.join("weights", "best_yolo_vn.pt"),
        "yolo11n.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description="Pre-label video frames using YOLO")
    parser.add_argument("--model", type=str, default=get_default_model(), help="Model weights")
    parser.add_argument(
        "--images",
        type=str,
        default=os.path.join("dataset", "videos", "frames"),
        help="Input frames directory",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="prelabels",
        help="Output labels directory",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.images):
        print(f"❌ Image directory not found: {args.images}")
        return

    print(f"🏷️ Loading model:  {args.model}")
    print(f"📂 Images source: {args.images}")
    print(f"💾 Labels output: {args.labels}")
    model = YOLO(args.model)
    os.makedirs(args.labels, exist_ok=True)

    for item in os.listdir(args.images):
        img_dir = os.path.join(args.images, item)
        if not os.path.isdir(img_dir):
            continue

        out_label_dir = os.path.join(args.labels, item)
        os.makedirs(out_label_dir, exist_ok=True)

        model.predict(
            source=img_dir,
            conf=args.conf,
            save=False,
            save_txt=True,
            project=args.labels,
            name=item,
            exist_ok=True,
        )
        print(f"   ✓ Pre-labeled {item}")

    print("\n✅ Pre-labeling complete!")


if __name__ == "__main__":
    main()
