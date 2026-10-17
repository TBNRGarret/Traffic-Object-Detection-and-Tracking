"""
Active Learning Loop: Enrich dataset from raw traffic videos
=============================================================
Workflow:
  1. Run trained model on raw videos to detect + track vehicles
  2. Extract frames where model is UNCERTAIN (low confidence = hard samples)
  3. Auto-generate pseudo-labels for manual review
  4. Add reviewed samples to training set for next round

Usage:
  python enrich_data.py --video "dataset/videos/vietnam_traffic_vids/nguyentraihn.mp4"
"""
import os
import cv2
import argparse
from ultralytics import YOLO


def extract_hard_samples(
    video_path: str,
    model_path: str,
    output_dir: str = "enrichment",
    conf_low: float = 0.15,
    conf_high: float = 0.45,
    frame_interval: int = 30,
    max_frames: int = 500,
    imgsz: int = 640,
):
    """
    Extract frames where the model is uncertain (conf between conf_low and conf_high).
    These are the most valuable samples for improving the model.
    """
    model = YOLO(model_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Output directories
    images_dir = os.path.join(output_dir, video_name, "images")
    labels_dir = os.path.join(output_dir, video_name, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {video_name}")
    print(f"   Frames: {total_frames} | FPS: {fps:.1f} | Duration: {total_frames/fps:.0f}s")
    print(f"   Sampling every {frame_interval} frames ({frame_interval/fps:.1f}s)")
    print(f"   Uncertainty range: conf [{conf_low}, {conf_high}]")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        # Run inference
        results = model.predict(frame, imgsz=imgsz, conf=conf_low, verbose=False)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            frame_idx += 1
            continue

        confs = result.boxes.conf.cpu().numpy()

        # Check if any detection falls in the uncertainty zone
        uncertain_mask = confs <= conf_high
        if not uncertain_mask.any():
            frame_idx += 1
            continue

        # Save frame
        fname = f"{video_name}_frame{frame_idx:06d}"
        img_path = os.path.join(images_dir, f"{fname}.jpg")
        cv2.imwrite(img_path, frame)

        # Save pseudo-labels (YOLO format) for ALL detections in this frame
        txt_path = os.path.join(labels_dir, f"{fname}.txt")
        h, w = frame.shape[:2]
        with open(txt_path, 'w') as f:
            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                # Add confidence as comment for manual review
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved += 1
        frame_idx += 1

    cap.release()
    print(f"\n✅ Extracted {saved} hard samples → {output_dir}/{video_name}/")
    print(f"   ⚠️  Review pseudo-labels before adding to training set!")
    return saved


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
    parser = argparse.ArgumentParser(description="Extract hard samples from traffic videos")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default=get_default_model(),
                        help="Path to trained model weights")
    parser.add_argument("--output", type=str, default="enrichment", help="Output directory")
    parser.add_argument("--conf-low", type=float, default=0.15, help="Lower confidence threshold")
    parser.add_argument("--conf-high", type=float, default=0.45, help="Upper confidence threshold")
    parser.add_argument("--interval", type=int, default=30, help="Sample every N frames")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to extract")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    extract_hard_samples(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        conf_low=args.conf_low,
        conf_high=args.conf_high,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
