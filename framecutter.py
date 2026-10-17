"""
Extract evenly-spaced frames from Vietnam traffic videos
=========================================================
Usage:
  python framecutter.py
  python framecutter.py --video-dir "../VN-Dataset/vietnam traffic vids" --output "frames"
"""
import os
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract frames from traffic videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.path.join("dataset", "videos", "vietnam_traffic_vids"),
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("dataset", "videos", "frames"),
        help="Root output directory for extracted frames",
    )
    parser.add_argument("--default-frames", type=int, default=200, help="Frames per video")
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        print(f"❌ Video directory not found: {args.video_dir}")
        return

    os.makedirs(args.output, exist_ok=True)
    video_files = [f for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.asf', '.mov'))]

    print(f"📂 Scanning: {args.video_dir} ({len(video_files)} videos found)")
    print(f"💾 Saving frames to: {args.output}")

    for video_name in video_files:
        video_path = os.path.join(args.video_dir, video_name)
        video_id = os.path.splitext(video_name)[0]

        target_frames = 250 if video_id.lower() == "nguyentraihn" else args.default_frames

        output_dir = os.path.join(args.output, video_id)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        interval = max(total_frames // target_frames, 1)

        frame_id = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret or saved >= target_frames:
                break

            if frame_id % interval == 0:
                out_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1

            frame_id += 1

        cap.release()
        print(f"   ✓ {video_id}: saved {saved} frames")

    print("\n✅ Frame extraction complete!")


if __name__ == "__main__":
    main()
