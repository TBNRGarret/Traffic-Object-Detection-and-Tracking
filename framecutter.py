import cv2
import os

video_dir = r"D:\FPT\vietnam traffic vids"
output_root = r"D:\FPT\frames"

os.makedirs(output_root, exist_ok=True)

for video_name in os.listdir(video_dir):
    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(video_dir, video_name)
    video_id = os.path.splitext(video_name)[0]

    # special case: Nguyen Trai rush hour
    if video_id.lower() == "nguyentraihn":
        target_frames = 250
    else:
        target_frames = 200

    output_dir = os.path.join(output_root, video_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    print(f"{video_id}: saved {saved} frames")
