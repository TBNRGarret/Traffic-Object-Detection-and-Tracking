from ultralytics import YOLO
import os

model = YOLO(r"D:\FPT\runs\detect\train5\weights\best.pt")

image_root = r"D:\FPT\frames"
label_root = r"D:\FPT\prelabels"

os.makedirs(label_root, exist_ok=True)

for video_id in os.listdir(image_root):
    img_dir = os.path.join(image_root, video_id)
    if not os.path.isdir(img_dir):
        continue

    out_label_dir = os.path.join(label_root, video_id)
    os.makedirs(out_label_dir, exist_ok=True)

    results = model.predict(
        source=img_dir,
        conf=0.3,
        save=False,
        save_txt=True,
        project=label_root,
        name=video_id
    )

    print(f"Pre-labeled {video_id}")
