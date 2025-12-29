from ultralytics import YOLO

# Load model đã train
model = YOLO(r"D:\FPT\runs\detect\train4\weights\best.pt")

# Chạy tracking

for _ in model.track(
    source="D:\\FPT\\vietnam traffic vids\\road1tdhhn.mp4",
    tracker="bytetrack.yaml",
    conf=0.25,
    iou=0.5,
    imgsz=640,
    device=0,
    save=True,
    stream=True,
    show_labels=True,
    show_conf=False
):
    pass


