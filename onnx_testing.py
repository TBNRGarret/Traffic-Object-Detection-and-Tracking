import ultralytics
from ultralytics import YOLO

model = YOLO(r"D:\FPT\runs\detect\train4\weights\best.onnx")

model.predict(
    source=r"D:\FPT\frames\chuongduonghn\frame_00003.jpg",
    imgsz=640,
    conf=0.25
)