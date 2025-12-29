from ultralytics import YOLO

model = YOLO(r"D:\FPT\runs\detect\train4\weights\best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=True
)