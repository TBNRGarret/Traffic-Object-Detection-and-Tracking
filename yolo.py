"""
Train YOLO on UA-DETRAC dataset (Legacy baseline script, standardized)
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.1,
        mosaic=1.0,
        workers=4,
        project="runs/detect",
        name="uadetrac_train",
    )


if __name__ == "__main__":
    main()
