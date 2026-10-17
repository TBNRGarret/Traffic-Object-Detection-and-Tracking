"""
Fine-tune YOLO on Vietnam Traffic Dataset (Legacy script, standardized)
Note: For modern optimized training, use `train_direct_vn.py` instead.
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="vn_dataset.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        lr0=0.001,
        device=0,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.1,
        mosaic=1.0,
        workers=4,
        project="runs/detect",
        name="vnyolo_legacy",
    )


if __name__ == "__main__":
    main()
