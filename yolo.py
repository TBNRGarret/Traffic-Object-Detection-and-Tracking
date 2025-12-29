import ultralytics
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"D:\FPT\data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        device=0,

        # Traffic-aware augmentation
        flipud=0.0,
        fliplr=0.5,
        degrees=0.1,
        mosaic=1.0,

        workers=4 # Number of data loading workers
    )

if __name__ == "__main__":
    main()
