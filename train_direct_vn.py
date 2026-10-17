"""
Train YOLOv11 directly on Vietnam Traffic Dataset (Cleaned)
============================================================
Strategy: Fine-tune from COCO-pretrained checkpoint → Vietnam data
No intermediate UA-DETRAC step needed (COCO already covers all 4 classes)

Dataset: 11,117 train / 2,751 val / 287 test (after deduplication)
Classes: motorbike, car, bus, truck
"""
import os
from ultralytics import YOLO


def main():
    # -----------------------------------------------------------
    # CONFIG - Adjust these based on your hardware
    # -----------------------------------------------------------
    BASE_MODEL = "yolo11n.pt"      # Options: yolo11n.pt (fast) | yolo11s.pt (recommended balance)
    DATA_CONFIG = "vn_dataset.yaml"
    EPOCHS = 50
    IMGSZ = 640                    # Try 960 if VRAM >= 8GB for better small-object detection
    BATCH = 16                     # Lower to 8 if OOM on 640, or 4 if using 960
    DEVICE = 0                     # GPU index (set to 'cpu' if no GPU)

    # -----------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,

        # --- Learning rate schedule ---
        lr0=0.01,               # Initial learning rate (default for fresh fine-tune)
        lrf=0.01,               # Final LR = lr0 * lrf = 0.0001
        cos_lr=True,            # Cosine annealing (smooth decay, avoids local minima)
        warmup_epochs=3.0,      # Warm up 3 epochs to protect pretrained features

        # --- Traffic-specific augmentation ---
        flipud=0.0,             # NO vertical flip (vehicles don't flip upside down)
        fliplr=0.5,             # Horizontal flip OK (left/right traffic symmetry)
        degrees=5.0,            # Slight rotation (tilted camera angles)
        mosaic=1.0,             # Mosaic ON: learn dense multi-vehicle scenes
        close_mosaic=10,        # Turn off mosaic last 10 epochs for bbox precision
        mixup=0.15,             # Mix images to handle occlusion between motorbikes
        copy_paste=0.15,        # Copy-paste rare classes (bus, truck) to balance dataset

        # --- Training stability ---
        patience=15,            # Early stop if no improvement for 15 epochs
        workers=4,
        amp=True,               # Mixed precision for faster training

        # --- Output ---
        project="runs/detect",
        name="vn_direct",
        save=True,
        plots=True,
    )

    print("\n✅ Training complete!")
    print(f"   Best weights: runs/detect/vn_direct/weights/best.pt")

    # -----------------------------------------------------------
    # VALIDATION on test set
    # -----------------------------------------------------------
    print("\n📊 Evaluating on test set...")
    best_model = YOLO("runs/detect/vn_direct/weights/best.pt")
    metrics = best_model.val(
        data=DATA_CONFIG,
        split="test",
        imgsz=IMGSZ,
        device=DEVICE,
    )
    print(f"   Test mAP@50:    {metrics.box.map50:.4f}")
    print(f"   Test mAP@50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
