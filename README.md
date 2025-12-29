# 🚦 Traffic Object Detection & Tracking (YOLOv11)

## 📌 Overview

This project focuses on **traffic video analysis** using **Deep Learning / Computer Vision**, including **object detection, tracking, model optimization, and edge-oriented deployment**.

The pipeline is built around **YOLOv11** and is trained in a **two-stage strategy**:

1. Pretraining on **UA-DETRAC** (generic traffic scenes)
2. Fine-tuning on **Vietnam traffic data** (domain adaptation)

The final model supports **real-time vehicle detection & tracking** and is optimized for **deployment-oriented inference** using **ONNX Runtime**.


## 🎯 Objectives

* Detect and track traffic participants (motorbike, car, bus, truck)
* Improve performance on **Vietnamese traffic scenes**
* Optimize model for **edge / low-resource environments**
* Align with real-world traffic AI applications


## 🧠 Model & Techniques

* **Model**: YOLOv11n (Ultralytics)
* **Tasks**:

  * Object Detection
  * Multi-object Tracking (ByteTrack)
* **Optimization**:

  * Lightweight architecture selection
  * ONNX export for deployment
  * Inference benchmarking (latency & memory)
* **Deployment mindset**:

  * Batch = 1 inference
  * Video stream processing
  * CPU / GPU compatible


## 📂 Datasets

### 1️⃣ UA-DETRAC

* Large-scale traffic dataset (cars, buses, trucks, vans)
* High-resolution frames extracted from surveillance videos
* Used for **pretraining** to learn generic traffic features

### 2️⃣ Vietnam Traffic Dataset

* Real traffic videos from Vietnam (dense motorbikes, mixed vehicles)
* Used for **fine-tuning** to adapt model to local domain
* Resolution: 1280×720 (auto-resized during training)

📌 **Reason for two-stage training**
Vietnam traffic distribution is very different from global datasets.
Pretraining helps stabilize learning, while fine-tuning improves real-world accuracy.


## 🏗️ Project Structure

```text
FPT/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
├── vietnam traffic vids/
│   ├── nguyentraihn.mp4
│   ├── chuongduonghn.mp4
│   └── ...
├── runs/
│   └── detect/
│       ├── train/
│       ├── train2/
│       ├── train3/
│       └── train4/   # final finetuned model
├── yolo.py           # training
├── tracking_yolo.py  # tracking + video output
├── export_onnx.py    # ONNX export
├── benchmark_onnx.py # inference benchmarking
└── README.md
```

## 🏋️ Training

### Pretraining (UA-DETRAC)

```python
model = YOLO("yolo11n.pt")
model.train(
    data="data.yaml",
    imgsz=640,
    batch=16,
    epochs=25,
    device=0
)
```

### Fine-tuning (Vietnam dataset)

* Lower learning rate
* Traffic-aware augmentation
* Preserve pretrained features


## 🚗 Tracking (ByteTrack)

```python
model.track(
    source="vietnam traffic vids/nguyentraihn.mp4",
    tracker="bytetrack.yaml",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=0,
    save=True
)
```

✔ Outputs video with:

* Bounding boxes
* Object IDs (tracking)
* Per-frame inference


## ⚙️ Model Optimization & Deployment

### ONNX Export

```python
model.export(format="onnx", opset=12)
```

### ONNX Runtime Inference

* Batch size = 1
* CPU / CUDA Execution Provider
* Suitable for **edge-like environments**

### Benchmarking

* Warm-up + repeated inference
* Measure:

  * Preprocess time
  * Inference latency
  * Postprocess time

This simulates **real-world edge deployment constraints**.


## 📊 Results (Vietnam Fine-tuned Model)

| Class     | mAP@50 | mAP@50-95 |
| --------- | ------ | --------- |
| Motorbike | ~0.95  | ~0.61     |
| Car       | ~0.95  | ~0.71     |
| Bus       | ~0.94  | ~0.76     |
| Truck     | ~0.94  | ~0.71     |


## 🔍 Key Learnings

* Domain gap between global datasets and Vietnam traffic is significant
* Pretrain → finetune strategy is crucial
* ONNX greatly reduces deployment friction
* Tracking adds strong practical value for traffic analysis
* Memory & latency must be considered early for edge deployment

## 🚀 Future Work

* INT8 quantization with TensorRT
* Pruning for further model compression
* Deployment on embedded devices (Jetson / AI camera)
* Traffic analytics (counting, speed estimation, ANPR)