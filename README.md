# 🚦 Vietnam Traffic Object Detection & Tracking (YOLOv11)

## 📌 Overview

Dự án phát triển hệ thống **Nhận diện & Theo dõi phương tiện giao thông Việt Nam** sử dụng mô hình học sâu **YOLOv11** kết hợp thuật toán tracking **ByteTrack**, tối ưu hóa suy luận biên qua **ONNX Runtime**, và vòng lặp học chủ động (**Active Learning Loop**) để làm giàu dữ liệu từ video thực tế.

---

## 🎯 Điểm nổi bật & Cải tiến hệ thống

1. **Dataset chuẩn hóa & Làm sạch**:
   - Gộp từ 3 nguồn dữ liệu giao thông VN (`CanTho`, `vn1`, `vehicle`).
   - Lọc sạch 100% ảnh trùng lặp và triệt tiêu **data leakage** giữa train/val/test (Dataset sạch: 14,155 ảnh).
   - 4 nhãn chuẩn: `0: motorbike`, `1: car`, `2: bus`, `3: truck`.
2. **Chiến lược huấn luyện trực tiếp (Direct Fine-tuning)**:
   - Fine-tune thẳng từ trọng số COCO-pretrained (`yolo11n.pt` / `yolo11s.pt`) sang dữ liệu giao thông VN.
   - Traffic-aware data augmentations: Cosine LR, Mosaic (tắt 10 epochs cuối), Mixup, Copy-paste.
3. **Vòng lặp làm giàu dữ liệu (Active Learning)**:
   - Khai thác mẫu khó (uncertainty mining, confidence 0.15 ~ 0.45) trực tiếp từ các video giao thông gốc.
   - Tự động sinh nhãn dự đoán (pseudo-labels) để review nhanh và mở rộng tập huấn luyện.
4. **Chuẩn hóa toàn diện**:
   - 100% đường dẫn sử dụng tương đối (portable), không phụ thuộc môi trường máy cục bộ.
   - Hỗ trợ CLI arguments linh hoạt trên mọi script.

---

## 🏗️ Cấu trúc thư mục dự án

```text
d:\Computer Vis Proj\
└── Traffic-Object-Detection-and-Tracking\     ← DUY NHẤT 1 THƯ MỤC DỰ ÁN
    ├── dataset\                               ← ⭐ GỘP TẤT CẢ DỮ LIỆU TẠI ĐÂY (git ignored)
    │   ├── images\{train,val,test}\           ← 14,155 ảnh sạch
    │   ├── labels\{train,val,test}\           ← 14,155 nhãn sạch
    │   ├── videos\                            ← Video đường phố & camera giám sát
    │   └── archive\                           ← UA-DETRAC + dữ liệu thô dự phòng
    ├── weights\                               ← Trọng số model (.pt)
    ├── runs\detect\                           ← Output log huấn luyện & checkpoints
    ├── train_direct_vn.py                     ← ⭐ Script train trực tiếp trên VN dataset
    ├── enrich_data.py                         ← ⭐ Vòng lặp Active Learning
    ├── vn_dataset.yaml                        ← Config dataset chuẩn hóa
    ├── tracking_yolo.py                       ← Multi-object tracking (ByteTrack)
    ├── export_onnx.py                         ← Xuất mô hình sang định dạng ONNX
    ├── benchmark_onnx.py                      ← Đánh giá độ trễ & FPS trên CPU/Edge
    ├── onnx_testing.py                        ← Test suy luận mô hình ONNX
    ├── framecutter.py                         ← Trích xuất frame từ video
    ├── scriptprelabel.py                      ← Tự động gán nhãn hàng loạt
    ├── Dockerfile & compose.yaml              ← Containerization
    └── requirements.txt                       ← Thư viện phụ thuộc
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
# Hoặc cài đặt trực tiếp Ultralytics:
pip install ultralytics
```

### 2. Huấn luyện mô hình trực tiếp (Phase 3)

```bash
python train_direct_vn.py
```
> Trọng số tốt nhất tự động lưu vào `runs/detect/vn_direct/weights/best.pt`.

### 3. Nhận diện & Theo dõi (Tracking)

```bash
# Chạy với video mặc định
python tracking_yolo.py

# Hoặc truyền video và model tùy chọn
python tracking_yolo.py --source "dataset/videos/vietnam_traffic_vids/chuongduonghn.mp4" --conf 0.3
```

### 4. Vòng lặp làm giàu dữ liệu (Active Learning)

```bash
python enrich_data.py \
  --video "dataset/videos/vietnam_traffic_vids/nguyentraihn.mp4" \
  --model "runs/detect/vn_direct/weights/best.pt" \
  --conf-low 0.15 \
  --conf-high 0.45
```
Các khung hình khó và nhãn đề xuất sẽ được lưu vào thư mục `enrichment/<video_name>/` để review trước khi đưa vào tập train.

### 5. Xuất ONNX & Benchmark suy luận

```bash
# Xuất ONNX
python export_onnx.py --model "runs/detect/vn_direct/weights/best.pt"

# Đo độ trễ suy luận trên CPU
python benchmark_onnx.py
```

---

## 📊 Phân bố dữ liệu sau lọc sạch (Deduplicated)

| Tập | Số lượng ảnh | Xe máy (0) | Ô tô (1) | Xe buýt (2) | Xe tải (3) |
|:---|:---|:---|:---|:---|:---|
| **Train** | 11,117 | Đa số | Trung bình | Ít | Ít |
| **Validation** | 2,751 | Đa số | Trung bình | Ít | Ít |
| **Test** | 287 | Cân đối | Cân đối | Cân đối | Cân đối |
| **Tổng** | **14,155** | Không rò rỉ dữ liệu chéo (0 leakage) |