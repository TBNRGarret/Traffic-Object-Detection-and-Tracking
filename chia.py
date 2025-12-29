import os
import shutil

# CẤU HÌNH
source_folder = r"D:\nighttime-dataset\nighttime"  # Tên folder đang chứa lộn xộn
output_folder = "nighttime" # Tên folder mới sẽ tạo ra

def organize_dataset():
    # 1. Tạo cấu trúc thư mục chuẩn
    images_path = os.path.join(output_folder, "images")
    labels_path = os.path.join(output_folder, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # 2. Duyệt qua file và di chuyển
    files = os.listdir(source_folder)
    count = 0
    
    for f in files:
        src = os.path.join(source_folder, f)
        
        if f.endswith(".jpg") or f.endswith(".png"):
            # Chuyển ảnh vào folder images
            shutil.move(src, os.path.join(images_path, f))
            count += 1
        elif f.endswith(".txt"):
            # Chuyển label vào folder labels
            shutil.move(src, os.path.join(labels_path, f))

    print(f"✅ Xong! Đã xử lý {count} cặp ảnh/label.")
    print(f"👉 Dữ liệu chuẩn nằm ở: {output_folder}")

if __name__ == "__main__":
    organize_dataset()