# Phân loại Biển báo Giao thông (Traffic Sign Classification)

## Mục tiêu
Xây dựng một mô hình học sâu để phân loại các biển báo giao thông từ ảnh, sử dụng tập dữ liệu **German Traffic Sign Recognition Benchmark (GTSRB)**.

---

## Yêu cầu dự án

- **Tập dữ liệu:** Sử dụng tập dữ liệu GTSRB (~700MB) với cấu trúc thư mục:
  - `meta/`: chứa các ảnh
  - `test/`: chứa các ảnh kiểm tra
  - `train/`: chứa các thư mục con, mỗi thư mục là một lớp biển báo, trong đó có các ảnh tương ứng
  - Các file `.csv` đi kèm chứa nhãn và meta dữ liệu.

- **Nhiệm vụ:** Xây dựng mô hình CNN để phân loại ảnh biển báo giao thông thành các lớp tương ứng.

- **Thư viện sử dụng:**
  - `tensorflow.keras` (xây dựng và huấn luyện mô hình)
  - `opencv` (xử lý ảnh)
  - `matplotlib` và `seaborn` (trực quan hóa dữ liệu và kết quả)

---

## Kết quả đầu ra

- Tiền xử lý ảnh: resize, tăng cường dữ liệu.
- Phân tích dữ liệu khám phá (EDA).
- Xây dựng và huấn luyện mô hình CNN.
- Đánh giá mô hình bằng độ chính xác và ma trận nhầm lẫn.
- Trực quan hóa các biểu đồ EDA và kết quả đánh giá.

---

## Các bước thực hiện

### 1. Chuẩn bị dữ liệu
- Tải dữ liệu GTSRB (train & test).
- Xử lý đúng cấu trúc dữ liệu.
- Resize ảnh về kích thước chuẩn (32x32 hoặc 64x64).
- Tăng cường dữ liệu (xoay, lật, điều chỉnh độ sáng).
- Chuẩn hóa giá trị pixel về [0, 1].

### 2. Phân tích Dữ liệu Khám phá (EDA)
- Trực quan hóa ảnh mẫu các lớp.
- Biểu đồ phân bố số lượng ảnh mỗi lớp.
- Xem xét mất cân bằng dữ liệu, đề xuất xử lý.

### 3. Xây dựng mô hình CNN
- Kiến trúc ví dụ:
  - Conv2D + ReLU + MaxPooling2D (nhiều lớp)
  - Flatten + Dense + Dropout
  - Lớp đầu ra softmax phân loại đa lớp
- Compile với loss `categorical_crossentropy`, optimizer `Adam`.

### 4. Huấn luyện mô hình
- Chia train/validation (80/20).
- Huấn luyện 20–50 epoch, batch size 32/64.
- Sử dụng early stopping.
- Lưu model tốt nhất.

### 5. Đánh giá mô hình
- Độ chính xác trên tập test.
- Ma trận nhầm lẫn.
- Biểu đồ accuracy/loss qua các epoch.

### 6. Kết quả đầu ra
- Mô hình đã huấn luyện (.h5).
- Mã nguồn đầy đủ (Python/Jupyter Notebook).
- Báo cáo tóm tắt kết quả, phân tích, đề xuất cải tiến.

---

## Ghi chú bổ sung
- Mã có chú thích rõ ràng.
- Sử dụng OpenCV để tải và resize ảnh.
- Lưu ảnh trực quan dưới dạng `.png`.
- Xử lý mất cân bằng lớp (ví dụ dùng trọng số lớp).
- Hiển thị dự đoán mẫu trên ảnh kiểm tra.

---

## Tiêu chí thành công
- Độ chính xác test ≥ 90%.
- Hình ảnh trực quan chất lượng cao.
- Mã chạy không lỗi, dễ tái tạo.

---

## Cấu trúc thư mục
traffic_sign_classification/
├── data/ # Dữ liệu gốc (GTSRB)
├── processed_data/ # Ảnh đã resize và phân chia
├── visualizations/ # Lưu biểu đồ và hình ảnh trực quan
├── models/ # Lưu model .h5
├── traffic_sign_classifier.ipynb # File notebook chính
└── utils.py # Tiện ích (nếu cần)