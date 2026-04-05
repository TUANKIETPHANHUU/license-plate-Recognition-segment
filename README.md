# Phát hiện và Nhận dạng Biển số xe Việt Nam (ALPR) bằng YOLOv8

Dự án này ứng dụng mô hình **YOLOv8** để tự động hóa quy trình phát hiện và nhận diện biển số xe tại Việt Nam. Đây là giải pháp cốt lõi cho các hệ thống quản lý bãi đỗ xe thông minh, trạm thu phí tự động hoặc giám sát giao thông.

---

## 📌 Tóm tắt dự án (Abstract)
Hệ thống được thiết kế để xử lý hình ảnh đầu vào, xác định vị trí biển số và trích xuất nội dung văn bản dựa trên quy trình:
1. **Phát hiện (Detection):** Sử dụng YOLOv8 để định vị vùng biển số (LP).
2. **Phân đoạn (Segmentation):** Tách từng ký tự (chữ và số) từ vùng biển số đã cắt.
3. **Nhận diện (Recognition):** Phân loại từng ký tự để chuyển thành văn bản thuần túy.

---

## 📂 Cấu trúc thư mục (Project Structure)

```text
license-plate-Recognition-segment/
├── src/                        # Chứa toàn bộ mã nguồn xử lý
│   ├── char_classification/    # Model/Logic phân loại ký tự
│   ├── lp_detection/           # Model/Logic phát hiện biển số
│   ├── weights/                # Chứa các file trọng số (.pt) của YOLOv8
│   ├── data_utils.py           # Các hàm bổ trợ (xử lý ảnh, vẽ khung)
│   └── lp_recognition.py       # Module tích hợp nhận diện tổng thể
├── sampledata/                 # Dữ liệu hình ảnh dùng để chạy thử (test)
├── app.py                      # Giao diện người dùng (Streamlit hoặc Flask)
├── main.py                     # File thực thi chính từ Terminal
├── requirements.txt            # Danh sách thư viện cần thiết
├── BaiDoAn.ipynb               # Notebook quá trình huấn luyện & thử nghiệm
└── data.zip                    # Bộ dữ liệu nén (dataset)

