# 📝 Hệ Thống Nhận Dạng Chữ Viết (OCR) - CRNN & Streamlit

Đây là đồ án môn học ứng dụng Trí tuệ Nhân tạo để xây dựng hệ thống nhận dạng chữ viết (Optical Character Recognition - OCR). Dự án sử dụng kiến trúc mô hình **CRNN (CNN + RNN + CTC Loss)** kết hợp với giao diện Web tương tác trực quan.

## 🌟 Tính Năng Nổi Bật

* **Lõi AI Mạnh Mẽ:** Sử dụng PyTorch để huấn luyện mô hình CRNN trên tập dữ liệu LMDB, đạt độ chính xác cao (~98%) trong việc nhận dạng chuỗi ký tự.
* **Giao Diện Web Thân Thiện:** Tích hợp `Streamlit` cho phép người dùng dễ dàng tải ảnh lên và nhận kết quả tức thì ngay trên trình duyệt mà không cần sử dụng dòng lệnh.
* **Tối Ưu Hóa Thiết Bị:** Hệ thống tự động nhận diện và sử dụng Card đồ họa (GPU/CUDA) nếu có để tăng tốc độ suy luận, hoặc chạy mượt mà trên CPU.

## 🛠️ Cài Đặt Môi Trường

**Bước 1:** Clone kho chứa này về máy:
```bash
git clone [https://github.com/Huynguyen2809/Optical-Character-Recognition.git]
cd OCR

Bước 2: Cài đặt các thư viện cần thiết:

pip install -r requirements.txt

Hướng Dẫn Sử Dụng

Bước 1: Tạo môi trường ảo:
python -m venv venv
venv\Scripts\activate

Bước 2: Sử dụng lệnh sau để khởi động trang web trên trình duyệt:
streamlit run app.py

