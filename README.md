# AI Photo Studio: Intelligent Video Production Pipeline

**AI Photo Studio** là một hệ thống tự động hóa quy trình phân tích hình ảnh và sản xuất video chuyên nghiệp, tích hợp trí tuệ nhân tạo để chọn lọc những khoảnh khắc đẹp nhất từ thư viện ảnh của bạn.

---

## Tính năng nổi bật

### 1. Phân tích AI đa tầng (Multi-model Analysis)
Hệ thống sử dụng các mô hình AI hiện đại nhất để đánh giá từng bức ảnh theo nhiều tiêu chí:
- **Face Analysis (MediaPipe)**: Nhận diện nụ cười (Smile score), kiểm tra trạng thái mắt (Eye open/closed) và bố cục khuôn mặt.
- **Aesthetic Scoring (CLIP + ML Head)**: Đánh giá tính thẩm mỹ, nghệ thuật và bố cục dựa trên mô hình học sâu của OpenAI.
- **Technical Metrics (OpenCV)**: Đo lường độ sắc nét (Sharpness) và độ phơi sáng (Exposure) bằng các thuật toán xử lý ảnh số.

### 2. Tự động hóa chọn lọc (Smart Filtering)
- **AI Recommendation**: Hệ thống tự động gắn nhãn "Được chọn" cho những ảnh đạt điểm cao nhất dựa trên độ nhạy (Sensitivity) tùy chỉnh.
- **Mood-based Optimization**: Điều chỉnh tiêu chí lọc theo chủ đề: Gia đình, Cá nhân hoặc Nhóm.

### 3. Dựng phim tự động (Automated Slideshow)
- Tạo video slideshow chất lượng cao với các hiệu ứng chuyển cảnh mượt mà.
- Tích hợp nhạc nền (BGM) đa dạng và đồng bộ hóa với tâm trạng (Mood) của bộ ảnh.
- Tối ưu hóa thời lượng dựa trên số lượng ảnh được chọn.

### 4. Giao diện Premium (Glassmorphism UI)
- Trải nghiệm người dùng cao cấp với phong cách thiết kế **Glassmorphism**.
- Hiệu ứng **Neon Glow** cho các nút điều khiển và micro-animations mượt mà.
- Chuyển Tab thông minh bằng JavaScript, tối ưu cho **Gradio 6**.

---

## Công nghệ sử dụng

- **Core Logic**: Python 3.11+
- **Frontend**: Gradio 6 (Custom CSS & JS)
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch, OpenCLIP
- **Video Engine**: MoviePy / FFmpeg

---

## Cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi chạy ứng dụng
```bash
python app.py
```
Sau đó truy cập địa chỉ `http://localhost:7860` trên trình duyệt.

---

## Tối ưu hóa phần cứng
Dự án được thiết kế để chạy mượt mà ngay cả trên các dòng GPU phổ thông:
- Hỗ trợ **FP16 Mixed Precision** để tiết kiệm VRAM.
- Cơ chế **Memory Cleanup** tự động giải phóng bộ nhớ sau mỗi phiên phân tích.
- Xử lý Batch thông minh tránh lỗi OOM (Out of Memory).
