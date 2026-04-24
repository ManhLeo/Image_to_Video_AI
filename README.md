# VibeFlow AI Photo Studio

**VibeFlow AI Photo Studio** là một hệ thống sản xuất video tự động từ ảnh, tích hợp các mô hình Computer Vision hiện đại để chọn lọc những khoảnh khắc chất lượng nhất. Hệ thống được tối ưu hóa cho môi trường sản xuất thực tế với kiến trúc 2 giai đoạn (Stage 1: Filtering & Stage 2: Ranking).

---

## Các mô hình AI sử dụng

Hệ thống kết hợp sức mạnh của nhiều mô hình AI để phân tích ảnh toàn diện:

1.  **Aesthetic Analysis (Thẩm mỹ)**:
    *   **Model**: CLIP (ViT-B-32) từ OpenAI.
    *   **Custom Head**: Một Linear Head được huấn luyện trên tập dữ liệu LAION-Aesthetics/AVA để dự đoán điểm số thẩm mỹ nghệ thuật (composition, color, professional look).
2.  **Face & Human Analysis (Khuôn mặt)**:
    *   **Model**: MediaPipe FaceLandmarker (Tasks API).
    *   **Metrics**: Trích xuất EAR (Eye Aspect Ratio) để đo độ mở mắt và tỉ lệ khuôn miệng để tính điểm nụ cười (Smile score).
3.  **Technical Quality (Chất lượng kỹ thuật)**:
    *   **Sharpness**: Sử dụng thuật toán Laplacian Variance để đo độ nét của các cạnh.
    *   **Lighting/Exposure**: Phân tích Histogram và Mean Brightness để phát hiện ảnh cháy sáng hoặc quá tối.
4.  **Scene & Context Aware**:
    *   Sử dụng CLIP embeddings để phân loại bối cảnh (Portrait, Landscape, Outdoor, Group) nhằm điều chỉnh trọng số chấm điểm linh hoạt.

---

## Quy trình chấm điểm 2 giai đoạn (Two-Stage Pipeline)

Để đảm bảo độ chính xác và tránh bias, hệ thống tách biệt hoàn toàn việc **Loại bỏ** và **Xếp hạng**.

### Giai đoạn 1: Lọc cứng (Hard Filtering)
Mọi ảnh phải vượt qua các "cổng gác" sau để không bị loại:
-   **Độ nét (Sharpness)**: Score < 30 → **Loại** (Ảnh quá mờ).
-   **Mở mắt (Eyes)**: Score < 30 → **Loại** (Nhắm mắt hoàn toàn).
-   **Nhận diện mặt**: Không thấy mặt → **Loại**.
    -   *Ngoại lệ (Rescue)*: Nếu điểm **Thẩm mỹ > 60**, ảnh sẽ được giữ lại dù không thấy mặt (ảnh phong cảnh, decor đẹp).

### Giai đoạn 2: Xếp hạng (Soft Scoring)
Sau khi vượt qua vòng lọc, điểm số cuối cùng được tính theo công thức:
```python
Final_Score = (50% Aesthetic) + (25% Sharpness) + (25% Lighting)
```
**Điều chỉnh nâng cao:**
-   **Penalty (Phạt)**: Nếu mắt hơi nhắm (30 ≤ Eyes < 50) → **Trừ 5 điểm**.
-   **Bonus (Thưởng)**: Nếu nụ cười tươi (> 80) → **Cộng 1.5 điểm**.

---

## Tối ưu hóa phần cứng (Low VRAM)

Dự án được thiết kế đặc biệt để chạy tốt trên các GPU 4GB VRAM:
-   **FP16 Mode**: Chạy CLIP inference ở chế độ bán chính xác để giảm 50% bộ nhớ.
-   **Aggressive Cleanup**: Tự động giải phóng cache Torch và thu hồi bộ nhớ ngay sau khi kết thúc Phase GPU.
-   **Batch Processing**: Tự động điều chỉnh kích thước Batch nếu phát hiện nguy cơ OOM (Out of Memory).

---

## Cài đặt & Khởi chạy

### 1. Cài đặt
```bash
# Tạo và kích hoạt venv
python -m venv venv
venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Sử dụng
1. Đặt trọng số local (nếu có) tại: `models/aesthetic_weights.pth`.
2. Chạy ứng dụng: `python app.py`.
3. Tải ảnh lên, chọn "Ưu tiên" và nhấn **Phân tích AI**.
4. Nếu chưa có model: Download tại link https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_b_32_linear.pth và đổi tên model thành aesthetic_weights.pth
---
*Phát triển bởi Antigravity AI Engineering Team.*
