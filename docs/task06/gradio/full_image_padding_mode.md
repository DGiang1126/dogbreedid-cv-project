# Chế độ Giữ trọn ảnh trong Gradio

**Chuẩn E4 — CenterCrop** là mặc định. Đây là pipeline RGB → Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize đã dùng trong Official Evaluation. Resize đưa cạnh ngắn về 256 px và giữ tỷ lệ ảnh. CenterCrop chỉ lấy khung 224 × 224 ở giữa. Với ảnh rộng, ảnh dọc hoặc chủ thể lệch tâm, nhiều chi tiết ở mép có thể nằm ngoài khung này.

**Giữ trọn ảnh — Padding** là chế độ thử nghiệm ở cấp ứng dụng. Sau khi kiểm tra và đọc ảnh RGB, ứng dụng resize bicubic theo một hệ số chung để cạnh dài đạt 224 px. Cạnh ngắn được làm tròn và giữ tối thiểu một pixel. Ảnh được dán vào giữa vùng 224 × 224 của canvas RGB 256 × 256, với nền `(124, 116, 104)` xấp xỉ ImageNet mean. Nếu khoảng trống còn lại là số lẻ, mép phải hoặc mép dưới nhận thêm một pixel. Transform E4 dùng chung sau đó chỉ cắt 16 px ngoài cùng mỗi cạnh canvas và giữ toàn bộ ảnh đã thu nhỏ.

Preview nội bộ 224 × 224 chứa pixel RGB ngay trước ToTensor và Normalize. Giao diện chính hiển thị ảnh gốc. Predict và Grad-CAM dùng cùng mode và cùng quy trình chuẩn bị ảnh. Dữ liệu dự đoán trong session lưu mode và SHA-256 của tệp ảnh gốc. Đổi ảnh hoặc chế độ xóa dự đoán và Grad-CAM cũ. Nút **Xóa ảnh và kết quả** xóa ảnh và kết quả, đồng thời giữ chế độ hiện tại.

## Kiểm tra với ảnh Doberman của Kiệt

1. Chạy `python scripts/run_demo.py --device cpu` rồi mở `http://127.0.0.1:7860`.
2. Tải [ảnh Doberman gốc 750 × 402](crop_doberman/doberman-original.webp). Ở **Chuẩn E4 — CenterCrop**, nhấn **Dự đoán**, cuộn xuống **03 · Giải thích dự đoán bằng Grad-CAM** rồi nhấn **Xem Grad-CAM**. Sau Resize, ảnh có kích thước 477 × 256. Vùng crop trên overlay bỏ một phần mõm, thân và chân ở mép ảnh.
3. Chuyển sang **Giữ trọn ảnh — Padding**. Kết quả cũ được xóa và **Xem Grad-CAM** yêu cầu dự đoán lại. Nhấn **Dự đoán**, rồi **Xem Grad-CAM** để kiểm tra bố cục 224 × 120 với hai dải nền 52 px ở trên và dưới.
4. Nhấn **Xóa ảnh và kết quả**. Kiểm tra ảnh gốc, kết quả và Grad-CAM đều trống, radio vẫn ở Padding.

Doberman nằm ngoài 30 lớp của hệ thống. Ví dụ này chỉ dùng để kiểm tra vùng ảnh được giữ lại.

## API `/predict`

Dùng `client.predict(handle_file(path), api_name="/predict")` như trước hoặc thêm `mode="fit"`. Các giá trị hợp lệ là `standard` và `fit`. Endpoint dùng cùng hàm với nút **Dự đoán** và giữ thứ tự bảy giá trị trả về gồm trạng thái, giống hiển thị, confidence, latency, Top-5, ảnh Grad-CAM (trống) và chú thích Grad-CAM (trống). Preview, Grad-CAM và Clear là sự kiện nội bộ của giao diện.

## Giới hạn đánh giá

Padding giữ toàn bộ bố cục, đồng thời thêm nền và có thể làm đối tượng nhỏ hơn trong ảnh đầu vào. Accuracy, ECE và E7 Selective Accuracy của Padding chưa được đo. Hai chế độ dùng chung checkpoint E4 seed 42, Temperature E6 `0.80246283961825` và ngưỡng E7 `0.5011672918550012`. Hiệu quả calibration và selective prediction trên ảnh Padding cần được kiểm tra trên cùng Validation split trước khi kết luận. Các kết quả Padding không dùng để thay thế chỉ số Official của E4.
