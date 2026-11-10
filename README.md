# DogBreedID

**DogBreedID** là đồ án môn Thị giác máy tính, tập trung xây dựng hệ thống **phân loại giống chó từ hình ảnh** với 30 giống được lựa chọn trước.

Đề tài sử dụng và so sánh các mô hình **Custom CNN, MobileNetV2 và ResNet18**, đồng thời đánh giá độ chính xác, hiệu quả mô hình, độ tin cậy dự đoán và khả năng hoạt động khi ảnh bị nhiễu.

Mục tiêu cuối cùng là chọn mô hình phù hợp và tích hợp vào một **ứng dụng Gradio** để người dùng tải ảnh chó lên và nhận kết quả dự đoán giống chó.

## Chạy demo Gradio trên máy cá nhân

Ứng dụng sử dụng E4 MobileNetV2 Frozen + CutMix, seed 42, cùng Temperature E6 và ngưỡng E7 đã khóa. Mô hình chỉ phân loại trong 30 giống đã chọn. Đối với REJECT, hệ thống chỉ cho biết dự đoán chưa đủ tự tin; không xác định ảnh có chó hay giống ngoài 30 lớp.

### Chuẩn bị

1. Dùng Python 3.10 trở lên và tạo môi trường riêng. Ví dụ trên Windows:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Cài PyTorch và torchvision tương thích với nhau và với CPU/CUDA của máy; sau đó cài các thư viện của dự án. Ví dụ môi trường CPU đã được kiểm thử trên Windows/Python 3.13:

   ```powershell
   python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   ```

   Với CUDA, chọn cặp PyTorch/torchvision phù hợp theo hướng dẫn chính thức của PyTorch. Demo tự chọn CUDA nếu khả dụng, nếu không dùng CPU.

3. Lấy `best.pt` của E4 seed 42 từ [thư mục checkpoint được nhóm chỉ định](https://drive.google.com/drive/folders/1eZ6h8nWZfUnUDozvC8hd55DZRk8oCIXJ) và đặt tại `outputs/checkpoints/E4/seed_42/best.pt`. Không đưa checkpoint lên GitHub. SHA-256 yêu cầu:

   ```text
   73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b
   ```

   Bộ nạp kiểm tra hash và nạp trọng số với `strict=True`; tệp thiếu hoặc sai hash không được sử dụng. Giữ nguyên class mapping, `configs/data.yaml` và artifact E6/E7 đã khóa trong repository.

### Khởi chạy và sử dụng

```powershell
python scripts/run_demo.py
```

Mở `http://127.0.0.1:7860` trong trình duyệt. Có thể dùng `--device cpu` hoặc `--device cuda`, `--port 7861`, và `--checkpoint <đường-dẫn>` khi cần. Ứng dụng chỉ lắng nghe trên máy cá nhân và không tạo public share link.

Tải ảnh JPG/PNG/WebP tối đa 10 MB và 25 triệu điểm ảnh. Chọn chế độ xử lý trong mục **Ảnh đầu vào**. Giao diện hiển thị ảnh gốc, còn preview 224 × 224 dùng cho kiểm thử nội bộ.

- **Chuẩn E4 — CenterCrop** (`standard`, mặc định) dùng pipeline Resize(256) → CenterCrop(224) đã sử dụng trong Official Evaluation.
- **Giữ trọn ảnh — Padding** (`fit`) thu ảnh theo đúng tỷ lệ và thêm nền RGB `(124, 116, 104)` để giữ toàn bộ bố cục. Đối tượng có thể nhỏ hơn trong khung 224 × 224. Đây là chế độ thử nghiệm ở cấp ứng dụng. Cần đo accuracy, ECE và E7 Selective Accuracy trên cùng Validation split trước khi kết luận. Temperature E6 và ngưỡng E7 vẫn dùng giá trị đã khóa, nhưng hiệu quả hiệu chỉnh và chọn lọc trên ảnh Padding chưa được kiểm chứng. Các kết quả Padding không thay thế chỉ số Official E4.

Nhấn **Dự đoán** để xem Top-1/Top-5, confidence và ACCEPT/REJECT trong mục **Kết quả phân loại**. Thời gian hiển thị chỉ đo lượt chạy model forward. Cuộn xuống **03 · Giải thích dự đoán bằng Grad-CAM** rồi nhấn **Xem Grad-CAM** để xem overlay trên cùng ảnh 224 × 224 mà mô hình nhận. Đổi ảnh hoặc chế độ sẽ xóa kết quả cũ và yêu cầu dự đoán lại. **Xóa ảnh và kết quả** xóa ảnh và kết quả, đồng thời giữ chế độ đã chọn. Grad-CAM hỗ trợ quan sát, không chứng minh nguyên nhân dự đoán.

Nếu giao diện báo thiếu hoặc sai checkpoint, kiểm tra lại đường dẫn và hash, rồi khởi động lại ứng dụng. Ảnh bị lỗi, không thuộc JPG/PNG/WebP hoặc vượt giới hạn sẽ được từ chối. Danh sách 30 giống và tham số hiệu chỉnh không được chỉnh bằng giao diện.

### API local

`/predict` nhận ảnh và tham số `mode` tùy chọn. Lời gọi chỉ truyền ảnh tiếp tục dùng `standard`. Giao diện và API dùng chung hàm dự đoán.

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
image = handle_file("dog.jpg")
standard = client.predict(image, api_name="/predict")
fit = client.predict(image, mode="fit", api_name="/predict")
```

Xem thứ tự kết quả trả về và cách kiểm tra hai chế độ trong [tài liệu chế độ Padding](docs/task06/gradio/full_image_padding_mode.md).

### Kiểm thử

```powershell
python -m pytest tests/test_gradio_app.py tests/test_gradcam.py tests/test_transforms.py -q
```

Hướng dẫn nghiệm thu và ảnh giao diện nằm trong `docs/task06/gradio/`. Không chạy lại Official Test để kiểm thử demo.
