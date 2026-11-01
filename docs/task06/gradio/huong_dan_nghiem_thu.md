# Nghiệm thu Gradio — Task 06

Ảnh giao diện lúc chưa tải ảnh: [giao_dien.png](giao_dien.png). Ảnh này không chứa dữ liệu Official hay ảnh người dùng.

Ví dụ thông báo với tệp PNG bị cắt cụt: [anh_hong.png](anh_hong.png). Tệp kiểm thử do chương trình tạo, không chứa ảnh thật hoặc dữ liệu Official.

Ghi nhận duy nhất gửi trưởng nhóm: [resize/crop hụt đối tượng trên ảnh Doberman, kèm ảnh trước–sau và Grad-CAM](nhan_xet_resize_crop.md).

## Chạy demo

Làm theo phần "Chạy demo Gradio trên máy cá nhân" trong README. Đặt đúng checkpoint E4 seed 42 đã được nhóm phê duyệt, rồi chạy:

```bash
python scripts/run_demo.py
```

Mở `http://127.0.0.1:7860`. Có thể chỉ định `--device cpu`, `--device cuda`, `--port`, hoặc `--checkpoint`. Ứng dụng không tạo public share link.

## Quy trình kiểm tra thủ công

1. Tải một ảnh JPG/PNG/WebP được phép sử dụng, nhấn Dự đoán, kiểm tra Top-1/Top-5, confidence, ACCEPT/REJECT và thời gian.
2. Nhấn Xem Grad-CAM, kiểm tra tên lớp được giải thích và overlay trên vùng center crop của ảnh.
3. Xóa, chọn ảnh khác và dự đoán lại; kiểm tra kết quả cũ không còn hiển thị. Kiểm tra trạng thái REJECT nếu có.
4. Kiểm tra chưa tải ảnh, ảnh lỗi/không đúng định dạng, thiếu checkpoint và checkpoint sai hash. Sau khi thay tệp checkpoint, khởi động lại ứng dụng.
5. Đối chiếu nguồn E6/E7, mapping và SHA-256 checkpoint với tài liệu đã khóa. Không sử dụng hoặc chạy lại Official Test cho demo.

Thời gian trên giao diện chỉ đo model forward. Grad-CAM là bằng chứng hỗ trợ quan sát, không chứng minh nguyên nhân mô hình dự đoán; REJECT không nhận diện ảnh ngoài tập 30 lớp.

## Ghi nhận trên máy kiểm thử

Đã chạy ứng dụng trên CPU với checkpoint E4 thật, kiểm tra hash trước khi nạp; thử upload ảnh ví dụ từ PyTorch Hub, dự đoán, Grad-CAM, xóa, chưa chọn ảnh, ảnh GIF, file giả PNG, PNG bị cắt cụt, REJECT và dự đoán ảnh kế tiếp bằng trình duyệt. Không thực hiện lượt Official Test. Các kết quả chi tiết, giới hạn và các kiểm thử không đạt của repository được ghi tại `docs/task_reports/kiet/31_10_task.md`.
