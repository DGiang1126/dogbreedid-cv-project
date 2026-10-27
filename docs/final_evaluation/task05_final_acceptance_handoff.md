# DogBreedID — Task 05: Tổng kết nghiệm thu kỹ thuật và bàn giao nhóm

> **Loại tài liệu:** Đồ án môn học — báo cáo nội bộ và bàn giao công việc, không phải báo cáo khóa luận tốt nghiệp.  
> **Ngày tổng hợp:** 27/10/2026  
> **Trạng thái:** **HOÀN THÀNH THỰC THI KỸ THUẬT** theo kết quả backend và log do trưởng nhóm cung cấp; **chờ trưởng nhóm xác nhận nghiệm thu nội bộ cuối cùng**.  
> **Nguyên tắc:** Không chạy lại Official Test, không dùng Official Test để điều chỉnh mô hình hay ngưỡng quyết định.

## 1. Mục tiêu và phạm vi Task 05

Task 05 thực hiện đánh giá cuối của hệ thống phân loại **30 giống chó** trên tập **Official Test** theo cấu hình đã khóa, gồm:

- **E4 — Classification:** Top-1 Accuracy, Top-5 Accuracy, Macro-F1.
- **E6 — Calibration:** Temperature Scaling, ECE, NLL và Multiclass Brier Score.
- **E7 — Selective Classification:** ACCEPT/REJECT, Coverage, Selective Accuracy, Selective Risk và AURC.
- **E8 — Robustness:** ảnh clean, Gaussian Blur, Gaussian Noise và Occlusion.

Task 05 không bao gồm phát triển lại mô hình, refit calibration, đổi threshold hoặc chạy lại Official Test để cải thiện điểm số.

## 2. Trạng thái thực thi chính thức

| Trường | Kết quả được báo cáo |
|---|---|
| Execution status | **PASS** |
| Execution code | `OFFICIAL_EXECUTION_COMPLETE` |
| Run ID | `task05-final-20261026-ea8c474981b14065af219fb43ee8e65e` |
| `claim_consumed` | `true` — quyền cho run này **đã sử dụng**, không được retry |
| Phạm vi | `E4`, `E6`, `E7`, `E8` |
| Official Test | **1.934 ảnh, 30 lớp** theo kết quả đã xuất |
| Output bundle | `outputs/final_evaluation_b2/task05-final-20261026-ea8c474981b14065af219fb43ee8e65e/` |

Kết quả tổng hợp này dựa trên log `PASS` và nội dung các tệp metrics do trưởng nhóm trích xuất. Chưa có kiểm toán độc lập lại toàn bộ artifact ở bước soạn tài liệu; việc nghiệm thu nội bộ cần dựa trên bundle gốc được bảo toàn.

## 3. Bằng chứng nguồn, cấu hình và khả năng tái lập

| Mốc | Git revision / bằng chứng |
|---|---|
| Dataset freeze (D) | `d0b6931250cc51088e7beded24737959ef9464cc` |
| Source freeze (S) | `02483f801277482fbe24c25048e6cc769a16295f` |
| Configuration Lock (A) | `90dc817b24a32d5038e282d012f21e85b40e72ed` — LOCKED |
| Reviewed Release (R) | `c3f3f17c5b08b1311d9a9394b12d962671afc361` |
| Canonical dataset reference | `data/splits/official_test.csv` |
| Canonical Git blob ID được chốt | `25af4d79d3b255df9a30db8f899f400d19910baa` |
| E4 checkpoint | `outputs/checkpoints/E4/seed_42/best.pt` |
| E4 checkpoint SHA-256 | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` |

**Mô hình/thiết lập đã khóa:** E4 MobileNetV2, seed 42, checkpoint đã chọn từ giai đoạn trước; E6 Temperature `0.80246283961825`; E7 threshold `0.5011672918550012`, nhận dự đoán khi `confidence >= threshold`; E8 theo cấu hình corruption đã được phê duyệt. Không chọn lại bất kỳ thông số nào trên Official Test.

**Runtime đã khóa:** Windows 11 `Windows-11-10.0.26300-SP0`, Python `3.12.4`, PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`, NumPy `2.5.3`, CUDA build `12.4`, `device=cuda`, batch size `32`, `num_workers=0`, `torch_num_threads=4`. GPU NVIDIA GTX 1650, driver `555.97`; phiên bản thư viện bổ sung và JSON 16 trường đầy đủ có trong Configuration Lock/approval.

**Các mốc nghiệm thu trước Official:** Phase A và B1 accepted; B2.1 thiết kế approved; B2.2-A authorization accepted; B2.2-B có **52 synthetic integration tests PASS** và source review kết luận `READY FOR SOURCE FREEZE`. Smoke test checkpoint E4 thật trên CUDA với một tensor giả đã `PASS`, logits `(1,30)` hữu hạn. Đây không phải các phép đo Official.

## 4. Kết quả E4 — Phân loại trên Official Test

| Chỉ số | Giá trị |
|---|---:|
| Số ảnh / số lớp | **1.934 / 30** |
| Top-1 Accuracy | **81,49%** (`0.8148913979530334`) |
| Số ảnh phân loại Top-1 đúng | **1.576 / 1.934** |
| Top-5 Accuracy | **99,53%** (`0.9953464269638062`) |
| Macro-F1 | **81,11%** (`0.8111032621745409`) |

**Nhận xét:** Mô hình phân loại đúng Top-1 phần lớn ảnh Official; Top-5 cao hơn đáng kể cho thấy nhãn đúng thường nằm trong nhóm năm dự đoán đứng đầu. Top-5 không thay thế Top-1 và không được trình bày là tỷ lệ dự đoán nhãn đầu tiên đúng.

## 5. Kết quả E6 — Hiệu chỉnh xác suất

Sử dụng **Temperature Scaling đã khóa** với `T = 0.80246283961825`; 15 bins ECE.

| Chỉ số | Trước Temperature Scaling | Sau Temperature Scaling |
|---|---:|---:|
| ECE | **7,49%** | **2,29%** |
| NLL | `0.549883` | `0.505778` |
| Multiclass Brier Score | `0.272403` | `0.262574` |
| Top-1 Accuracy | **81,49%** | **81,49%** |

ECE giảm khoảng **5,20 điểm phần trăm**. Kết quả `top1_prediction_unchanged=true` cho thấy hiệu chỉnh thay đổi phân bố độ tin cậy mà không đổi nhãn Top-1. Biểu đồ minh họa có sẵn: `reliability_diagram.png`.

## 6. Kết quả E7 — Phân loại có quyền từ chối

Threshold đã khóa `0.5011672918550012`, quy tắc `confidence >= threshold`, `threshold_reselected=false`.

| Chỉ số | Giá trị |
|---|---:|
| Tổng ảnh | **1.934** |
| ACCEPT | **1.736** |
| REJECT | **198** |
| Dự đoán đúng trong nhóm ACCEPT | **1.478** |
| Coverage | **89,76%** |
| Selective Accuracy | **85,14%** |
| Selective Risk | **14,86%** |
| AURC | `0.05665629453149997` |

Selective Accuracy cao hơn Top-1 toàn bộ tập khoảng **3,65 điểm phần trăm**, đổi lại hệ thống từ chối **10,24%** số ảnh. AURC sử dụng quy ước `grouped right-step over confidence tie groups`. Biểu đồ: `risk_coverage_curve.png`.

> **Lưu ý bắt buộc khi viết báo cáo:** **85,14% là Selective Accuracy trên 1.736 ảnh được chấp nhận, KHÔNG PHẢI Top-1 Accuracy trên toàn bộ Official Test.**

## 7. Kết quả E8 — Độ bền vững với biến đổi ảnh

| Điều kiện | Số ảnh đúng / 1.934 | Accuracy | Suy giảm so với Clean |
|---|---:|---:|---:|
| Clean | **1.576** | **81,49%** | — |
| Gaussian Blur | **1.261** | **65,20%** | **−16,29 điểm %** |
| Gaussian Noise | **1.403** | **72,54%** | **−8,95 điểm %** |
| Occlusion | **1.478** | **76,42%** | **−5,07 điểm %** |

Cấu hình E8 đã khóa: Blur kernel `7`, sigma `1.5`; Gaussian Noise mean `0`, std `0.05`, clamp `[0,1]`; Occlusion hình vuông có cạnh bằng `30%` cạnh ảnh, màu lấp RGB `[0.5,0.5,0.5]`, vị trí deterministic theo ảnh.

**Nhận xét:** Gaussian Blur là điều kiện làm giảm Accuracy mạnh nhất trong ba biến đổi đã thử. Đây là hạn chế thực nghiệm cần nêu trung thực; không chỉnh tham số hậu nghiệm trên Official Test để làm đẹp số liệu.

## 8. So sánh Validation và Official Test

| Chỉ số | Validation (300 ảnh) | Official (1.934 ảnh) |
|---|---:|---:|
| Top-1 Accuracy E4 | **81,67%** | **81,49%** |
| Top-5 Accuracy E4 | **98,67%** | **99,53%** |
| Macro-F1 E4 | **81,51%** | **81,11%** |
| E7 Coverage | **87,00%** | **89,76%** |
| E7 Selective Accuracy | **84,29%** | **85,14%** |

Hai tập dùng cho mục đích khác nhau. **Validation** được dùng trong quy trình phát triển/khóa lựa chọn; **Official Test** chỉ được dùng cho lượt đánh giá cuối sau cấp quyền. Các kết quả gần nhau là tín hiệu tích cực, nhưng không chứng minh tính khái quát cho mọi ảnh ngoài đời thực.

## 9. Đánh giá theo mục tiêu đồ án và hạn chế

- **Lượt Official Final Evaluation:** `PASS`; output và claim được backend báo cáo hoàn tất.
- **Mục tiêu Top-1 Accuracy ≥ 85% trong đề cương:** **CHƯA ĐẠT**. Official Top-1 là **81,49%**, thấp hơn mục tiêu **3,51 điểm phần trăm**.
- **Calibration:** ECE giảm sau Temperature Scaling, không thay đổi dự đoán Top-1.
- **Selective prediction:** nâng Accuracy trên nhóm dự đoán được chấp nhận nhưng phải từ chối một phần dữ liệu.
- **Robustness:** ảnh mờ Gaussian Blur gây giảm hiệu suất rõ rệt.
- **Giới hạn suy luận:** Đây là đánh giá trên bộ Official Test được khóa; không mặc nhiên áp dụng cho mọi môi trường chụp, thiết bị hay phân bố dữ liệu mới.

**Hướng phát triển sau khi kết thúc đồ án (không làm thay đổi kết quả Task 05 đã khóa):** thu thập thêm ảnh thực tế, phân tích lỗi theo giống chó, cải thiện độ bền vững với ảnh mờ, tối ưu trải nghiệm demo và kiểm thử người dùng.

## 10. Danh mục artifact phải bảo toàn

Thư mục run:

`outputs/final_evaluation_b2/task05-final-20261026-ea8c474981b14065af219fb43ee8e65e/`

Các file được báo cáo hiện diện:

- **Kết quả:** `e4_e6_metrics.json`, `selective_metrics.json`, `e8_metrics.json`.
- **Biểu đồ:** `reliability_diagram.png`, `risk_coverage_curve.png`.
- **Dữ liệu dự đoán/đường cong:** `clean_predictions.npz`, `risk_coverage.csv`.
- **Provenance và hoàn tất:** `manifest.json`, `controls.json`, `metadata.json`, `protocol.json`, `.complete`, `.inference_run`, `claim.json`.
- **Bản snapshot phục vụ bundle:** `class_mapping.json`, `dataset.json`, `split.csv`.

**Không chạy lại run đã consumed, không xóa claim receipt, không ghi đè output.** Những artifacts này hiện là **untracked**; **không đưa cả `outputs/` lên GitHub** khi chỉ có mục tiêu push tài liệu và các commit source/control. Các tệp có thể chứa danh sách mẫu và metadata của Official Test; việc chia sẻ bundle cần được xem xét riêng theo quy định đồ án.

## 11. Bàn giao và phân công giai đoạn hoàn thiện đồ án môn học

| Vai trò / người được đề xuất | Việc cần làm | Sản phẩm bàn giao |
|---|---|---|
| **Trưởng nhóm** | Xác nhận nghiệm thu Task 05, đưa tài liệu này vào repo, push nhánh, giữ nguyên evidence và duyệt bản nộp cuối | Mốc Git được push; xác nhận nghiệm thu nội bộ; checklist trước nộp |
| **Phạm Lê Nhật Thuận** | Hoàn thiện và kiểm thử Gradio demo; viết hướng dẫn sử dụng, nội dung triển khai Chương 5 | Demo chạy được với ảnh minh họa **không thuộc Official Test**; README; ảnh màn hình; mục Chương 5 |
| **Lê Hồ Đức Anh** | Gom kết quả E0–E5 và các seed đã thực hiện, bảng so sánh hiệu năng/chi phí | Bảng và biểu đồ có nguồn artifact rõ; nội dung so sánh ở Chương 4 |
| **Nguyễn Anh Kiệt** | Biên tập Chương 4, confusion matrix, error analysis/Grad-CAM **nếu đã có bằng chứng**, ghép báo cáo và slide | Bản báo cáo/slide nhất quán số liệu; bảng/hình có chú thích và nguồn |

*Bảng phân công là đề xuất chờ nhóm xác nhận, không mặc nhiên coi mọi người đã nhận việc.* Không được bịa số liệu thí nghiệm chưa thực hiện hoặc sử dụng ảnh Official Test làm ảnh demo công khai nếu chưa được phép.

### Checklist đóng đồ án

- [ ] Trưởng nhóm kiểm tra báo cáo này với các JSON metrics trong output bundle.
- [ ] Trưởng nhóm xác nhận nghiệm thu nội bộ Task 05 (người xác nhận, ngày).
- [ ] Đưa Markdown vào repository, chỉ stage/commit file tài liệu mong muốn, push nhánh `task05-final-evaluation-preflight`.
- [ ] Chương 4 dùng số liệu Official, không nhầm Selective Accuracy thành Top-1 toàn tập.
- [ ] Chương 5 có demo và hướng dẫn sử dụng; ảnh demo không tái sử dụng Official Test tùy tiện.
- [ ] Nhóm hoàn thiện phần so sánh E0–E5, hình minh họa, Chương 1–5 và tài liệu tham khảo theo yêu cầu môn học.
- [ ] Chuẩn bị slide, tổng duyệt demo và trả lời được câu hỏi **vì sao Top-1 chưa đạt 85%**.
- [ ] Không đụng vào claim ledger và artifacts Official đã hoàn thành.

### Xác nhận nghiệm thu nội bộ

- **Kết quả kỹ thuật:** Hoàn thành theo báo cáo `OFFICIAL_EXECUTION_COMPLETE` / `PASS`.
- **Người xác nhận:** _Trưởng nhóm điền khi duyệt_.
- **Ngày xác nhận:** _Điền khi duyệt_.
- **Quyết định nghiệm thu nội bộ:** _Chờ trưởng nhóm xác nhận_.

---

**Tóm tắt bàn giao:** Task 05 đã hoàn thành thực thi một lượt Official theo authorization; việc còn lại của đồ án là **kiểm tra nghiệm thu nội bộ, ghép báo cáo môn học, hoàn thiện demo và chuẩn bị bảo vệ** — không phải huấn luyện hay chạy lại Official Test.
