# Nghiệm thu hồi quy cuối cùng DogBreedID

Rà soát ngày 24/11/2026 trên nhánh `fix/final-regression-tests`, bắt đầu từ `main` tại `2a0df0581482a5487dc2a3748a99c480f00335b0`. Nhánh sửa lỗi nằm trong worktree riêng.

Commit được nghiệm thu là `bb1aa4501542f834d476c236a79ee5c498d4cef2`. Full Tracked Suite chạy trên checkout biệt lập có cây Git trùng với commit này. Các lượt kiểm thử đại diện, ba tệp hồi quy và Full Local Suite dùng cùng nội dung mã và tệp test của commit được nghiệm thu, với phạm vi riêng được ghi bên dưới.

## Kết quả gốc

| Lượt chạy | Passed | Failed | Errors | Skipped | Warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full suite theo kết quả bàn giao | 1206 | 91 | 23 | 7 | 25 |
| Last-failed theo `DogBreedID_failed_tests.log` | 0 | 91 | 23 | 0 | 0 |

Log last-failed ghi nhận 47 failed và 23 errors ở `test_final_evaluation_execution.py`, 43 failed ở `test_final_evaluation_preflight.py` và 1 failed ở `test_official_evaluation_integration.py`.

## Nguyên nhân gốc

- Phase A/B1 dùng `_protocol` để kiểm tra trạng thái `BẢN NHÁP — CHƯA KHÓA CẤU HÌNH`. Fixture sao chép Configuration Lock thật đã được phê duyệt, nên pipeline dừng trước các phép kiểm. Log xác nhận 70 traceback execution dừng trong `_protocol`. Các test preflight nhận `BLOCKED` trước bước kiểm thử dự kiến.
- Test tích hợp dùng Lock thật nhưng vẫn chờ `LOCK_DRAFT`. Với Lock đã duyệt và `release=None`, gate trả về `TRUSTED_RELEASE_REQUIRED`.

## Tệp thay đổi

| Tệp | Thay đổi |
| --- | --- |
| `tests/test_final_evaluation_execution.py` | Tạo Lock bản nháp trong checkout tạm trước khi commit fixture, kể cả fixture dùng để tạo bundle provenance. |
| `tests/test_final_evaluation_preflight.py` | Tạo Lock bản nháp trước khi lưu bytes tham chiếu Git. Các reference còn lại vẫn dùng bytes gốc. |
| `tests/test_official_evaluation_integration.py` | Kiểm thử riêng Lock bản nháp và Lock đã duyệt thiếu trusted release. Cả hai xác nhận chỉ đọc Lock và dừng trước reader, checkpoint, dữ liệu Official, claim và output. |
| `requirements.txt` | Khai báo `timm==1.0.31` theo môi trường nghiệm thu E10/E11. |
| `docs/research/results/final_regression_acceptance.md` | Báo cáo nghiệm thu. |

## Kiểm thử đại diện

`26 passed` trong nhóm luồng chạy hợp lệ, provenance, snapshot và hai nhánh default deny. Bảy ca gần điểm crash trong lượt chạy đầu cũng pass khi chạy riêng.

## Ba file hồi quy

```powershell
python -m pytest tests/test_final_evaluation_execution.py tests/test_final_evaluation_preflight.py tests/test_official_evaluation_integration.py -q --tb=short -ra
```

**302 passed, 0 failed, 0 errors, 0 skipped trong 8 phút 54 giây.**

## Full Tracked Suite và Full Local Suite

```powershell
python -m pytest tests/ -q --tb=short -ra
```

| Lượt chạy | Passed | Failed | Errors | Skipped | Warnings | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Tracked Suite | 1292 | 0 | 0 | 9 | 25 | 10 phút 06 giây |
| Full Local Suite | 1319 | 0 | 0 | 9 | 25 | 12 phút 10 giây |

Full Tracked Suite chỉ gồm 1.301 ca test thuộc Git. Full Local Suite chạy thêm 27 ca E1 nội bộ từ ba tệp local chưa được Git theo dõi, tổng cộng 1.328 ca. Ba tệp E1 nội bộ nằm ngoài PR. Hai lượt có cùng lý do skip và warning.

### Skipped

- 7 ca trong `test_selective_prediction.py` yêu cầu `numpy.longdouble` có độ chính xác cao hơn `float64`. Môi trường Windows này không có độ chính xác bổ sung.
- 1 ca kiểm tra OOD-200 và 1 ca đối chiếu hash checkpoint E4 dùng điều kiện skip sẵn có. Bản sao biệt lập không chứa OOD-200 hay checkpoint E4 thật.

### Warnings

- 18 cảnh báo từ Gradio về tùy chọn pandas `future.no_silent_downcasting` đã được đánh dấu lỗi thời.
- 6 cảnh báo từ Gradio về đối số `copy` của pandas đã được đánh dấu lỗi thời.
- 1 cảnh báo NumPy khi chuyển chuỗi `1e400` thành `longdouble` trong ca kiểm thử tràn số.

## Môi trường kiểm thử

Lượt ba tệp hồi quy và hai lượt full suite dùng Python 3.12.4, torch 2.6.0+cu124, torchvision 0.21.0+cu124 và timm 1.0.31 từ `.venv` gốc. Các lượt thử ban đầu bị gián đoạn khi các tiến trình Git FSMonitor từ repo fixture tạm làm hết bộ nhớ commit của Windows. Lượt nghiệm thu tắt `core.fsmonitor` theo tiến trình, đặt các thư viện toán chạy một luồng và dùng backend Matplotlib Agg. Cấu hình Git hệ thống giữ nguyên.

## Protected artifacts

Configuration Lock thật, authorization, kết quả Official đã công bố, cấu hình nghiên cứu, split và các artifact E4/E6/E7/E8 nằm ngoài diff. Các luồng thực thi trong test dùng ảnh, checkpoint, claim và output từ fixture tạm. Checkout `main` vẫn ở commit gốc với các tệp tracked nguyên trạng. Danh sách untracked và ignored giữ nguyên. Đối chiếu metadata theo đường dẫn, kích thước và thời gian sửa của 20.919 tệp local trong các thư mục dữ liệu, checkpoint và output cho cùng SHA-256 trước và sau full suite.

## Trạng thái Git

`fix/final-regression-tests` theo dõi `origin/fix/final-regression-tests` và là nhánh nguồn của PR #31 vào `main`. PR đang mở. Ở lần rà soát commit nghiệm thu, GitHub báo `MERGEABLE` và `CLEAN`. Bản cập nhật này chỉ sửa báo cáo Markdown và đã qua `git diff --check`. Checkout chính vẫn ở `main`. [PR #31](https://github.com/DGiang1126/dogbreedid-cv-project/pull/31)

## Kết luận nghiệm thu

Full Tracked Suite đạt tiêu chí nghiệm thu với 0 failed và 0 errors trên commit đã ghi ở trên. Các kết quả được giữ lại cho bản cập nhật Markdown. Không chạy lại full suite hoặc thực thi Official Test thật. Đề nghị trưởng nhóm duyệt diff và log kiểm thử trước khi merge PR #31.
