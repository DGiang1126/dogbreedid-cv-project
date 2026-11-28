# Task 08.04 — Group-aware Discriminative Knowledge Base Design

**Đề xuất: READY_FOR_DESIGN_APPROVAL** · 28/11/2026 (UTC+7) · Integration revision **`448e890ed5e85b8d543304f2cbcaaadbfe619fa2`**.

## 1. Design objectives

- Cố định 30 canonical IDs, exact names và 10 nhóm G1–G10 của dự án. Kiểm đủ ba cặp trong mỗi nhóm và ghi phần chồng lấn mà ảnh đơn lẻ khó giải.
- Ưu tiên tín hiệu có nguồn, có ích với ít nhất một lớp cạnh tranh và có điều kiện quan sát cụ thể. Khi tiêu chuẩn chỉ hỗ trợ thuộc tính chung hoặc kích thước cần đo, giữ đúng mức bằng chứng.
- Đề xuất 30 prompts tiếng Anh cho cùng một Morphology Bank của E12-B/E13-B. Chọn trước bằng tiêu chuẩn giống và lập luận quan sát. Phạm vi lần này gồm thiết kế và đối chiếu nguồn công khai. Ảnh, nhãn, annotations, dự đoán và kết quả Validation/formal nằm ngoài đầu vào thiết kế.

### Cách đọc

**V** (*Visual*) — có thể nhìn thấy khi vùng liên quan đủ rõ. **C** (*Conditional*) — cần đúng góc, tư thế, tuổi, grooming hoặc điều kiện khác. **S** (*Scale-dependent*) — cần tỷ lệ hoặc phép đo kích thước thực tế. Đây là nhãn thiết kế ở cấp claim. Assistant dùng visibility enum của Contract 04.

**SUPPORTED** — nguồn trực tiếp hậu thuẫn câu ở phạm vi đã ghi. **SUPPORTED_WITH_SCOPE** — có căn cứ khi giữ ngoại lệ, biến thể, hệ chuẩn, cách đo hoặc điều kiện quan sát nêu trong bảng. **NEEDS_REVISION** — claim hiện hữu cần sửa câu, ID hoặc cách dẫn. **UNVERIFIABLE** — nguồn đã đối chiếu chưa đủ cho cách diễn giải đó. Các trạng thái ngữ nghĩa phục vụ review thiết kế. Phê duyệt nguồn runtime là quyết định riêng.

## 2. Hạn chế của Draft 2

- `breed-kb-draft-2` có 58 thuộc tính. `morphology-draft-2` có 30 prompts và `name-1` có 30 prompts tên. Đối chiếu bằng canonical ID và exact name tại revision nêu trên. Khi bỏ tên canonical, ba Poodle có cùng một hậu tố, ba Schnauzer cũng có cùng một hậu tố. Tiêu chuẩn hậu thuẫn phần mô tả chung. Draft 2 thiếu dữ liệu kích thước theo từng chuẩn và điều kiện để dùng kích thước trong so sánh.
- Một số claim ghép nhiều khái niệm trong một ID. ID 0 dùng `coat_texture` cho độ đầy lông theo gloss, ba Schnauzer gộp râu với texture, Golden Retriever gộp độ gợn với chiều dài cục bộ của lông tua. Một số dấu hiệu có thể chồng lấn, chẳng hạn tai bán dựng ở nhóm Collie, màu lông ở nhóm Spaniel và màu kem của Golden với Labrador vàng nhạt.
- Hồ sơ nguồn của Thuận có review theo claim của draft 1 và `follow_up` của draft 2. Lượt này đối chiếu lại toàn bộ 58 claim của draft 2 với tài liệu công khai thực tế. Mục 6 ghi locator và disposition cho từng claim.

Đối chiếu các tệp tại đúng integration revision. [Canonical mapping](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/data/metadata/class_mapping.json) · [KB Draft 2](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/data/knowledge/vlm/breed_kb.json) · [Morphology Draft 2](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/data/knowledge/vlm/morphology_prompt_bank.json) · [Name Bank](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/data/knowledge/vlm/name_prompt_bank.json) · [Hồ sơ nguồn trước đó](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/outputs/research/vlm/task08_04/kb_source_review.json)

## 3. Vocabulary proposal

| Phạm vi | ID đề xuất | Diễn giải |
| --- | --- | --- |
| Visual Observation hiện tại | `ear_position`, `ear_shape`, `coat_color`, `coat_length`, `coat_texture`, `muzzle_length`, `tail_shape`, `body_build` | Giữ đúng 8 ID, enum, quy tắc visibility/null và màu dạng array theo palette hiện tại. |
| Chỉ KB | `breed_size` | Mức hoặc khoảng cao vai của chó trưởng thành theo hệ ghi rõ. Độ cao đo được khác tỷ lệ thân trong `body_build`. Cần phép đo hoặc ảnh có tỷ lệ. Dùng cho G3, G4 và so sánh G8. |
| Chỉ KB | `head_shape` | Hình thái sọ và đường viền đầu nhìn trước/nghiêng, gồm stop và tỷ lệ hình đầu khi có nguồn. `muzzle_length` chỉ chiều dài mõm. Dùng cho G7, G8 và G10. |
| Chỉ KB | `facial_hair` | Râu, ria, lông mày và hướng lông có vai trò tạo viền mặt. Dùng cho G4 và G10. Kết cấu sợi lông dùng `coat_texture`. |
| Chỉ KB, bổ sung cho ID 0 | `coat_fullness` | Mô tả lớp lông nhìn dày/đầy ở mức định tính. Gloss WordNet của ID 0 không xác định kết cấu sợi, độ dài, số lớp hay mật độ định lượng. Một claim riêng giữ đúng nguồn và điều kiện mỗi breed có ít nhất một thuộc tính của loader. |
| Để quyết định sau nếu cần runtime | `tail_length` hoặc một ID cho biến thể đuôi bẩm sinh | EntleBucher có đuôi dài tự nhiên hoặc bobtail bẩm sinh theo chuẩn. Giữ sự kiện ở bảng phân tích cặp. Hình cong của đuôi thuộc `tail_shape`. |

**Mở rộng tối thiểu.** Dùng allowlist KB có version mà caller sở hữu, gồm 8 ID hiện có cộng bốn ID KB-only đề xuất. `MorphAttribute.attribute_id` và `value` là chuỗi. `load_knowledge_base` chuyển `allowed_attribute_ids` xuống validator, kiểm mỗi giống có thuộc tính và yêu cầu source review ref do caller tin cậy cung cấp. Thêm ID ở allowlist KB và test của caller, đồng thời ghi rõ hai vocabulary trong Contract 01. Ghi điều kiện quan trọng ngay trong `value` vì renderer chỉ đưa tên breed, attribute ID, value và source IDs vào context. Các trường `variability_note` và `uncertainty_note` có thể bổ sung thông tin cho người review.

**Caller và tests.** Tại revision này các đường gọi loader được tìm thấy trực tiếp ở `tests/vlm_foundation/test_knowledge_base.py`. Foundation còn có fixtures và tests gọi `validate_knowledge_snapshot` bằng allowlist nhỏ, gồm ID synthetic. Khi Đức Anh và Kiệt nối consumer thật, kiểm đường nạp KB/config, prompt-bank binding, vocabulary đã pin và giữ ranh giới giữa KB với structured observations, scorer và rubric. Bổ sung fixtures cho ID mới được phép ở KB và bị từ chối trong `AttributeObservation`, snapshot dùng ID mới bị từ chối dưới allowlist cũ, nguồn thiếu review ref, wrong hash/version, attribution mismatch và context budget. Giữ các case alias collision, unknown breed, policy và F2 ở biên caller. [Contracts 01, 02, 04](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/docs/research/vlm/interface_contract.md) · [Validators](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/src/vlm/contracts.py#L627) · [Loader](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/src/vlm/knowledge_base.py#L7) · [Renderer và retrieval](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/src/vlm/retrieval.py#L21)

## 4. Phân tích theo nhóm

### G1 — Arctic Dogs

Lông trông dày và dáng chó kéo xe vùng lạnh là nét chung ở mức mô tả. Dáng hai giống có tiêu chuẩn FCI có thể so bằng thân và đuôi. ID 0 được neo bằng synset, nên chỉ dùng đặc điểm lông dày được gloss hậu thuẫn.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 0–1 | ID 1 có ngực sâu, thân chắc nặng xương và đuôi xù dạng chùm lông thường mang trên lưng khi nghỉ làm việc. | Lông dày chồng lấn. WordNet không cho ngưỡng dáng hay đuôi của ID 0. Ảnh riêng lẻ thiếu cơ sở loại ID 0. |
| 0–2 | ID 2 có dáng vừa phải, đường nét gọn, tai dựng cao và đuôi dạng lưỡi liềm khi chú ý. | Dấu hiệu phù hợp ID 2, vẫn chồng lấn nhóm husky. Cần thân ngang, tai và đuôi nhìn thấy. ‘Husky’ trong query trần vẫn mơ hồ [0, 2]. |
| 1–2 | Malamute thiên về xương nặng, ngực chắc và đuôi lông xòe. Siberian Husky thiên về khung vừa, ngực sâu nhưng không quá rộng, đuôi dạng chổi cong lưỡi liềm khi chú ý; có thể rủ lúc nghỉ. | C – xem ngang toàn thân và đuôi, ghi trạng thái chú ý. Cả hai đuôi có thể mang qua lưng; không dùng mẹo ‘con nào mang cao là Malamute’. Tai đều có thể dựng. |

**Bằng chứng nhóm** · princeton-wordnet; imagenet-synsets; fci-243 tr. 2–5; fci-270 tr. 2–5, 7; fci-211 tr. 1–3 (URL và version ở mục 6).

### G2 — Swiss Dogs

Kiểu tam thể là nét chung. Nền đen cùng mảng nâu đỏ và trắng chồng lấn cả ba. Nền nâu Havana là một lựa chọn chuẩn của Appenzeller, không phải màu duy nhất của lớp đó.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 3–4 | ID 3 thân nặng xương, đuôi nặng rủ lúc nghỉ và chỉ cong nhẹ khi nâng. ID 4 có đuôi cuộn chặt trên mông khi chuyển động; nền Havana là tín hiệu phụ. | C – cần thấy đuôi trong tư thế phù hợp. Ở trạng thái nghỉ ID 4 có thể thả đuôi. Cả hai tỷ lệ cao:dài khoảng 9:10 theo tiêu chuẩn, không dùng tỷ lệ này để tách cặp. |
| 3–5 | ID 3 lực lưỡng và xương nặng, ID 5 thân gọn hơi dài với tỷ lệ cao:dài 8:10; ID 5 có thể có đuôi ngắn bẩm sinh. | C – thân đứng ngang, đủ cẳng và đường lưng. Đuôi dài tự nhiên ở ID 5 cũng hợp lệ. Nền tam thể đen không giải được cặp. |
| 4–5 | Đuôi cuộn chặt khi đi ở ID 4 đối chiếu đuôi dài tự nhiên hoặc bobtail bẩm sinh ở ID 5; đuôi vòng là lỗi loại ở ID 5. | C – thấy cả đuôi và tư thế. So dáng gọn gần vuông của ID 4 với dáng dài hơn của ID 5 để bổ trợ. Ảnh crop hoặc đuôi bị che có thể bất định. |

**Bằng chứng nhóm** · fci-058 tr. 3–5; fci-046 tr. 3, 5–6; fci-047 tr. 3–6, 8 (URL và version ở mục 6).

### G3 — Poodle

Lông xoăn hoặc bện dây và tai rủ sát má dùng chung. Grooming có thể đổi hình dáng lông và đường viền. Tỷ lệ tổng thể chồng lấn; không gán kiểu tai, kiểu cắt lông hay màu khác nhau cho ba lớp.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 6–7 | Toy thấp hơn Miniature trong từng hệ tiêu chuẩn với ngưỡng của chính hệ đó. | S – người đo, mặt phẳng đứng và tuổi trưởng thành; ảnh crop không thước không đủ. |
| 6–8 | Toy và Standard cách nhau về chiều cao tham chiếu, vẫn cùng kiểu coat và tai. | S – khoảng cách camera và kích thước khung có thể đảo cảm nhận nhỏ/lớn. |
| 7–8 | Miniature thấp hơn Standard trong từng hệ. FCI có thêm Medium nằm giữa hai nhóm; AKC dùng ba cỡ và một hệ ngưỡng riêng. | S – ngay cả khi có số đo, chỉ đối chiếu với hệ được nêu, không suy nhãn dữ liệu từ chuẩn triển lãm. |

**Bằng chứng nhóm** · fci-172 tr. 4, 6, 8; akc-poodle tr. 1–2; princeton-wordnet; stanford-dogs-paper §2 (URL và version ở mục 6).

**Đối chiếu cỡ Poodle.** FCI #172 đo cao vai. Toy >24 đến 28 cm, dung sai −1 cm, lý tưởng 25 cm. Miniature >28 đến 35 cm. Medium >35 đến 45 cm. Standard >45 đến 60 cm, dung sai +2 cm. AKC đo ở điểm cao nhất của vai và dùng ba cỡ. Toy ≤10 in, Miniature >10 đến 15 in, Standard >15 in. Chẳng hạn 28 cm gặp ngưỡng Toy của FCI và nằm trong Miniature của AKC. Không gộp ngưỡng. Standard Poodle, Toy Poodle và Miniature Poodle ở đây vẫn là các nhãn canonical. [FCI #172, tr. 8](https://www.fci.be/nomenclature/Standards/172g09-en.pdf#page=8) · [AKC Poodle, tr. 1](https://images.akc.org/pdf/breeds/standards/Poodle.pdf#page=1)

### G4 — Schnauzer

Ba cỡ chia sẻ thân gần vuông, lông thô dạng sợi, râu và lông mày rậm, tai tự nhiên gập xuôi. Grooming đổi hình râu, bờm chân và đường viền. Tai đã can thiệp không phản ánh dáng tai tự nhiên.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 9–10 | FCI đo vai: Miniature 30–35 cm, Standard 45–50 cm. Tỷ lệ gần vuông, râu và coat chồng lấn. | S – ảnh có thước cùng mặt phẳng, chó trưởng thành đứng. Màu trắng thuần có trong FCI 183 nhưng không dùng làm bộ phân loại mặc định. |
| 9–11 | Miniature 30–35 cm, Giant 60–70 cm theo FCI. Standard về khung gần vuông và coat/râu giống nhau. | S – góc rộng hoặc crop có thể làm mất chênh lệch. Black/silver có trong cả FCI 183 và 181. |
| 10–11 | Standard 45–50 cm, Giant 60–70 cm theo FCI. Giant mô tả khung mạnh, dáng tăng cỡ của Schnauzer; chất xương trên ảnh chỉ gợi ý. | S – ưu tiên số đo. Cả hai có đen và pepper-and-salt; grooming, tuổi và phối cảnh làm yếu so sánh dáng. |

**Bằng chứng nhóm** · fci-183 tr. 2–3, 5–6; fci-182 tr. 2–3, 5–6; fci-181 tr. 2, 4–6 (URL và version ở mục 6).

### G5 — Setter

Tai rủ thấp và lông có vùng đổ tua là nét chung. Ưu tiên màu, nền và cách phân bố mảng hoặc chấm. Tai thấp một mình có ít giá trị phân biệt.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 12–13 | English Setter có belton lấm tấm trên nền trắng ở các phối màu chuẩn. Irish Setter có nền hạt dẻ đậm, cho phép dấu trắng nhỏ ở vị trí nêu trong chuẩn. | V – nhìn đủ thân và ánh sáng trung tính. Mảng lớn có thể che mẫu chấm ở English Setter. |
| 12–14 | Belton hoặc phối ba màu của English Setter đối chiếu nền đen và mảng tan ở lông mày, mõm, ngực của Gordon Setter. | V – cần toàn thân và mặt. Crop đen/tan có thể bỏ phần nền trắng của English Setter. |
| 13–14 | Nền hạt dẻ không đen của Irish Setter đối chiếu nền đen với điểm nâu hạt dẻ phân bố của Gordon Setter. | V – cần phân biệt màu nền với vùng tan. Ánh đèn hoặc bóng tối làm lệch cảm nhận đỏ/đen. |

**Bằng chứng nhóm** · fci-002 tr. 3–4; fci-120 tr. 3–4; fci-006 tr. 3–4 (URL và version ở mục 6).

### G6 — Spaniel

Tai rủ có lông tua, lông thân thường phẳng hoặc thẳng theo các chuẩn được dùng. Tỷ lệ dáng và hình tai có giá trị khi tư thế cho phép. Cocker Spaniel ở đây bám synset English Cocker Spaniel.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 15–16 | English Springer đen-trắng hoặc nâu gan-trắng, có thể có tan, tai thùy dài rộng và dáng cao chân. Welsh Springer đỏ đậm-trắng và tai nhỏ thon. | V/C – dùng màu khi ánh sáng tốt; dùng tai và dáng khi ảnh đủ thân. Nâu gan bị ám đỏ dễ nhầm. |
| 15–17 | Nhiều tổ hợp màu chồng lấn. English Springer thiên dáng cao chân; Cocker Spaniel thân gọn cân đối, vành tai dài khi kéo thẳng tới chóp mũi. | C – cần góc ngang, toàn thân và tai. Độ dài vành tai theo phép kiểm tiêu chuẩn không đo được từ viền lông tai. |
| 16–17 | Welsh Springer tai nhỏ thon và đỏ-trắng. Cocker Spaniel có thể cam-trắng hoặc lemon-trắng, kèm tai thùy dài. | V/C – màu đỏ/cam dễ lẫn theo white balance. Dùng hình vành tai và tỷ lệ hỗ trợ; ảnh crop khó kết luận. |

**Bằng chứng nhóm** · fci-125 tr. 2–4; fci-126 tr. 2–4; rkc-welsh-springer mục Ears, Coat, Colour; fci-005 tr. 2–4; princeton-wordnet (URL và version ở mục 6).

### G7 — Terrier

Lông cứng thô và dáng làm việc kiểu terrier chồng lấn. Norfolk và Norwich đặc biệt gần nhau ở đầu, mõm và kích cỡ chuẩn. Ear position có giá trị cao nhất khi tai nguyên vẹn và chó chú ý.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 18–19 | Norfolk tai chữ V rủ về trước; Norwich tai nhọn dựng khi chú ý, có thể ngả sau khi nghỉ. | C – quan sát hai tai, tư thế chú ý. Vệt màu, mõm nêm và lông thô thường không giải được cặp. |
| 18–20 | Cả hai tai chữ V rủ. Border Terrier có dạng đầu như rái cá, mõm ngắn khỏe; Norfolk có đầu rộng nhẹ vồng và mõm hình nêm. | C – xem mặt trước/nghiêng và đường viền mõm. Khác biệt tinh tế; ảnh tóc/grooming che đầu có thể không đủ. |
| 19–20 | Tai dựng nhọn lúc chú ý của Norwich đối chiếu tai chữ V nhỏ rủ phía trước của Border. Dáng đầu rái cá của Border bổ trợ. | C – tai ngả sau ở Norwich hoặc mắt nhìn góc thấp có thể gây nhầm. Lông thô chung nhóm. |

**Bằng chứng nhóm** · fci-272 tr. 2–4; fci-072 tr. 2–4; fci-010 tr. 2–3 (URL và version ở mục 6).

### G8 — Collie and Sheepdogs

Đầu thon dạng nêm và tai bán dựng chóp gập có thể chồng lấn, đặc biệt giữa Collie và Shetland Sheepdog. Bộ lông có biến thể. Không thu hẹp Collie thành Rough-only, cũng không suy tập ảnh chứa cả hai biến thể từ hai chuẩn.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 21–22 | Dáng đầu nêm, tai bán dựng và bờm ở Rough Collie đều chồng lấn Shetland Sheepdog. Chiều cao vai chuẩn Sheltie nhỏ hơn rõ khi đo. | S – cần tỷ lệ. Smooth Collie lông ngắn có thể gợi ý ID 21 so với Sheltie có lông dài theo chuẩn, nhưng Rough và Sheltie crop thường không đủ. |
| 21–23 | Collie hai biến thể có nêm tương đối dài, stop nhẹ và chóp tai gập khi chú ý. Border Collie có stop rõ, tai dựng hoặc bán dựng. | C – ảnh ngang rõ sống mũi và trán. Đuôi rủ chót cong và bộ lông dài/ngắn không tách sạch cặp. |
| 22–23 | Sheltie thiên đầu nêm dài, stop nhẹ và bờm rõ; Border Collie có stop rõ, tai linh hoạt và cả kiểu lông trơn hoặc dài vừa. | C/S – bờm có thể xuất hiện ở Border Collie lông dài vừa. Chiều cao vai hỗ trợ nếu đo thực tế; phối cảnh và grooming làm tín hiệu yếu. |

**Bằng chứng nhóm** · fci-156 tr. 3–5; fci-296 tr. 2–5; fci-088 tr. 2–5; fci-297 tr. 2–4; princeton-wordnet (URL và version ở mục 6).

### G9 — Retriever

Lông có lớp dưới và nhiều trạng thái đuôi chồng lấn. Ưu tiên chiều dài lông, feathering, độ gợn theo vùng và biên dạng phần gốc đuôi. Màu vàng/kem hoặc nâu có thể chồng lấn từng cặp.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 24–25 | Golden có lông tua rõ, lông phẳng/gợn và dải vàng/kem. Labrador lông ngắn đều, đuôi dày tròn ở gốc, thon dần, không tua. | V/C – thấy thân và đuôi trọn, xem độ dài lông trước màu. Labrador vàng nhạt chồng Golden kem. |
| 24–26 | Golden có tua nổi rõ và màu vàng/kem. Chesapeake lông ngắn và gợn giới hạn vùng cổ, vai, lưng, thắt lưng; dải deadgrass có thể gần vàng nhạt. | C – xem lông thân, tránh kết luận từ màu hoặc vài lọn ở đuôi. Chesapeake có thể có tua vừa ở đuôi/hông sau. |
| 25–26 | Cùng lông ngắn và có sắc nâu. Labrador lông không gợn và đuôi otter; Chesapeake có thể gợn lưng và đuôi chỉ vừa nặng ở gốc. | C – khô, ánh sáng xiên nhẹ, thấy thân và gốc đuôi. Ảnh ướt, lông nén hoặc góc crop có thể không đủ. |

**Bằng chứng nhóm** · fci-111 tr. 3; fci-122 tr. 3–4; fci-263 tr. 4–5 (URL và version ở mục 6).

### G10 — Short-muzzled Companion Dogs

Mõm tương đối ngắn/rộng và ngoại hình chó bầu bạn lông dài là nét chồng lấn. Mục tiêu phân biệt là kiểu lông trên mặt/mõm, bờm và dáng, không dùng độ bẹt cực đoan hoặc dấu hô hấp.

| Cặp ID | Tín hiệu theo nguồn | Quan sát, chồng lấn và trường hợp thiếu căn cứ |
| --- | --- | --- |
| 27–28 | Japanese Spaniel thường có mặt lông ngắn, lông thân mượt và nền trắng với mảng đen/đỏ theo FCI. Pekinese có thân chắc, bờm cổ vừa, đầu rộng và phẳng giữa tai. | V/C – cần thấy lông mặt, cổ và toàn thân. Bờm/cổ có thể trùng khi crop. Màu chỉ bổ trợ vì Pekinese đa màu. |
| 27–29 | Japanese Spaniel mặt ít lông dài, Shih-Tzu có râu và lông mọc hướng lên trên mõm theo chuẩn. | C – mặt chính diện và profile; cắt ngắn hoặc buộc lông biến đổi tín hiệu. Lông thân dài và mõm ngắn chồng lấn. |
| 28–29 | Pekinese có bờm dạng áo choàng quanh cổ, đầu rộng phẳng giữa tai và thân chắc. Shih-Tzu thiên lông mọc hướng lên ở mõm và thân dài hơn chiều cao vai theo chuẩn. | C – cần mặt và cổ rõ, góc ngang cho tỷ lệ. Grooming có thể che hoặc mô phỏng cả hai. Không suy mức độ ngắn mõm khi mặt bị lông che. |

**Bằng chứng nhóm** · fci-206 tr. 2–4; akc-japanese-chin tr. 1–2; fci-207 tr. 2–4; fci-208 tr. 2–4 (URL và version ở mục 6).

## 5. Bảng thiết kế 30 lớp và Morphology Prompts

Mỗi dòng có khóa `(group_id, class_id, canonical_name)`. Hai cột thuộc tính liệt kê ID có claim cụ thể ở mục 6. Mục 6 nối từng ID tới claim tiếng Việt, phrase tiếng Anh, V/C/S, điều kiện, source ID/locator, URL và bản tài liệu. “Phân biệt” nghĩa là có ích với ít nhất một lớp cạnh tranh trong nhóm dưới điều kiện đã nêu. Cột “Dễ nhầm” là đánh giá dựa vào hình thái, không phải thống kê lỗi thực nghiệm.

| Group | ID | Canonical name | Thuộc tính chung | Thuộc tính phân biệt | Dễ nhầm và giới hạn | Morphology prompt chính |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | 0 | Eskimo Dog | `coat_fullness` | Chưa xác định | IDs 1, 2. Gloss ở mức synset. Lông dày chồng lấn cả hai lớp cùng nhóm. Dáng, đuôi và quan hệ với một giống đăng ký chưa được xác lập. | A photo of the dog breed Eskimo Dog, with a thick coat. |
| G1 | 1 | Malamute | `coat_texture` | `body_build`, `tail_shape` | IDs 0, 2. Tư thế, mùa lông và góc máy đổi dáng đuôi/thân. Tai dựng chồng Siberian Husky. | A photo of the dog breed Malamute, with a powerful deep-chested build and a bushy plumed tail. |
| G1 | 2 | Siberian Husky | `coat_texture`, `ear_position` | `body_build`, `tail_shape` | IDs 0, 1. Đuôi có thể rủ khi nghỉ. Dáng gọn và đuôi cong đều cần đọc có điều kiện. | A photo of the dog breed Siberian Husky, with a moderately compact build and a brush tail curving like a sickle when alert. |
| G2 | 3 | Greater Swiss Mountain Dog | `coat_color` | `body_build`, `tail_shape` | IDs 4, 5. Tam thể và tỷ lệ 9:10 chồng Appenzeller. Cần thấy khung xương, ngực hoặc đuôi. | A photo of the dog breed Greater Swiss Mountain Dog, with black, tan and white markings and a heavy-boned muscular build. |
| G2 | 4 | Appenzeller | `coat_color` | `body_build`, `tail_shape` | IDs 3, 5. Chỉ dùng đuôi cuộn chặt khi thấy tư thế. Nền đen chồng hai lớp kia. | A photo of the dog breed Appenzeller, with tan and white markings on a black or Havana-brown coat and a tightly curled tail over the rump when moving. |
| G2 | 5 | EntleBucher | `coat_color`, `coat_length` | `body_build` | IDs 3, 4. Giữ biến thể đuôi ở phân tích nhóm. Bobtail không bắt buộc. Lông ngắn cũng có thể thấy ở ID 3. | A photo of the dog breed EntleBucher, with black, tan and white markings and a compact, slightly elongated build. |
| G3 | 6 | Toy Poodle | `coat_texture`, `ear_position` | `breed_size` | IDs 7, 8. Ảnh đơn không tỷ lệ khó tách cỡ. Grooming và khác ngưỡng FCI/AKC. | A photo of the dog breed Toy Poodle, with a curly or corded coat and ears hanging close to the cheeks. |
| G3 | 7 | Miniature Poodle | `coat_texture`, `ear_position` | `breed_size` | IDs 6, 8. Ảnh đơn không tỷ lệ khó tách cỡ. FCI Medium là một cỡ riêng ngoài 30 lớp. | A photo of the dog breed Miniature Poodle, with a curly or corded coat and ears hanging close to the cheeks. |
| G3 | 8 | Standard Poodle | `coat_texture`, `ear_position` | `breed_size` | IDs 6, 7. Ảnh đơn không tỷ lệ khó tách cỡ. FCI và AKC đặt ranh giới Standard khác nhau. | A photo of the dog breed Standard Poodle, with a curly or corded coat and ears hanging close to the cheeks. |
| G4 | 9 | Miniature Schnauzer | `coat_texture`, `facial_hair`, `ear_position`, `body_build` | `breed_size` | IDs 10, 11. Cần đo chiều cao vai. Màu trắng là lựa chọn FCI theo scope, không phải dấu hiệu bắt buộc. | A photo of the dog breed Miniature Schnauzer, with a wiry coat and a pronounced beard. |
| G4 | 10 | Standard Schnauzer | `coat_texture`, `facial_hair`, `ear_position`, `body_build` | `breed_size` | IDs 9, 11. Cần đo chiều cao vai. Râu, tai và thân vuông là nét chung. | A photo of the dog breed Standard Schnauzer, with a wiry coat and a pronounced beard. |
| G4 | 11 | Giant Schnauzer | `coat_texture`, `facial_hair`, `ear_position`, `body_build` | `breed_size` | IDs 9, 10. Cần đo chiều cao vai. Dáng lực lưỡng phụ thuộc tuổi, giới, grooming và phối cảnh. | A photo of the dog breed Giant Schnauzer, with a wiry coat and a pronounced beard. |
| G5 | 12 | English Setter | `coat_texture` | `coat_color` | IDs 13, 14. Belton ưu tiên chấm rải. Có thể có mảng lớn hoặc phối tam thể, dễ nhầm khi crop. | A photo of the dog breed English Setter, with belton-style flecking. |
| G5 | 13 | Irish Setter | `ear_position` | `coat_color` | IDs 12, 14. Dấu trắng nhỏ vẫn được phép. Ánh sáng ảnh ảnh hưởng phân biệt đỏ/hạt dẻ. | A photo of the dog breed Irish Setter, with a rich chestnut coat. |
| G5 | 14 | Gordon Setter | `ear_position` | `coat_color` | IDs 12, 13. Cần thấy màu nền và điểm tan đồng thời. Ảnh chỉ một mảng tan dễ nhầm. | A photo of the dog breed Gordon Setter, with a black coat with tan points over the eyes and on the chest. |
| G6 | 15 | English Springer | `ear_shape` | `coat_color`, `body_build` | IDs 16, 17. Chồng màu với Cocker Spaniel. Cần thấy đủ chân để đánh giá cao chân tương đối. | A photo of the dog breed English Springer, with black-and-white or liver-and-white markings, sometimes with tan and an athletic, high-on-leg build. |
| G6 | 16 | Welsh Springer Spaniel | `coat_texture` | `coat_color`, `ear_shape` | IDs 15, 17. Cocker hai màu cam-trắng có thể gần sắc đỏ-trắng dưới ánh sáng khác. Tai giúp khi rõ vành. | A photo of the dog breed Welsh Springer Spaniel, with rich red and white markings and small, tapering ears. |
| G6 | 17 | Cocker Spaniel | `coat_texture` | `ear_shape`, `body_build` | IDs 15, 16. Scope English Cocker theo WordNet. Độ dài vành theo chuẩn không quy thành phép đo từ ảnh. | A photo of the dog breed Cocker Spaniel, with long, lobed ears and a compact, balanced build. |
| G7 | 18 | Norfolk Terrier | `coat_texture` | `ear_position`, `head_shape` | IDs 19, 20. Tai rủ chồng Border Terrier. Dáng đầu gần Norwich hơn so với Border. | A photo of the dog breed Norfolk Terrier, with a hard wiry coat and forward-dropping ears close to the cheeks. |
| G7 | 19 | Norwich Terrier | `coat_texture` | `ear_position` | IDs 18, 20. Tai có thể ngả sau khi không chú ý. Đầu và coat gần Norfolk. | A photo of the dog breed Norwich Terrier, with a hard wiry coat and ears held upright when alert. |
| G7 | 20 | Border Terrier | `coat_texture`, `ear_position` | `head_shape` | IDs 18, 19. Đầu kiểu rái cá cần đủ góc. Tai rủ và coat thô chồng Norfolk. | A photo of the dog breed Border Terrier, with forward-dropping ears and an otter-like head. |
| G8 | 21 | Collie | `ear_position`, `coat_length` | `head_shape` | IDs 22, 23. Dạng Rough/Smooth tham chiếu chuẩn. Phân bố biến thể trong lớp ảnh chưa xác minh. | A photo of the dog breed Collie, with semi-erect ears tipped forward when alert and a tapering blunt-wedge head with a slight stop. |
| G8 | 22 | Shetland Sheepdog | `ear_position` | `coat_length`, `head_shape`, `breed_size` | IDs 21, 23. Bờm và tai chồng Rough Collie. Cần tỷ lệ để dùng chiều cao. | A photo of the dog breed Shetland Sheepdog, with a long outer coat with a distinct neck ruff and a long wedge-shaped head with a slight stop. |
| G8 | 23 | Border Collie | `ear_position`, `tail_shape` | `head_shape` | IDs 21, 22. Tai bán dựng và coat dài vừa hoặc smooth chồng lấn. Stop rõ chỉ là tín hiệu hỗ trợ. | A photo of the dog breed Border Collie, with erect or semi-erect ears and a head with a clearly marked stop. |
| G9 | 24 | Golden Retriever | — | `coat_color`, `coat_texture`, `coat_length` | IDs 25, 26. Màu vàng/kem chồng Labrador nhạt. Feathering phụ thuộc góc và vùng bị crop. | A photo of the dog breed Golden Retriever, with a golden or cream coat and distinct feathering. |
| G9 | 25 | Labrador Retriever | `coat_length` | `coat_texture`, `tail_shape` | IDs 24, 26. Lông ngắn chồng Chesapeake. Đuôi phải thấy đủ từ gốc để dùng dáng otter. | A photo of the dog breed Labrador Retriever, with a short, dense coat and a thick-based, tapering otter tail. |
| G9 | 26 | Chesapeake Bay Retriever | `coat_length` | `coat_texture`, `coat_color` | IDs 24, 25. Gợn là xu hướng theo vùng. Ảnh ướt hạn chế phân biệt với Labrador. | A photo of the dog breed Chesapeake Bay Retriever, with a short coat and waves over the neck, shoulders and back. |
| G10 | 27 | Japanese Spaniel | `muzzle_length`, `coat_texture` | `facial_hair`, `coat_color` | IDs 28, 29. Màu và lông mặt chỉ hỗ trợ. FCI và AKC có phạm vi màu hơi khác. | A photo of the dog breed Japanese Spaniel, with short facial hair and a white coat with black or red patches. |
| G10 | 28 | Pekinese | `muzzle_length` | `head_shape`, `body_build`, `coat_length` | IDs 27, 29. Bờm và đầu cần nhìn rõ, mức bẹt mặt không dùng để xếp loại. | A photo of the dog breed Pekinese, with a broad head flat between the ears and a moderate mane around the neck. |
| G10 | 29 | Shih-Tzu | `muzzle_length`, `coat_length` | `facial_hair`, `body_build` | IDs 27, 28. Grooming có thể che lông mọc hướng lên. Mõm ngắn chồng hai lớp kia. | A photo of the dog breed Shih-Tzu, with a long outer coat and muzzle hair growing upward. |

### Bank chính và phương án size prior

Bank chính đề xuất dùng đúng 30 câu trong bảng cho cả E12-B và E13-B. Template chung là `A photo of the dog breed {canonical_name}, with {phrases}.` Name Bank tiếp tục dùng `A photo of the dog breed {canonical_name}.` Có 26 câu morphology dùng hai phrase và 4 câu dùng một phrase. Cả bank chính không thêm thuộc tính S vào hậu tố. Các từ Toy, Miniature, Standard và Giant vẫn nằm trong tên canonical, nên bank chính vẫn giữ tín hiệu kích thước có sẵn trong tên. G3/G4 giữ phần giống nhau khi nguồn không hỗ trợ tín hiệu độc lập với kích thước.

Phương án G3 có size prior là một biến thể ngoài 30 câu chính. Giữ template và hai phrase chung, rồi thêm ở cuối sau dấu phẩy `and a very small stature` cho ID 6, `and a small stature` cho ID 7, `and a larger stature` cho ID 8. Ba cụm nói về quan hệ kích thước trong nhóm, truy vết đến `breed_size` của đúng lớp. Phương án không prior chính là ba câu ở bảng. Không dùng trị số triển lãm trong text encoder. Trước Protocol Lock, chốt biến thể nào được phép và khóa một bank chính dùng chung. Size prior, nếu duyệt cho đối chứng riêng, phải có tên và bank hash riêng. Ảnh crop vẫn có scale ambiguity. [Research Protocol §5 và §8](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/docs/research/vlm/research_protocol.md)

## 6. Source evidence

Số trang là số trang PDF tính từ bìa là trang 1. Với trang HTML dùng tên mục, với WordNet dùng file và byte offset của synset. Đọc các tiêu chuẩn FCI/AKC và trang RKC trực tiếp cùng archive WordNet và bài Stanford Dogs. URL FCI có thể phát hành lại PDF cùng đường dẫn. Cột version tách ngày bản EN với ngày publication của chuẩn trên thân tài liệu. Dùng version và locator cùng nhau khi đối chiếu. Mọi câu trong thiết kế là diễn giải, trừ tên tiêu chuẩn hoặc thuật ngữ chuyên môn cần giữ.

### 6.1. Danh mục nguồn chính thức

| Source ID | Tài liệu | Version / date | URL chính thức |
| --- | --- | --- | --- |
| `fci-243` | FCI #243 Alaskan Malamute | Bản EN 10/03/2023, chuẩn có ngày 09/03/2023 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/243g05-en.pdf) |
| `fci-270` | FCI #270 Siberian Husky | Bản EN 18/11/2022, chuẩn có ngày 14/11/2022 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/270g05-en.pdf) |
| `fci-211` | FCI #211 Canadian Eskimo Dog, đối chiếu taxonomy | Bản EN 07/11/2018, chuẩn có ngày 28/02/2018 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/211g05-en.pdf) |
| `fci-058` | FCI #058 Great Swiss Mountain Dog | Bản EN 05/05/2003, chuẩn có ngày 25/03/2003 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/058g02-en.pdf) |
| `fci-046` | FCI #046 Appenzell Cattle Dog | Bản EN 05/05/2003, chuẩn có ngày 25/03/2003 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/046g02-en.pdf) |
| `fci-047` | FCI #047 Entlebuch Cattle Dog | Bản EN 28/06/2002, chuẩn có ngày 28/11/2001 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/047g02-en.pdf) |
| `fci-172` | FCI #172 Poodle | Bản EN 05/04/2024, chuẩn có ngày 20/03/2024, áp dụng từ 01/08/2024 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/172g09-en.pdf) |
| `fci-183` | FCI #183 Miniature Schnauzer | Bản EN 18/04/2007, chuẩn có ngày 06/03/2007 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/183g02-en.pdf) |
| `fci-182` | FCI #182 Schnauzer | Bản EN 18/04/2007, chuẩn có ngày 06/03/2007 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/182g02-en.pdf) |
| `fci-181` | FCI #181 Giant Schnauzer | Bản EN 10/06/2021, chuẩn có ngày 25/05/2021 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/181g02-en.pdf) |
| `fci-002` | FCI #002 English Setter | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/002g07-en.pdf) |
| `fci-120` | FCI #120 Irish Red Setter | Bản EN 02/04/2001, chuẩn có ngày 13/03/2001 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/120g07-en.pdf) |
| `fci-006` | FCI #006 Gordon Setter | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/006g07-en.pdf) |
| `fci-125` | FCI #125 English Springer Spaniel | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/125g08-en.pdf) |
| `fci-126` | FCI #126 Welsh Springer Spaniel | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/126g08-en.pdf) |
| `fci-005` | FCI #005 English Cocker Spaniel | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/005g08-en.pdf) |
| `fci-272` | FCI #272 Norfolk Terrier | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/272g03-en.pdf) |
| `fci-072` | FCI #072 Norwich Terrier | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/072g03-en.pdf) |
| `fci-010` | FCI #010 Border Terrier | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/010g03-en.pdf) |
| `fci-156` | FCI #156 Collie (Rough) | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/156g01-en.pdf) |
| `fci-296` | FCI #296 Collie (Smooth) | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/296g01-en.pdf) |
| `fci-088` | FCI #088 Shetland Sheepdog | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/088g01-en.pdf) |
| `fci-297` | FCI #297 Border Collie | Bản EN 28/10/2009, chuẩn có ngày 24/06/1987 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/297g01-en.pdf) |
| `fci-111` | FCI #111 Golden Retriever | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/111g08-en.pdf) |
| `fci-122` | FCI #122 Labrador Retriever | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/122g08-en.pdf) |
| `fci-263` | FCI #263 Chesapeake Bay Retriever | Bản EN 12/03/1999, chuẩn có ngày 14/02/1995 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/263g08-en.pdf) |
| `fci-206` | FCI #206 Japanese Chin | Bản EN 03/06/2016, chuẩn có ngày 04/04/2016 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/206g09-en.pdf) |
| `fci-207` | FCI #207 Pekingese | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/207g09-en.pdf) |
| `fci-208` | FCI #208 Shih Tzu | Bản EN 06/10/2026, chuẩn có ngày 08/09/2026 | [Mở nguồn](https://www.fci.be/nomenclature/Standards/208g09-en.pdf) |
| `akc-poodle` | AKC Official Standard of the Poodle | Duyệt 14/08/1984 | [Mở nguồn](https://images.akc.org/pdf/breeds/standards/Poodle.pdf) |
| `akc-japanese-chin` | AKC Official Standard of the Japanese Chin | Duyệt 11/10/2011, hiệu lực 30/11/2011 | [Mở nguồn](https://cdn.akc.org/JapaneseChin.pdf) |
| `rkc-welsh-springer` | Royal Kennel Club Welsh Springer Spaniel | Trang cập nhật 01/01/2025 | [Mở nguồn](https://www.royalkennelclub.com/breed-standards/gundog/spaniel-welsh-springer/) |
| `princeton-wordnet` | Princeton WordNet noun database | WordNet 3.0, archive `WNdb-3.0.tar.gz`, file `dict/data.noun` | [Mở nguồn](https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz) |
| `imagenet-synsets` | ImageNet danh mục synsets | ILSVRC 2012, mục Browse Synsets, định danh `n02109961` | [Mở nguồn](https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php) |
| `stanford-wordnet` | Stanford CS231A, Silvio Savarese, Lecture 16 | 04/03/2015, slide ‘WordNet taxonomy and definitions’ | [Mở nguồn](https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/lecture16_datasets_3D_scene_understanding.pdf) |
| `stanford-dogs-paper` | Khosla và cộng sự, Stanford Dogs | 2011, §2 ‘Stanford Dogs Dataset’, tr. 1 | [Mở nguồn](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf) |

**Phạm vi nguồn taxonomy** · WordNet `data.noun` 02109961 hỗ trợ lớp Eskimo Dog ở mức nghĩa chó kéo xe Bắc Cực có lông dày. FCI 211 tr. 1–3 mô tả Canadian Eskimo Dog riêng và chỉ dùng để đối chiếu phạm vi, không gán claims của chuẩn đó cho ID 0. WordNet 02102318 liệt kê English Cocker Spaniel cùng cocker spaniel. WordNet 02106030 mô tả Collie ở mức từ vựng, không xác định phân bố biến thể ảnh. WordNet 02085782 neo tên Japanese Spaniel. Tiêu chuẩn dùng cho ID 27 là Japanese Chin, đối chiếu với danh mục tên giống của ImageNet. [Princeton WordNet 3.0](https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz) · [ImageNet ILSVRC 2012](https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php) · [FCI 211](https://www.fci.be/Nomenclature/Standards/211g05-en.pdf)

### 6.2. Review độc lập 58 thuộc tính của Draft 2

Mỗi dòng ứng với đúng một `attributes[]` ở Draft 2 theo thứ tự canonical IDs và thứ tự attributes cũ. Kết quả là **30 SUPPORTED, 23 SUPPORTED_WITH_SCOPE và 5 NEEDS_REVISION**, không có claim hiện hữu bị bỏ qua. Những sửa đổi chủ yếu tách khái niệm hoặc thêm điều kiện được nguồn hỗ trợ.

| ID | ID thuộc tính và claim Draft 2 | Semantic review | Source ID và locator | Disposition |
| --- | --- | --- | --- | --- |
| 0 | `coat_texture` · bộ lông dày | NEEDS_REVISION | princeton-wordnet, `data.noun` 02109961; stanford-wordnet, WordNet taxonomy and definitions | Nội dung lông dày có căn cứ từ gloss. `coat_texture` đang gán độ đầy vào kết cấu. Chuyển `coat_fullness`. |
| 1 | `body_build` · thân chắc khỏe, ngực sâu | SUPPORTED | fci-243 tr. 2, 4, General Appearance, Body | Giữ và bổ sung đối chiếu dáng với ID 2. |
| 1 | `coat_texture` · lông hai lớp dày | SUPPORTED | fci-243 tr. 5, Coat | Giữ cấu trúc hai lớp làm mô tả chung, nêu lông ngoài thô và mùa thay lông. |
| 2 | `ear_position` · tai dựng | SUPPORTED | fci-270 tr. 3, Ears | Giữ. Thêm vị trí đặt cao trong claim KB. |
| 2 | `coat_texture` · bộ lông hai lớp dày | SUPPORTED_WITH_SCOPE | fci-270 tr. 5, Coat | Lớp lót dày hỗ trợ mô tả, tiêu chuẩn cho phép thiếu lớp lót lúc rụng lông. Nêu lông ngoài thẳng và áp thân. |
| 3 | `coat_color` · nền đen với mảng nâu đỏ và trắng | SUPPORTED | fci-058 tr. 5, Colour | Giữ cùng giới hạn tam thể chồng lấn. |
| 3 | `body_build` · thân cơ bắp chắc khỏe | SUPPORTED | fci-058 tr. 3–4, General Appearance, Body | Giữ, nêu xương nặng. |
| 4 | `tail_shape` · đuôi cuộn cao trên lưng khi vận động | SUPPORTED_WITH_SCOPE | fci-046 tr. 5, Tail | Giữ điều kiện chuyển động và mô tả đuôi cuộn chặt. Lúc nghỉ đuôi có thể thả. |
| 4 | `coat_color` · nền đen hoặc nâu havana với mảng nâu đỏ và trắng | SUPPORTED | fci-046 tr. 6, Colour and Markings | Giữ hai màu nền, Havana chỉ là một lựa chọn. |
| 5 | `coat_length` · lông ngắn và ôm sát thân | SUPPORTED | fci-047 tr. 6, Coat | Giữ thuộc tính KB chung. FCI 058 cũng cho phép lông ngoài ngắn. |
| 5 | `coat_color` · nền đen với mảng nâu vàng đến nâu đỏ và trắng | SUPPORTED | fci-047 tr. 6, Colour and Markings | Giữ dải mảng tan vàng-nâu đến đỏ-nâu. |
| 6 | `coat_texture` · lông dày, có thể xoăn hoặc bện dây | SUPPORTED_WITH_SCOPE | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | Hai kiểu lông đúng, không dùng để phân cỡ. Grooming ảnh hưởng. |
| 6 | `ear_position` · tai rủ dọc má | SUPPORTED | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | Giữ claim tai rủ sát má. |
| 7 | `coat_texture` · lông dày, có thể xoăn hoặc bện dây | SUPPORTED_WITH_SCOPE | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | Giữ chung cả ba cỡ với giới hạn grooming. |
| 7 | `ear_position` · tai rủ dọc má | SUPPORTED | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | Giữ claim tai rủ sát má. |
| 8 | `coat_texture` · lông dày, có thể xoăn hoặc bện dây | SUPPORTED_WITH_SCOPE | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | Giữ chung cả ba cỡ với giới hạn grooming. |
| 8 | `ear_position` · tai rủ dọc má | SUPPORTED | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | Giữ claim tai rủ sát má. |
| 9 | `coat_texture` · lông thô cứng với râu ở mõm | NEEDS_REVISION | fci-183 tr. 5, Coat Hair | Lông thô và râu đều có căn cứ. Tách râu khỏi `coat_texture` sang `facial_hair`. |
| 9 | `ear_position` · tai tự nhiên gập và rủ | SUPPORTED_WITH_SCOPE | fci-183 tr. 3, Ears | Giữ dưới scope tai tự nhiên. |
| 10 | `coat_texture` · lông thô cứng với râu ở mõm | NEEDS_REVISION | fci-182 tr. 5, Coat Hair | Lông thô và râu đều có căn cứ. Tách râu khỏi `coat_texture`. |
| 10 | `ear_position` · tai tự nhiên gập và rủ | SUPPORTED_WITH_SCOPE | fci-182 tr. 3, Ears | Giữ dưới scope tai tự nhiên. |
| 11 | `coat_texture` · lông thô cứng với râu ở mõm | NEEDS_REVISION | fci-181 tr. 5, Coat Hair | Lông thô và râu đều có căn cứ. Tách râu khỏi `coat_texture`. |
| 11 | `ear_position` · tai tự nhiên gập và rủ | SUPPORTED_WITH_SCOPE | fci-181 tr. 4, Ears | Giữ dưới scope tai tự nhiên. |
| 12 | `coat_texture` · lông dài, mượt, hơi gợn ở sau đầu ngang tai | SUPPORTED | fci-002 tr. 4, Coat Hair | Giữ đúng vị trí lông hơi gợn, không khái quát gợn trên toàn thân. |
| 12 | `coat_color` · lông có đốm lấm tấm kiểu belton, ưu tiên phân bố đều | SUPPORTED_WITH_SCOPE | fci-002 tr. 4, Colour | Giữ điều kiện về mẫu belton, ‘lấm tấm đều’ là ưu tiên hơn mảng lớn. |
| 13 | `coat_color` · lông màu hạt dẻ đậm | SUPPORTED_WITH_SCOPE | fci-120 tr. 4, Colour | Giữ màu nền hạt dẻ. Chuẩn cho phép một số dấu trắng nhỏ. |
| 13 | `ear_position` · tai thấp, rủ sát đầu | SUPPORTED | fci-120 tr. 3, Ears | Giữ. Tai rủ thấp chồng lấn nhóm. |
| 14 | `coat_color` · lông đen với mảng nâu đỏ | SUPPORTED | fci-006 tr. 4, Colour and Tan Markings | Giữ, bổ sung vị trí mảng tan và ngoại lệ chấm trắng rất nhỏ ở ngực. |
| 14 | `ear_position` · tai thấp, nằm sát đầu | SUPPORTED | fci-006 tr. 3, Ears | Giữ thuộc tính chung. |
| 15 | `ear_shape` · tai dạng thùy, dài và rộng | SUPPORTED | fci-125 tr. 3, Ears | Giữ thuộc tính KB, độ dài viền lông không phải số đo vành. |
| 15 | `coat_color` · lông đen trắng hoặc nâu gan trắng, có thể kèm mảng nâu vàng | SUPPORTED | fci-125 tr. 4, Colour | Giữ đen-trắng/nâu gan-trắng có thể thêm tan. Chồng lấn ID 17. |
| 16 | `coat_color` · lông đỏ đậm và trắng | SUPPORTED | rkc-welsh-springer, Colour; fci-126 tr. 4, Colour | Hai nguồn đồng thuận đỏ đậm-trắng. |
| 16 | `coat_texture` · lông thẳng hoặc phẳng, mượt và dày | SUPPORTED | rkc-welsh-springer, Coat; fci-126 tr. 4, Coat | Hai nguồn đồng thuận thẳng/phẳng, mượt, dày. Nêu giới hạn về ảnh khi nói kết cấu. |
| 17 | `ear_shape` · tai dạng thùy, phần vành tai kéo tới chóp mũi | SUPPORTED_WITH_SCOPE | fci-005 tr. 3, Ears; princeton-wordnet, `data.noun` 02102318 | Chóp mũi là mốc khi đối chiếu vành được duỗi, chỉ dùng ‘long, lobed ears’ trong prompt. |
| 17 | `coat_texture` · lông phẳng và mượt | SUPPORTED_WITH_SCOPE | fci-005 tr. 4, Hair; princeton-wordnet, `data.noun` 02102318 | Đọc chuẩn English Cocker cho morphology. Gloss dùng ‘wavy’ nhưng chuẩn FCI quy định phẳng, nên WordNet chỉ neo taxonomy. |
| 18 | `ear_position` · tai chữ V rủ về phía trước | SUPPORTED_WITH_SCOPE | fci-272 tr. 3, Ears | Mô tả đúng. Dáng V thuộc hình, câu prompt vị trí chỉ giữ hướng rủ. |
| 18 | `coat_texture` · lông cứng và thô | SUPPORTED | fci-272 tr. 4, Coat | Giữ mô tả lông cứng thô. |
| 19 | `ear_position` · tai chóp nhọn, dựng khi chú ý | SUPPORTED_WITH_SCOPE | fci-072 tr. 3, Ears | Tiêu chuẩn xác nhận tai dựng lúc chú ý, có thể ngả khi nghỉ. Tách chóp nhọn khỏi claim tư thế. |
| 19 | `coat_texture` · lông cứng và thô | SUPPORTED | fci-072 tr. 4, Coat | Giữ mô tả lông cứng thô. |
| 20 | `ear_position` · tai chữ V nhỏ rủ về phía má | SUPPORTED_WITH_SCOPE | fci-010 tr. 2, Ears | Dáng V, độ nhỏ và hướng rủ có căn cứ. Giữ hướng rủ ở ID ear_position, chuyển tín hiệu chính sang đầu. |
| 20 | `coat_texture` · lông thô và dày | SUPPORTED_WITH_SCOPE | fci-010 tr. 3, Coat | Lông thô dày có căn cứ, độ dày lớp lót khó xác nhận từ ảnh. |
| 21 | `ear_position` · tai bán dựng, chóp gập về trước khi chú ý | SUPPORTED_WITH_SCOPE | fci-156 tr. 4, Ears; fci-296 tr. 3, Ears | Cả hai chuẩn hỗ trợ khi chú ý, phạm vi biến thể ảnh của synset chưa biết. |
| 22 | `coat_length` · lông ngoài dài, tạo bờm quanh cổ | SUPPORTED | fci-088 tr. 4, Coat Hair | Giữ lông ngoài dài và bờm. Có chồng lấn Rough Collie. |
| 22 | `ear_position` · tai bán dựng khi chú ý | SUPPORTED_WITH_SCOPE | fci-088 tr. 3, Ears | Giữ tai bán dựng khi chú ý. Chóp gập ra trước còn có ở ID 21. |
| 23 | `ear_position` · tai dựng hoặc bán dựng | SUPPORTED | fci-297 tr. 3, Ears | Giữ cả hai tư thế tai, có chồng lấn nhóm. |
| 23 | `tail_shape` · đuôi thấp với đoạn chót cong lên | SUPPORTED_WITH_SCOPE | fci-297 tr. 3, Tail | Đuôi thấp và chót cong đúng mô tả, có thể nâng khi kích thích và chồng lấn ID 21–22. |
| 24 | `coat_color` · lông màu vàng hoặc kem | SUPPORTED | fci-111 tr. 3, Colour | Giữ vàng/kem, chồng Labrador vàng nhạt. |
| 24 | `coat_texture` · lông phẳng hoặc gợn, có lông dài trang trí | NEEDS_REVISION | fci-111 tr. 3, Coat Hair | Cả gợn/phẳng và feathering có căn cứ. Chuyển feathering sang claim chiều dài cục bộ. |
| 25 | `coat_length` · lông ngắn và dày | SUPPORTED | fci-122 tr. 4, Coat Hair | Giữ ngắn và dày. Loại trừ ‘dày’ như phép đo từ ảnh. |
| 25 | `tail_shape` · đuôi dày ở gốc và thuôn dần về chóp | SUPPORTED | fci-122 tr. 3, Tail | Giữ dáng dày gốc, thuôn; bổ sung tròn và không tua theo chuẩn. |
| 26 | `coat_texture` · lông có thể gợn ở vai, cổ, lưng và thắt lưng | SUPPORTED_WITH_SCOPE | fci-263 tr. 5, Coat Hair | Giữ xu hướng gợn theo vùng và tách chiều dài ngắn thành claim riêng. |
| 26 | `coat_color` · lông tông nâu, màu cói hoặc cỏ khô | SUPPORTED | fci-263 tr. 5, Colour | Giữ nhóm màu nâu, sedge, deadgrass. Không dùng màu riêng lẻ để tách lớp. |
| 27 | `muzzle_length` · mõm ngắn và rộng | SUPPORTED_WITH_SCOPE | akc-japanese-chin tr. 1, Head; fci-206 tr. 3, Facial Region | AKC nói rõ ngắn và rộng. Giữ chiều dài trong `muzzle_length`, chiều rộng là thông tin hình dạng. |
| 27 | `coat_texture` · lông thẳng và mượt | SUPPORTED | fci-206 tr. 4, Coat Hair | Giữ thẳng và mượt, phân biệt mô tả lông mặt. |
| 28 | `muzzle_length` · mõm rõ, có thể tương đối ngắn và rộng | SUPPORTED_WITH_SCOPE | fci-207 tr. 2, Facial Region | Giữ mõm phải hiện rõ, có thể ngắn tương đối. Chiều rộng thuộc hình dạng, bỏ ở claim chiều dài. |
| 28 | `coat_length` · lông tương đối dài, có bờm ở cổ | SUPPORTED | fci-207 tr. 4, Coat Hair | Giữ lông dài vừa, bờm cổ và giới hạn vị trí. |
| 29 | `coat_length` · lông ngoài dài và dày | SUPPORTED | fci-208 tr. 4, Coat Hair | Giữ dài/dày theo chuẩn, nêu grooming có thể che khác biệt. |
| 29 | `muzzle_length` · mõm ngắn, rộng | SUPPORTED_WITH_SCOPE | fci-208 tr. 3, Muzzle | Ngắn/rộng có căn cứ. Tách vai trò chiều dài khỏi mô tả hình dạng; ưu tiên lông mọc ngược làm tín hiệu. |

### 6.3. Bằng chứng tới từng claim và phrase đề xuất

Cột Loại dùng **chung** hoặc **phân biệt** như mục 5. “Có” ở cột Bank nghĩa phrase được dùng nguyên văn trong prompt chính của chính lớp đó. Các phrase còn lại giải thích claim và hỗ trợ review, gồm các phrase S dành cho trao đổi KB hoặc biến thể size prior đã nêu. Source IDs nối 1:1 với danh mục 6.1, trong đó có URL và version tương ứng.

| ID | Attribute ID / Loại | Claim tiếng Việt | Morphology phrase tiếng Anh | V/C/S | Điều kiện quan sát | Source ID và locator | Semantic review | Bank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `coat_fullness` · chung | Bộ lông trông dày, theo gloss của synset chó kéo xe Bắc Cực. | a thick coat | C | Thân và phần lông ngoài thấy rõ, phụ thuộc mùa và hình chiếu. | princeton-wordnet, `data.noun` offset 02109961 | SUPPORTED_WITH_SCOPE | Có |
| 1 | `coat_texture` · chung | Lông bảo vệ thô trên lớp lót dày. | a coarse outer coat | C | Vùng thân rõ, mức dày lớp lót phụ thuộc mùa, kết cấu sợi cần cận cảnh. | fci-243 tr. 5, Coat | SUPPORTED_WITH_SCOPE | — |
| 1 | `body_build` · phân biệt | Thân lực lưỡng, xương nặng và ngực sâu. | a powerful deep-chested build | C | Thấy ngang thân và chân. Lông dày, mỡ và góc máy có thể làm thân trông rộng. | fci-243 tr. 2–4, General Appearance, Proportions, Body | SUPPORTED_WITH_SCOPE | Có |
| 1 | `tail_shape` · phân biệt | Đuôi xù dạng chùm lông, thường đưa qua lưng khi không làm việc, không cuộn sát. | a bushy plumed tail | C | Đuôi hiện đầy đủ và thấy đường viền lưng. Tư thế thay đổi cách mang. | fci-243 tr. 4, Tail | SUPPORTED_WITH_SCOPE | Có |
| 2 | `coat_texture` · chung | Lông ngoài thẳng, khá áp thân, trên lớp lót mềm dày có thể thay theo mùa. | a smooth-lying outer coat | C | Lông khô, rõ vùng thân. Lớp lót không luôn nhìn thấy. | fci-270 tr. 5, Coat | SUPPORTED_WITH_SCOPE | — |
| 2 | `ear_position` · chung | Tai dựng, đặt cao và tương đối gần nhau. | high-set upright ears | V | Chính diện khi hai tai hiện rõ. | fci-270 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 2 | `body_build` · phân biệt | Dáng gọn vừa phải, ngực sâu nhưng không quá rộng. | a moderately compact build | C | Ảnh ngang đủ thân, tránh góc rộng gần đầu. So với chuẩn của ID 1 chỉ là nhận định tương đối. | fci-270 tr. 2, 4, General Appearance, Body | SUPPORTED_WITH_SCOPE | Có |
| 2 | `tail_shape` · phân biệt | Đuôi dạng chổi cong lưỡi liềm khi chú ý, có thể rủ lúc nghỉ. | a brush tail curving like a sickle when alert | C | Thấy cả đuôi khi chú ý. Không dựa vào chỉ riêng việc mang cao. | fci-270 tr. 4, Tail | SUPPORTED_WITH_SCOPE | Có |
| 3 | `coat_color` · chung | Nền đen với mảng nâu đỏ và trắng, kiểu tam thể điển hình. | black, tan and white markings | V | Thân và mặt đủ sáng. | fci-058 tr. 5, Colour | SUPPORTED_WITH_SCOPE | Có |
| 3 | `body_build` · phân biệt | Thân lực lưỡng, xương nặng và cơ bắp. | a heavy-boned muscular build | C | Chó đứng ngang, thấy chân và thân, đối chiếu cùng tuổi trưởng thành khi có thể. | fci-058 tr. 3–4, General Appearance, Body | SUPPORTED_WITH_SCOPE | Có |
| 3 | `tail_shape` · phân biệt | Đuôi nặng, rủ lúc nghỉ, chỉ cong nhẹ lên khi nâng. | a heavy tail with a slight upward curve when raised | C | Thấy đuôi và tư thế, ảnh tĩnh lúc nghỉ ít phân biệt hơn. | fci-058 tr. 4, Tail | SUPPORTED_WITH_SCOPE | — |
| 4 | `coat_color` · chung | Tam thể với mảng nâu đỏ và trắng, nền có thể đen hoặc nâu Havana. | tan and white markings on a black or Havana-brown coat | V | Ánh sáng trung tính. Nền đen chồng lấn ID 3 và 5. | fci-046 tr. 6, Colour and Markings | SUPPORTED_WITH_SCOPE | Có |
| 4 | `body_build` · phân biệt | Dáng gọn gần vuông theo mô tả, tỷ lệ cao:dài chuẩn 9:10. | a compact, nearly square build | C | Ảnh ngang, đủ điểm vai và mông. Tỷ lệ gần ID 3. | fci-046 tr. 3, General Appearance, Important Proportions | SUPPORTED_WITH_SCOPE | — |
| 4 | `tail_shape` · phân biệt | Đuôi cuộn chặt trên mông khi chuyển động. Khi nghỉ có thể thả. | a tightly curled tail over the rump when moving | C | Ảnh đang di chuyển và thấy đường viền đuôi. | fci-046 tr. 5, Tail | SUPPORTED | Có |
| 5 | `coat_color` · chung | Nền đen với mảng vàng nâu đến đỏ nâu và trắng. | black, tan and white markings | V | Mặt và thân nhìn rõ. Màu tam thể chồng lấn. | fci-047 tr. 6, Colour and Markings | SUPPORTED_WITH_SCOPE | Có |
| 5 | `coat_length` · chung | Lông ngoài ngắn, áp thân. | a short close-fitting coat | V | Thân rõ, tránh lông bị nén. FCI 058 cũng cho phép lông ngoài ngắn. | fci-047 tr. 6, Coat | SUPPORTED_WITH_SCOPE | — |
| 5 | `body_build` · phân biệt | Dáng gọn hơi thuôn dài, tỷ lệ cao:dài chuẩn 8:10. | a compact, slightly elongated build | C | Ảnh ngang trên mặt phẳng, đủ chân và thân. Tỷ lệ chuẩn không dùng như phép đo từ crop. | fci-047 tr. 3, General Appearance, Important Proportions | SUPPORTED_WITH_SCOPE | Có |
| 6 | `coat_texture` · chung | Lông xoăn dày hoặc bện dây. Đây là hai kiểu lông chuẩn, các cỡ dùng chung. | a curly or corded coat | C | Bộ lông đủ chi tiết ở vùng thân. Clip và grooming ảnh hưởng hình lông. | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | SUPPORTED_WITH_SCOPE | Có |
| 6 | `ear_position` · chung | Tai rủ dọc má. | ears hanging close to the cheeks | C | Thấy vành và vị trí tai. Tua dài có thể che dáng. | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | SUPPORTED_WITH_SCOPE | Có |
| 6 | `breed_size` · phân biệt | Chiều cao vai theo FCI trên 24 đến 28 cm, dung sai −1 cm (lý tưởng 25 cm). AKC đo ở điểm cao nhất của vai và giới hạn đến 10 in. | very small stature | S | Chó trưởng thành đứng trên bề mặt phẳng, đo ở vai đúng hệ chuẩn. Không suy từ crop. | fci-172 tr. 8, Size and Weight; akc-poodle tr. 1, Size, Proportion, Substance | SUPPORTED_WITH_SCOPE | — |
| 7 | `coat_texture` · chung | Lông xoăn dày hoặc bện dây. Đây là hai kiểu lông chuẩn, các cỡ dùng chung. | a curly or corded coat | C | Bộ lông đủ chi tiết ở vùng thân. Clip và grooming ảnh hưởng hình lông. | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | SUPPORTED_WITH_SCOPE | Có |
| 7 | `ear_position` · chung | Tai rủ dọc má. | ears hanging close to the cheeks | C | Thấy vành và vị trí tai. Tua dài có thể che dáng. | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | SUPPORTED_WITH_SCOPE | Có |
| 7 | `breed_size` · phân biệt | Chiều cao vai theo FCI trên 28 đến 35 cm. AKC đo ở điểm cao nhất của vai và dùng ngưỡng trên 10 đến 15 in. | small stature | S | Chó trưởng thành đứng trên bề mặt phẳng, đo ở vai đúng hệ chuẩn. Không suy từ crop. | fci-172 tr. 8, Size and Weight; akc-poodle tr. 1, Size, Proportion, Substance | SUPPORTED_WITH_SCOPE | — |
| 8 | `coat_texture` · chung | Lông xoăn dày hoặc bện dây. Đây là hai kiểu lông chuẩn, các cỡ dùng chung. | a curly or corded coat | C | Bộ lông đủ chi tiết ở vùng thân. Clip và grooming ảnh hưởng hình lông. | fci-172 tr. 6, Coat; akc-poodle tr. 2, Coat | SUPPORTED_WITH_SCOPE | Có |
| 8 | `ear_position` · chung | Tai rủ dọc má. | ears hanging close to the cheeks | C | Thấy vành và vị trí tai. Tua dài có thể che dáng. | fci-172 tr. 4, Ears; akc-poodle tr. 1, Head and Expression | SUPPORTED_WITH_SCOPE | Có |
| 8 | `breed_size` · phân biệt | Chiều cao vai theo FCI trên 45 đến 60 cm, dung sai +2 cm. AKC đo ở điểm cao nhất của vai và dùng ngưỡng trên 15 in. | larger stature | S | Chó trưởng thành đứng trên bề mặt phẳng, đo ở vai đúng hệ chuẩn. Không suy từ crop. | fci-172 tr. 8, Size and Weight; akc-poodle tr. 1, Size, Proportion, Substance | SUPPORTED_WITH_SCOPE | — |
| 9 | `coat_texture` · chung | Lông ngoài thô cứng dạng sợi, tương đối áp thân. | a wiry coat | C | Thấy vùng lông thân chưa tỉa quá sát. Ảnh không kiểm độ cứng khi sờ. | fci-183 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 9 | `facial_hair` · chung | Râu ở mõm và lông mày rậm. | a pronounced beard | C | Mặt rõ và grooming phù hợp. Cụm prompt chỉ dùng phần râu. | fci-183 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 9 | `ear_position` · chung | Tai tự nhiên gập rủ về phía trước. | naturally folded drop ears | C | Tai nguyên vẹn và nhìn rõ. Tư thế hoặc can thiệp vào tai làm mất tín hiệu. | fci-183 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 9 | `body_build` · chung | Dáng gần vuông theo tỷ lệ tiêu chuẩn. | a nearly square build | C | Ảnh ngang đủ vai và thân, không dùng để phân biệt ba cỡ. | fci-183 tr. 2, General Appearance and Important Proportions | SUPPORTED_WITH_SCOPE | — |
| 9 | `breed_size` · phân biệt | Chiều cao vai chuẩn FCI 30–35 cm cho cả hai giới. | small stature | S | Chó trưởng thành đứng thẳng, đo ở vai. Không xếp cỡ qua ảnh không có chuẩn tỷ lệ. | fci-183 tr. 6, Size and Weight | SUPPORTED_WITH_SCOPE | — |
| 10 | `coat_texture` · chung | Lông ngoài thô cứng dạng sợi, tương đối áp thân. | a wiry coat | C | Thấy vùng lông thân chưa tỉa quá sát. Ảnh không kiểm độ cứng khi sờ. | fci-182 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 10 | `facial_hair` · chung | Râu ở mõm và lông mày rậm. | a pronounced beard | C | Mặt rõ và grooming phù hợp. Cụm prompt chỉ dùng phần râu. | fci-182 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 10 | `ear_position` · chung | Tai tự nhiên gập rủ về phía trước. | naturally folded drop ears | C | Tai nguyên vẹn và nhìn rõ. Tư thế hoặc can thiệp vào tai làm mất tín hiệu. | fci-182 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 10 | `body_build` · chung | Dáng gần vuông theo tỷ lệ tiêu chuẩn. | a nearly square build | C | Ảnh ngang đủ vai và thân, không dùng để phân biệt ba cỡ. | fci-182 tr. 2, General Appearance and Important Proportions | SUPPORTED_WITH_SCOPE | — |
| 10 | `breed_size` · phân biệt | Chiều cao vai chuẩn FCI 45–50 cm cho cả hai giới. | medium stature | S | Chó trưởng thành đứng thẳng, đo ở vai. Không xếp cỡ qua ảnh không có chuẩn tỷ lệ. | fci-182 tr. 6, Size and Weight | SUPPORTED_WITH_SCOPE | — |
| 11 | `coat_texture` · chung | Lông ngoài thô cứng dạng sợi, tương đối áp thân. | a wiry coat | C | Thấy vùng lông thân chưa tỉa quá sát. Ảnh không kiểm độ cứng khi sờ. | fci-181 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 11 | `facial_hair` · chung | Râu ở mõm và lông mày rậm. | a pronounced beard | C | Mặt rõ và grooming phù hợp. Cụm prompt chỉ dùng phần râu. | fci-181 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 11 | `ear_position` · chung | Tai tự nhiên gập rủ về phía trước. | naturally folded drop ears | C | Tai nguyên vẹn và nhìn rõ. Tư thế hoặc can thiệp vào tai làm mất tín hiệu. | fci-181 tr. 4, Ears | SUPPORTED_WITH_SCOPE | — |
| 11 | `body_build` · chung | Dáng gần vuông theo tỷ lệ tiêu chuẩn. | a nearly square build | C | Ảnh ngang đủ vai và thân, không dùng để phân biệt ba cỡ. | fci-181 tr. 2, General Appearance and Important Proportions | SUPPORTED_WITH_SCOPE | — |
| 11 | `breed_size` · phân biệt | Chiều cao vai chuẩn FCI 60–70 cm cho cả hai giới. | large stature | S | Chó trưởng thành đứng thẳng, đo ở vai. Không xếp cỡ qua ảnh không có chuẩn tỷ lệ. | fci-181 tr. 6, Size and Weight | SUPPORTED_WITH_SCOPE | — |
| 12 | `coat_texture` · chung | Lông thường dài mượt và hơi gợn ở vùng sau đầu ngang tai, có lông tua. | silky, lightly wavy hair behind the head | C | Góc nghiêng, thấy lông sau đầu và tai, phụ thuộc grooming. | fci-002 tr. 4, Coat | SUPPORTED_WITH_SCOPE | — |
| 12 | `coat_color` · phân biệt | Các phối màu belton có nền trắng và chấm màu. Tiêu chuẩn ưu tiên chấm trải đều hơn mảng lớn. | belton-style flecking | V | Thấy nhiều vùng thân. Mảng lớn và mẫu ba màu vẫn có thể có. | fci-002 tr. 4, Colour | SUPPORTED_WITH_SCOPE | Có |
| 13 | `ear_position` · chung | Tai đặt thấp về phía sau, rủ sát đầu. | low-set ears hanging close to the head | V | Hai bên đầu rõ, phân biệt bờ tai với lông tua. | fci-120 tr. 3, Ears | SUPPORTED | — |
| 13 | `coat_color` · phân biệt | Nền hạt dẻ đậm, không có đen theo FCI, cho phép dấu trắng nhỏ ở một số vị trí. | a rich chestnut coat | V | Chiếu sáng trung tính, quan sát toàn nền lông. | fci-120 tr. 4, Colour | SUPPORTED_WITH_SCOPE | Có |
| 14 | `ear_position` · chung | Tai đặt thấp và áp sát đầu. | low-set drop ears | V | Thấy mặt bên và vị trí tai gắn. | fci-006 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 14 | `coat_color` · phân biệt | Nền đen với các điểm tan ở vùng như trên mắt, mõm và ngực. Chấm trắng rất nhỏ ở ngực được phép. | a black coat with tan points over the eyes and on the chest | V | Thấy đủ mặt và ngực, tránh tối vùng màu tan. | fci-006 tr. 4, Colour and Tan Markings | SUPPORTED_WITH_SCOPE | Có |
| 15 | `ear_shape` · chung | Tai dạng thùy, có chiều dài và chiều rộng rõ, có lông tua. | long, broad lobed ears | C | Thấy dáng vành tai, lông tua có thể làm tai nhìn dài hoặc rộng hơn. | fci-125 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 15 | `coat_color` · phân biệt | Đen-trắng hoặc nâu gan-trắng, có thể kèm mảng tan. | black-and-white or liver-and-white markings, sometimes with tan | V | Ánh sáng trung tính, màu da và lông tránh lẫn. Chồng lấn ID 17. | fci-125 tr. 4, Colour | SUPPORTED | Có |
| 15 | `body_build` · phân biệt | Dáng cân đối, chắc, tương đối cao chân trong nhóm spaniel đất liền Anh theo FCI. | an athletic, high-on-leg build | C | Chó đứng ngang, nhìn đủ chân và thân. Không suy chiều cao thật từ pixel. | fci-125 tr. 2, General Appearance | SUPPORTED_WITH_SCOPE | Có |
| 16 | `coat_texture` · chung | Lông thẳng hoặc phẳng, mượt và dày. | a straight, silky coat | C | Quan sát lông khô rõ nét. Không suy cảm giác khi chạm từ ảnh. | fci-126 tr. 4, Coat; rkc-welsh-springer, Coat | SUPPORTED_WITH_SCOPE | — |
| 16 | `coat_color` · phân biệt | Màu đỏ đậm và trắng theo hai tiêu chuẩn được đối chiếu. | rich red and white markings | V | Thấy thân, kiểm ánh sáng đỏ/cam; dải màu ảnh có thể chồng ID 17. | fci-126 tr. 4, Colour; rkc-welsh-springer, Colour | SUPPORTED | Có |
| 16 | `ear_shape` · phân biệt | Vành tai tương đối nhỏ và thuôn dần về chóp. | small, tapering ears | C | Cần thấy vành tai, góc ngang hoặc ba phần tư, tránh lông che. | fci-126 tr. 3, Ears; rkc-welsh-springer, Ears | SUPPORTED_WITH_SCOPE | Có |
| 17 | `coat_texture` · chung | Lông phẳng, mượt theo chuẩn English Cocker. | a flat, silky coat | C | Thấy lông tự nhiên, không suy từ bóng phản chiếu. WordNet dùng để chọn tên tương đương. | fci-005 tr. 4, Hair | SUPPORTED_WITH_SCOPE | — |
| 17 | `ear_shape` · phân biệt | Tai dạng thùy, vành mảnh dài tới chóp mũi khi duỗi theo phép đối chiếu chuẩn. | long, lobed ears | C | Ảnh thường chỉ hỗ trợ độ dài tương đối. Cần thao tác đo thật để kiểm chính xác chóp mũi. | fci-005 tr. 3, Ears | SUPPORTED_WITH_SCOPE | Có |
| 17 | `body_build` · phân biệt | Dáng gọn, chắc và cân đối, hai kích thước vai–đất và vai–gốc đuôi gần nhau theo tiêu chuẩn. | a compact, balanced build | C | Ảnh bên đủ toàn thân, chân đứng tự nhiên. Không ước tính số đo từ crop. | fci-005 tr. 2, General Appearance, Important Proportions | SUPPORTED_WITH_SCOPE | Có |
| 18 | `coat_texture` · chung | Lông cứng, thẳng và thô, nằm sát thân. | a hard wiry coat | C | Vùng thân rõ và không vừa tỉa sát. | fci-272 tr. 4, Coat | SUPPORTED | Có |
| 18 | `ear_position` · phân biệt | Tai rủ ra trước và áp gần má. | forward-dropping ears close to the cheeks | C | Thấy hướng mang tai từ chính diện hoặc ba phần tư. Dáng tai chữ V là dấu hiệu hình tai riêng. | fci-272 tr. 3, Ears | SUPPORTED_WITH_SCOPE | Có |
| 18 | `head_shape` · phân biệt | Đầu rộng hơi vồng, mõm nêm khỏe. | a broad head with a wedge-shaped muzzle | C | Thấy mặt trước và profile, không dùng đầu để tách riêng Norfolk với Norwich. | fci-272 tr. 3, Cranial and Facial Regions | SUPPORTED_WITH_SCOPE | — |
| 19 | `coat_texture` · chung | Lông cứng, thẳng và thô, áp sát thân. | a hard wiry coat | C | Grooming và độ phân giải có thể che kết cấu lông. | fci-072 tr. 4, Coat | SUPPORTED_WITH_SCOPE | Có |
| 19 | `ear_position` · phân biệt | Tai dựng lúc chú ý, có thể ngả ra sau lúc nghỉ. | ears held upright when alert | C | Hai tai rõ, chó đang chú ý. Tai nhọn thuộc hình tai, chưa cần xuất ID riêng ở lớp này. | fci-072 tr. 3, Ears | SUPPORTED_WITH_SCOPE | Có |
| 20 | `coat_texture` · chung | Lông ngoài thô trên lớp lót sát thân. | a harsh outer coat | C | Lông khô đủ chi tiết. Mật độ lớp lót không suy chắc từ ảnh. | fci-010 tr. 3, Coat | SUPPORTED_WITH_SCOPE | — |
| 20 | `ear_position` · chung | Tai rủ về trước gần má. | forward-dropping ears | C | Tai hai bên rõ. Chồng lấn Norfolk. | fci-010 tr. 2, Ears | SUPPORTED_WITH_SCOPE | Có |
| 20 | `head_shape` · phân biệt | Đầu kiểu rái cá, với sọ rộng vừa và mõm ngắn khỏe. | an otter-like head | C | Chính diện và góc ngang rõ mõm, chỉ gợi ý khi thấy hình học đầu. | fci-010 tr. 2, Head, Cranial Region, Facial Region | SUPPORTED | Có |
| 21 | `ear_position` · chung | Tai bán dựng khi chú ý, chóp gập ra trước. Cả chuẩn Rough và Smooth mô tả cách mang này. | semi-erect ears tipped forward when alert | C | Thấy hai tai, chó chú ý. Chồng lấn Sheltie và một số Border Collie. | fci-156 tr. 4, Ears; fci-296 tr. 3, Ears | SUPPORTED_WITH_SCOPE | Có |
| 21 | `head_shape` · phân biệt | Đầu dạng nêm tù, thon dần, với đoạn chuyển trán–mõm nhẹ nhưng thấy được. | a tapering blunt-wedge head with a slight stop | C | Ưu tiên góc nghiêng đủ trán và mõm. Dạng nêm chồng lấn Sheltie. | fci-156 tr. 3, Head; fci-296 tr. 2–3, Head | SUPPORTED_WITH_SCOPE | Có |
| 21 | `coat_length` · chung | Hai chuẩn tham chiếu mô tả Rough có bờm rõ và Smooth có lông ngoài ngắn; phân bố biến thể của lớp dữ liệu chưa xác minh. | rough- or smooth-coated reference varieties | C | Chỉ dùng với bộ lông đủ rõ. Giữ biến thiên ở KB, không chèn vào prompt. | fci-156 tr. 5, Coat; fci-296 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | — |
| 22 | `ear_position` · chung | Tai bán dựng, chóp gập ra trước lúc chú ý. | semi-erect ears with forward tips | C | Chó đang chú ý và thấy hai tai. Chồng lấn ID 21. | fci-088 tr. 3, Ears | SUPPORTED_WITH_SCOPE | — |
| 22 | `coat_length` · phân biệt | Lông ngoài dài, có bờm và diềm cổ rõ. | a long outer coat with a distinct neck ruff | C | Cổ và thân thấy rõ, lông khô, chồng lấn Rough Collie. | fci-088 tr. 2, 4, General Appearance, Coat | SUPPORTED_WITH_SCOPE | Có |
| 22 | `head_shape` · phân biệt | Đầu nêm tù dài từ tai tới mũi, stop nhẹ nhưng rõ. | a long wedge-shaped head with a slight stop | C | Ảnh ngang rõ trán–mõm. Dùng để phân biệt có điều kiện với ID 23. | fci-088 tr. 2, Head, Cranial Region | SUPPORTED_WITH_SCOPE | Có |
| 22 | `breed_size` · phân biệt | FCI ghi chiều cao vai lý tưởng đực 37 cm, cái 35,5 cm, sai khác quá 2,5 cm rất không mong muốn. | smaller stature relative to Collie under FCI standards | S | Đo thật ở vai chó trưởng thành. Dùng chuẩn tham chiếu, không suy từ kích thước ảnh. | fci-088 tr. 5, Size; fci-156 tr. 5, Size; fci-296 tr. 5, Size | SUPPORTED_WITH_SCOPE | — |
| 23 | `ear_position` · chung | Tai dựng hoặc bán dựng và linh hoạt theo trạng thái. | erect or semi-erect ears | C | Hai tai hiện rõ. Chồng lấn Collie và Sheltie. | fci-297 tr. 3, Ears | SUPPORTED_WITH_SCOPE | Có |
| 23 | `tail_shape` · chung | Đuôi đặt thấp thường có đoạn chót cong lên. | a tail with a slight upward sweep at the tip | C | Thấy toàn đuôi và tư thế. Chồng lấn cả ID 21 và 22. | fci-297 tr. 3, Tail | SUPPORTED_WITH_SCOPE | — |
| 23 | `head_shape` · phân biệt | Sọ khá rộng và đoạn chuyển trán–mõm rõ. | a head with a clearly marked stop | C | Ảnh nghiêng, đọc đường chuyển trán–mõm; không dùng như điều kiện đủ. | fci-297 tr. 2–3, Cranial and Facial Regions | SUPPORTED_WITH_SCOPE | Có |
| 24 | `coat_color` · phân biệt | Lông trong dải vàng hoặc kem trong tiêu chuẩn, có thể có ít sợi trắng ở ngực. | a golden or cream coat | V | Thân đủ sáng. Màu kem chồng lấn Labrador vàng nhạt. | fci-111 tr. 3, Colour | SUPPORTED_WITH_SCOPE | Có |
| 24 | `coat_texture` · phân biệt | Lông phẳng hoặc gợn. | flat or wavy hair | C | Góc đủ thấy thân, lông khô. Vài gợn lông đơn lẻ có thể chồng ID 26. | fci-111 tr. 3, Coat Hair | SUPPORTED_WITH_SCOPE | — |
| 24 | `coat_length` · phân biệt | Lông dài trang trí rõ ở dạng feathering theo chuẩn; nguồn không nêu một chiều dài toàn thân cố định. | distinct feathering | C | Thấy vùng lông tua, grooming hoặc crop có thể làm giảm tín hiệu. | fci-111 tr. 3, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 25 | `coat_length` · chung | Lông ngắn, dày, không có lông tua theo chuẩn. | a short, dense coat | V | Thân khô, đủ độ phân giải. Chiều dài ngắn cũng có ở ID 26. | fci-122 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 25 | `coat_texture` · phân biệt | Lông không gợn, sợi có độ cứng khi sờ. | an even, unwaved coat | C | Thân khô, có ánh sáng. Ảnh không xác nhận trực tiếp độ cứng. | fci-122 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | — |
| 25 | `tail_shape` · phân biệt | Đuôi tròn, rất dày gần gốc rồi thon dần, không tua, dạng đuôi rái cá. | a thick-based, tapering otter tail | C | Đuôi nhìn rõ từ gốc đến chóp, tránh góc foreshortening. | fci-122 tr. 3, Tail | SUPPORTED_WITH_SCOPE | Có |
| 26 | `coat_length` · chung | Lông thân ngắn, lông mặt và chân rất ngắn. Tua vừa ở vùng sau đùi và đuôi được phép. | a short coat | V | Thấy thân. Phân biệt lông ngắn toàn thân với tua cục bộ. | fci-263 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 26 | `coat_texture` · phân biệt | Lông có xu hướng gợn ở vai, cổ, lưng và thắt lưng. Lông mặt và chân thẳng. | waves over the neck, shoulders and back | C | Lông khô, nhìn đủ vùng vai–lưng. Gợn là xu hướng, không bắt buộc thấy. | fci-263 tr. 5, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 26 | `coat_color` · phân biệt | Dải tông màu nâu, sedge hoặc deadgrass theo FCI. | brown, sedge or deadgrass coloring | V | Ánh sáng trung tính. Màu gần nâu Labrador hoặc vàng nhạt Golden. | fci-263 tr. 5, Colour | SUPPORTED_WITH_SCOPE | — |
| 27 | `muzzle_length` · chung | Mõm ngắn theo AKC, FCI mô tả sống mũi rất ngắn. | a short muzzle | C | Nhìn góc nghiêng rõ mũi và đoạn chuyển. Ít phân biệt trong G10. | akc-japanese-chin tr. 1, Head; fci-206 tr. 3, Facial Region | SUPPORTED_WITH_SCOPE | — |
| 27 | `coat_texture` · chung | Lông thân thẳng và mượt. | a straight, silky coat | C | Lông khô, ảnh đủ nét để đánh giá bề ngoài. | fci-206 tr. 4, Coat; akc-japanese-chin tr. 1–2, Coat | SUPPORTED | — |
| 27 | `facial_hair` · phân biệt | Lông trên mặt và mõm ngắn, lông tai dài và nhiều. | short facial hair | C | Mặt và mõm rõ, tai ngoài vùng claim chính. Grooming làm biến đổi dấu hiệu. | akc-japanese-chin tr. 2, Coat; fci-206 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 27 | `coat_color` · phân biệt | FCI mô tả nền trắng với mảng đen hoặc đỏ. AKC có quy định thêm về tan points, dùng nguồn FCI cho phrase. | a white coat with black or red patches | V | Thấy đủ nền lông và mảng. Sắc độ ở vùng crop nhỏ có thể gây nhầm. | fci-206 tr. 4, Colour | SUPPORTED_WITH_SCOPE | Có |
| 28 | `muzzle_length` · chung | Mõm phải hiện rõ, có thể tương đối ngắn. Chiều rộng được ghi riêng ở head_shape nếu cần. | a visible, relatively short muzzle | C | Dùng góc nghiêng rõ đường viền. Không dùng độ bẹt để xếp loại. | fci-207 tr. 2, Facial Region | SUPPORTED_WITH_SCOPE | — |
| 28 | `head_shape` · phân biệt | Đầu tương đối rộng, đỉnh sọ phẳng giữa hai tai theo FCI. | a broad head flat between the ears | C | Ảnh mặt thẳng hoặc ba phần tư, cần tách đường viền lông và sọ. | fci-207 tr. 2, Head, Cranial Region | SUPPORTED_WITH_SCOPE | Có |
| 28 | `body_build` · phân biệt | Dáng nhỏ chắc vừa phải, thân cân bằng. | a moderately thick-set build | C | Ảnh ngang đủ thân. Lông không được tính là khối thân. | fci-207 tr. 2, General Appearance | SUPPORTED_WITH_SCOPE | — |
| 28 | `coat_length` · phân biệt | Lông thẳng dài vừa, bờm quây quanh cổ như áo choàng và không vượt quá vai. | a moderate mane around the neck | C | Cổ và vai hiện rõ. Bờm thay đổi do grooming. | fci-207 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 29 | `muzzle_length` · chung | Mõm ngắn theo chuẩn. Chiều rộng và độ vuông là mô tả khác của hình mõm. | a short muzzle | C | Ảnh ngang thấy mũi và điểm stop, tóc có thể che. Không suy 2,5 cm từ ảnh. | fci-208 tr. 3, Muzzle | SUPPORTED_WITH_SCOPE | — |
| 29 | `coat_length` · chung | Lông ngoài dài, dày, không xoăn theo chuẩn, có thể hơi gợn. | a long outer coat | C | Lông khô và rõ thân. Grooming có thể làm lông thực tế ngắn. | fci-208 tr. 4, Coat Hair | SUPPORTED_WITH_SCOPE | Có |
| 29 | `facial_hair` · phân biệt | Có râu, ria và lông mọc hướng lên trên mõm, tạo kiểu mặt như cánh hoa cúc. | muzzle hair growing upward | C | Mặt và mõm đủ rõ, buộc hoặc tỉa lông có thể đổi viền mặt. | fci-208 tr. 2–3, General Appearance, Head | SUPPORTED_WITH_SCOPE | Có |
| 29 | `body_build` · phân biệt | Thân chắc, chiều dài từ vai tới gốc đuôi lớn hơn chiều cao vai theo chuẩn. | a sturdy build longer than tall | C | Dáng ngang đủ vai và gốc đuôi. Không suy tỷ lệ chính xác từ góc xiên. | fci-208 tr. 2, General Appearance, Important Proportions | SUPPORTED_WITH_SCOPE | — |

### 6.4. Cách diễn giải loại khỏi bank chính hoặc runtime

| Mệnh đề | Semantic review | Disposition |
| --- | --- | --- |
| Eskimo Dog là Canadian Eskimo Dog theo FCI 211 | UNVERIFIABLE | WordNet và ImageNet cung cấp định danh lexical. FCI 211 áp riêng giống đăng ký. Giữ chỉ một coat claim ở ID 0. |
| Một chó EntleBucher luôn có bobtail hoặc luôn có đuôi dài | NEEDS_REVISION | FCI 047 tr. 4 ghi hai dạng đều chấp nhận. Dùng ở phân tích cặp, chờ ID riêng nếu cần lưu runtime. |
| Các ngưỡng số FCI hoặc AKC là quy tắc gán nhãn Stanford Dogs | UNVERIFIABLE | Bài dataset §2 mô tả tải lớp từ ImageNet và chọn ảnh. Ghi chuẩn riêng trong KB và không suy nhãn đăng ký. |
| Collie trong dataset luôn là Rough hoặc chắc chắn có cả hai biến thể | UNVERIFIABLE | Hai tiêu chuẩn FCI xác lập phenotype hai biến thể, không cho tỷ lệ xuất hiện trong lớp ảnh. Giữ claim chung và variability note. |
| Một màu duy nhất phân tách mọi cỡ Schnauzer | NEEDS_REVISION | FCI 181–183 có các màu chồng lấn. Chiều cao là dấu hiệu cần đo. Bank dùng lông thô và râu. |
| Có thể suy chiều cao thực từ bounding box/crop không có tỷ lệ | UNVERIFIABLE | Khoảng cách camera và tiêu cự có thể cho cùng kích thước biểu kiến. Không đưa vào bank chính. |

## 7. So sánh trước và sau thiết kế

| Hạng mục | Draft 2 | Đề xuất |
| --- | --- | --- |
| Canonical mapping và Name Bank | 30 lớp, Name Bank `name-1` | Giữ 30/30 exact IDs/names và chuỗi name prompt hiện tại. |
| Nội dung KB | 58 records thuộc tính theo từng lớp | 95 records đề xuất có ID, claim, phạm vi quan sát và locator. Số lượng phản ánh nội dung thiết kế, không đo chất lượng hay độ chính xác. |
| Bằng chứng phân biệt | Mô tả từng breed là chính | Đối chiếu 30 cặp trong 10 nhóm. Ghi tín hiệu đối chiếu và trường hợp thiếu căn cứ. |
| Poodle và Schnauzer | Hậu tố morphology chung theo từng nhóm | Giữ shared morphology trong bank chính. Thêm kích thước có hệ chuẩn vào KB. Chỉ G3 có biến thể thiết kế size prior để duyệt trước lock. |
| Eskimo Dog, Collie và Cocker Spaniel | Đã có ghi chú hoặc tên đồng nghĩa trong nguồn/alias | Liên kết trực tiếp WordNet. ID 0 dùng lexical coat claim, ID 21 giữ phạm vi biến thể có điều kiện, ID 17 chỉ dùng chuẩn English Cocker cho morphology. |
| Provenance và contract | Pin mapping/KB/bank, nguồn của draft 2 còn pending | Bổ sung locator, bản chuẩn chính xác và dấu hiệu được quan sát ở cấp claim. Việc ánh xạ sang JSON, source approval và pin lại bank thực hiện sau duyệt. |

## 8. Giới hạn khoa học

- Tiêu chuẩn giống mô tả hình thái lý tưởng hoặc điều kiện thi triển lãm. Cá thể, giống lai, grooming, tuổi, giới, mùa lông, ánh sáng và kiểu chụp tạo biến thiên. Quan sát phù hợp với một chuẩn chỉ là một dấu hiệu hỗ trợ.
- Với G3/G4 và cặp Collie–Shetland Sheepdog, crop và độ phóng đại làm mất tỷ lệ thực. Tuổi, vị trí chó và mặt phẳng tham chiếu phải được biết trước khi diễn giải kích thước. Các giá trị FCI và AKC thuộc từng hệ; không suy từ ảnh rằng chó thuộc một cỡ đăng ký.
- Ảnh có thể cho thấy đường viền hoặc lọn lông. Cảm giác khi chạm, độ dày lớp lót, tầm với chính xác của vành tai và chiều dài mõm đo giữa hai mốc cần căn cứ riêng. Với ảnh thiếu thông tin, assistant cần thể hiện unknown hoặc not_visible theo Contract 04.
- Dấu “D” áp dụng cho ít nhất một cặp trong nhóm và thường cần kết hợp dấu hiệu. Ví dụ, đuôi thấp chót cong ở nhóm Collie chồng lấn, vàng/kem có thể chồng giữa hai Retriever, tai rủ không tự tách Norfolk khỏi Border Terrier.
- Stanford Dogs dùng ảnh và taxonomy từ ImageNet. Tài liệu công bố phương pháp tạo dataset, không xác lập rằng từng ảnh tuân thủ ngưỡng kennel club, rằng ID 0 ánh xạ một-một sang FCI 211, hay phân bố Rough/Smooth trong ảnh của lớp Collie. Các mệnh đề đó được loại khỏi thiết kế runtime. [Stanford Dogs §2](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf)

## 9. Các quyết định cần trưởng nhóm phê duyệt

1. Phê duyệt các claim và locator trong bảng, gồm phạm vi lexical của ID 0, biến thể Collie và convention chọn English Cocker của ID 17. Xác nhận phiên bản tài liệu nguồn theo registry.
2. Chốt version và owner của allowlist riêng cho KB. Duyệt `breed_size`, `head_shape`, `facial_hair` và `coat_fullness`. Nếu `coat_fullness` chưa được duyệt, giữ ID 0 và cả bank đề xuất ở trạng thái draft cho tới khi có cách biểu diễn phù hợp Contract 01.
3. Chốt bank chính dùng chung E12-B/E13-B ở mục 5 và quyết định giữ size prior ngoài formal chính. Nếu cần đối chứng có size prior, xác định trước tập thay đổi, chỉ tiêu và cách ghi nhận ở Protocol Lock. Không chọn hậu tố sau khi xem Validation/formal.
4. Sau duyệt thiết kế, trưởng nhóm xác nhận nguồn/claim và trusted `review_ref` riêng trước khi dùng dữ liệu. Trạng thái Data Acceptance còn `PENDING` và Formal Evaluation Readiness còn `BLOCKED`. [AC-0804-01–04](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/docs/research/vlm/acceptance_criteria.md#L74)

## 10. Implementation plan sau duyệt

1. Chuẩn hóa JSON KB của revision kế tiếp với exact 30 IDs/names và aliases hiện có. Sắp xếp thuộc tính trong mỗi breed theo thứ tự ưu tiên đọc context đã chốt với Kiệt. Ghi nguyên điều kiện có ý nghĩa trong `value`, giữ source references và pin vocabulary version ở caller. Dùng tài liệu này làm bằng chứng thiết kế cho group, V/C/S và locator. Schema runtime hiện tại chỉ nhận các trường của `MorphAttribute` và `BreedRecord`.
2. Sau source approval chính thức, ghi review refs và pin bytes KB, mapping và allowlist/config. Dựng đúng 30 Morphology Bank entries với `kb_version`, `kb_sha256`, `mapping_sha256`. Tạo `attribute_ids` từ những claim gắn “Có” trong bảng. `source_ids` lấy từ các claim thực sự dùng. Kiểm bằng tay tới cấp phrase vì validator chỉ kiểm membership/provenance, không suy ngữ nghĩa.
3. Kiểm structure, hash mismatch, approved-source binding, allowlist KB, rejection ở Visual Observation, stable retrieval order, ambiguity, budget và citations bằng fixtures. Với 95 records, kiểm context nhỏ và nhiều giống để thấy claim phân biệt ưu tiên ra trước, không để mô tả chung chiếm hết budget. Xác nhận F2 ở caller assistant và giữ ranh giới dữ liệu scorer.
4. Đức Anh kiểm cả 30 prompts theo hai native tokenizer/profile thực tế trước lock, pin audit và ghi token limit/truncation. Cố định bank và template chung, model/processor revisions, canonical order và scoring cho cặp A/B như protocol. Dùng lại chính sách sau duyệt. Các bước kiểm runtime và native thuộc lần triển khai sau. [Research Protocol §§5, 8](https://github.com/DGiang1126/dogbreedid-cv-project/blob/448e890ed5e85b8d543304f2cbcaaadbfe619fa2/docs/research/vlm/research_protocol.md)

**Kết luận — READY_FOR_DESIGN_APPROVAL**
