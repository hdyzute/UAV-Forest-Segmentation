# Danh sách lỗi cần sửa — Bản thảo "Adaptive Multi-Loss Optimization for Class-Imbalanced Semantic Segmentation in UAV Forest Inspection"

> File này dùng làm chỉ dẫn cho AI agent (Antigravity) định vị và sửa trực tiếp trong file nguồn (LaTeX/Word).
> Cập nhật lần 5 (FINAL): tất cả các lỗi phát hiện qua 5 vòng review (A→F) đã được xử lý và xác nhận RESOLVED. Còn lại 3 gợi ý nâng cấp mang tính tùy chọn.

---

## 🎯 TÌNH TRẠNG TỔNG QUAN

**✅ 10/10 lỗi số liệu/trích dẫn/logic phát hiện qua 5 vòng review đầu đã được xử lý xong** — không còn mâu thuẫn số học hay claim sai nào trong Abstract, Table 1–5, hay các con số trong văn bản.

**🟠 Phát sinh thêm 2 mục mới (J, K) ở vòng rà soát Section 3 (Methodology)** — chưa phải lỗi sai, nhưng ảnh hưởng trực tiếp đến độ thuyết phục của luận điểm chính ("adaptive weighting" là đóng góp cốt lõi của bài) nên cần xử lý trước khi coi bài là hoàn chỉnh về mặt khoa học, không chỉ về mặt số liệu nhất quán.

| # | Vấn đề | Trạng thái |
|---|---|---|
| 1 | Reference [3] sai/không tồn tại (ICCP 2022 giả) | ✅ Resolved |
| 2 | Exp 1–4 chưa được định nghĩa trong bài | ✅ Resolved |
| 3 | Thiếu thông tin GPU/thời gian huấn luyện | ✅ Resolved |
| 4 | Table 2 và Table 3 lệch số học (SegFormer Exp2 Combined) | ✅ Resolved |
| 5 | Tiền xử lý ảnh không đồng bộ giữa 3 weather regime | ✅ Resolved |
| 6 | Số "+4.15→+4.88pp" chưa cập nhật sau khi sửa Table 2 | ✅ Resolved |
| 7 | Overclaim "matches" khi so sánh gain WildUAV với [3] | ✅ Resolved |
| 8 | Sky/Building bị đưa vào class subset dù không có nhãn thật trên WildUAV | ✅ Resolved |
| 9 | Abstract "+4.88pp" không khớp global max thực tế (+5.15pp) | ✅ Resolved |
| 10 (mục F) | Chưa mô tả cách xử lý road/water/people trong WildUAV eval | ✅ Resolved |
| 11 (mục J) | Thiếu ablation nối Exp1/Exp3 (đã định nghĩa) với phần Results (chỉ show Exp2 vs Exp4) | 🟠 Open — nên làm |
| 12 (mục K) | Section 3.4 chỉ nêu công thức σₖ, thiếu chi tiết khởi tạo/parameterize/giá trị hội tụ | 🟠 Open — nên làm |

---

## ✅ CHI TIẾT CÁC MỤC ĐÃ RESOLVED (giữ lại để đối chiếu/kiểm tra chéo)

- **Ref [3]** → đã đổi đúng thành *Scientific Data* 13, 298 (2026).
- **Section 4.3** → đã định nghĩa rõ Exp 1 (baseline CE), Exp 2 (ENet-weighted CE), Exp 3 (fixed compound loss), Exp 4 (adaptive, đề xuất).
- **Section 4.4** → đã bổ sung GPU (Tesla T4), thời gian huấn luyện/epoch (~11–12 phút), và lý do seed-check chỉ chạy 1 cấu hình (giới hạn compute).
- **Table 2**, ô SegFormer/Exp2/Combined → đã sửa khớp với Table 3 (0.7698).
- **Section 4.4** → tiền xử lý ảnh đã đồng bộ 768×1280 (giữ tỷ lệ 5:3 gốc) cho cả 3 weather regime.
- **Section 5.1 & Conclusion** → số gain Combined đã sửa đúng thành "+4.18 to +4.88 pp".
- **Section 5.4** → câu overclaim "matches" đã sửa thành diễn đạt thận trọng ("consistent... though magnitude differs, e.g. +0.2pp vs +1.2pp in [3]").
- **Section 5.4** → class mapping WildUAV đã sửa đúng theo mapping thật trong [3]: chỉ còn Forest Vegetation, Dirt Ground, Car (loại bỏ Sky/Building/Fences/Rocks vì không có nhãn thật phía WildUAV).
- **Abstract** → "+4.88pp" đã sửa thành "+5.15pp", khớp đúng global max toàn bài (DeepLabV3+, Sunny-only).
- **Section 5.4 (mục F)** → đã bổ sung câu mô tả rõ cách xử lý road/water/people:
  > *"pixels corresponding to target-dataset classes with no counterpart in the Forest Inspection taxonomy (such as road, water, and people) are assigned to an ignore_index and excluded from the test mIoU computation."*

  Giờ toàn bộ 9 lớp gốc của WildUAV (ground vegetation, trees, dirt ground, road, water, empty, static car, moving car, people) đã được xử lý rõ ràng và có căn cứ trong pipeline đánh giá: 3 lớp ánh xạ (Forest Veg/Dirt Ground/Car), phần còn lại bị ignore_index. Không còn khoảng mờ nào trong quy trình evaluation Table 5.

---

## 🟠 NÊN LÀM (không phải lỗi sai, nhưng ảnh hưởng trực tiếp đến độ thuyết phục của luận điểm chính — ưu tiên cao hơn nhóm gợi ý tùy chọn bên dưới)

### J. Thiếu ablation nối giữa Exp 1/Exp 3 (đã định nghĩa) và phần Results (chỉ show Exp 2 vs Exp 4)

**Vị trí:** Section 4.3 (định nghĩa 4 cấu hình) vs. Section 5 / Table 2, 3, 4 (chỉ so sánh Exp 2 và Exp 4).

**Vấn đề:** Section 4.3 định nghĩa đầy đủ 4 cấu hình:
- Exp 1: CE thường (không weighting)
- Exp 2: CE + ENet weighting
- Exp 3: Compound loss (Focal+Dice+CE) với trọng số **tĩnh** bằng nhau (λ=1.0)
- Exp 4: Compound loss với trọng số **adaptive** (đề xuất chính của bài)

Nhưng toàn bộ Table 2/3/4 trong Section 5 chỉ so sánh Exp 2 vs Exp 4. Exp 1 và Exp 3 không xuất hiện lại ở đâu.

**Hệ quả:** Không thể tách bạch được đóng góp đến từ đâu — liệu gain +4–5pp đến từ việc *gộp 3 loss* (Focal+Dice+CE, bất kể trọng số tĩnh hay adaptive), hay thực sự đến từ *cơ chế adaptive weighting*. Nếu Exp3 (static) ≈ Exp4 (adaptive), thì đóng góp cốt lõi mà Abstract claim ("dynamically balances... eliminate manual parameter tuning") sẽ không được chứng minh — compound loss tĩnh đã đủ tốt rồi. Đây gần như chắc chắn là câu hỏi đầu tiên reviewer sẽ hỏi, vì chính bài tự đặt kỳ vọng này khi liệt kê đủ 4 cấu hình.

**Việc cần làm:**
- [ ] Bổ sung ít nhất 1 bảng ablation đầy đủ Exp1 → Exp2 → Exp3 → Exp4 (tối thiểu cho Combined regime, lý tưởng là cả 3 regime nếu compute cho phép).
- [ ] Thêm 1 đoạn phân tích trong Section 5 so sánh riêng Exp3 vs Exp4 để chứng minh phần đóng góp thực sự của adaptive weighting so với compound loss tĩnh.

### K. Section 3.4 (Adaptive Uncertainty Loss Weighting) chỉ nêu công thức, thiếu chi tiết implementation của σₖ

**Vị trí:** Section 3.4, công thức (5).

**Vấn đề:** Công thức hiện tại chỉ định nghĩa toán học của σₖ mà không trả lời các câu hỏi cần thiết để tái lập (reproduce) và để làm bằng chứng cho luận điểm "adaptive":
- σₖ khởi tạo giá trị bao nhiêu? (thường trong Kendall et al. [8], khởi tạo log σ² = 0).
- σₖ được ràng buộc dương như thế nào để tránh chia cho 0 trong 1/(2σ²)? (cách chuẩn: parameterize qua s = log σ², dùng exp(-s) thay vì học trực tiếp σ).
- σₖ hội tụ về giá trị nào sau khi train xong? — đây là con số **quan trọng nhất còn thiếu**: nếu có, nó là bằng chứng trực tiếp nhất cho thấy cơ chế adaptive thực sự học được trọng số khác nhau giữa các loss/regime, thay vì chỉ là claim lý thuyết.

**Việc cần làm:**
- [ ] Thêm 1–2 câu trong Section 3.4 mô tả cách khởi tạo và parameterize σₖ (hoặc log σₖ²) trong implementation thực tế.
- [ ] Thêm 1 bảng nhỏ hoặc vài câu trong Section 5 báo cáo giá trị σ_CE, σ_Focal, σ_Dice học được cuối cùng cho ít nhất 1–2 cấu hình tiêu biểu (ví dụ UNet++/Combined).

---

## 🟢 GỢI Ý NÂNG CẤP — không phải lỗi, mang tính tùy chọn để bài mạnh hơn khi nộp tạp chí thật

Đây không phải lỗi cần sửa gấp, chỉ là các điểm sẽ giúp bài thuyết phục hơn với reviewer nếu còn thời gian trước khi nộp chính thức.

### G. Bổ sung hình minh họa (hiện bài chưa có hình nào)

**Vấn đề:** Toàn bộ bản thảo chỉ có bảng số liệu và công thức toán, không có bất kỳ hình ảnh nào — khác thường với một bài segmentation gửi TGRS.

**Gợi ý bổ sung:**
- [ ] 1 hình sơ đồ tổng quan phương pháp (pipeline: 3 loss component → adaptive uncertainty weighting → tổng loss).
- [ ] 1–2 hình so sánh định tính giữa Exp 2 và Exp 4 trên các lớp hiếm (fence, fallen trees, car) — minh chứng trực quan cho trade-off Precision/Recall đã nêu ở Section 5.3.

### H. Bổ sung per-class breakdown cho Table 5 (WildUAV)

**Vấn đề:** Table 5 hiện chỉ có 1 con số tổng (overall mIoU), trong khi các bảng khác trong bài (Table 3, 4) đều có breakdown theo từng lớp/kiến trúc.

**Gợi ý bổ sung:**
- [ ] Thêm bảng per-class IoU cho 3 lớp shared subset (Forest Vegetation, Dirt Ground, Car), so sánh Real-only vs Pretrain+Fine-tune — giúp minh chứng rõ hơn improvement +0.2pp đến từ đâu, thay vì chỉ có 1 con số tổng khó diễn giải.

### I. Mở rộng multi-seed robustness check

**Vấn đề:** Hiện seed 123/2025/42 chỉ chạy trên UNet++/Sunny-only — đã có giải thích lý do (giới hạn compute), chấp nhận được cho báo cáo tiến độ, nhưng nếu có thêm thời gian trước khi nộp thật:

**Gợi ý bổ sung (không bắt buộc):**
- [ ] Cân nhắc chạy thêm multi-seed cho ít nhất 1 cấu hình Combined (regime dùng cho phần lớn kết luận chính của bài) để tăng độ tin cậy thống kê.

---

## Ghi chú

- File này tổng hợp toàn bộ 5 vòng rà soát bản thảo, đối chiếu trực tiếp với nguồn [3] (Blaga & Nedevschi, *Scientific Data* 13, 298, 2026) và các quy định trang/định dạng của IEEE TGRS.
- Về mặt tính đúng đắn nội bộ (số liệu khớp giữa các bảng, trích dẫn chính xác, mô tả pipeline đầy đủ), bản thảo hiện tại **không còn lỗi nào được phát hiện**.
- Các mục G, H, I là tùy chọn — có thể bỏ qua nếu đây vẫn là báo cáo tiến độ cho thầy, và chỉ cần cân nhắc nghiêm túc khi chuẩn bị bản nộp chính thức lên tạp chí.