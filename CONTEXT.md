# CONTEXT: fl_veremi_bert / 14 baseline FL với client FeatureBERT trên VeReMi NextGen v4 (5.343 xe)

Đọc theo thứ tự: CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).

## Bài toán

IDS V2X 16 lớp, học cộng tác, **mỗi xe nhận là một client** (5.343 client, dataset v4).
Repo này chạy lại các phương pháp của [`fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2) với **cấu hình hoàn toàn giống v2**.
Khác biệt duy nhất: model client đổi từ DAGSNet sang **FeatureBERT** (77.904 tham số), lấy từ repo
[`dangvanhoang1711/veremi-bert-77`](https://github.com/dangvanhoang1711/veremi-bert-77).
Code `core/` và `methods/` chép từ v2. Chỉ phần phụ thuộc kiến trúc được sửa ([methods](docs/research/methods.md)).

## Quyết định owner (13/11/2026)

- Cấu hình giống v2: 50 round, mọi client mỗi round, 1 epoch local trên fold train + val, batch 64, AdamW cosine 1e-3 → 1e-5,
  clip 1, AMP, gộp theo n_k. Đánh giá giữ như v2 (global trên toàn test; client trên test của chính xe; 10 metric). Chi tiết: [protocol](docs/research/protocol.md).
- **14 phương pháp FL**: các baseline của v2 **bỏ SCAFFOLD**. **Không chạy MobiHet/CBB.** **Bỏ centralized.**
- Bản đầu chỉ chạy **seed 42**.
- LwFedNIDS tỉa 70 % theo các nhóm d / head-dim / FFN của BERT ([knowledge/bert](knowledge/bert.md) §5).
- Kaggle: đẩy ngay trên các account còn quota, **chừa 4 account B20 của mobihet** (novahihi, keocaosu, emtrangtri, minhtran0601).
  Phần còn thiếu sẽ chạy tiếp sau khi quota reset (thứ Bảy 00:00 UTC).
- Repo GitHub **public** `odixe06/fl_veremi_bert`. Không đẩy data, PDF bài báo, credential. Env conda `nckh`. W&B project `fl-veremi-bert`.
- Skill và AGENTS chép từ `../mobihet`; cấu trúc tài liệu giống mobihet / v2.

## Trạng thái

Kiểm 13/11/2026 10:00 VN: 13 kernel v1 RUNNING; AFPHA có r1 T4, timing đã ghi LESSONS; CANPerFL chờ FedAvg r50.
Xem [bàn giao sống](docs/operations/continue.md). Kết quả: [results](docs/research/results.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, job nào chạy, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Cấu hình chung, cách chấm | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, chỗ đổi cho BERT | [methods](docs/research/methods.md) |
| Kết quả 50 round | [results](docs/research/results.md) |
| Model FeatureBERT, engine grouped, bản tỉa | [knowledge/bert](knowledge/bert.md) |
| Dataset, cột, số liệu | [knowledge/dataset](knowledge/dataset.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bản DAGSNet (kết quả 3 seed để so) | `../fl_dagsnet_veremi_v2/docs/research/results.md` |
