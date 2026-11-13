# knowledge/ — dữ kiện bất biến

Chỉ ghi thứ không đổi theo phương pháp hay theo thời gian chạy. Trạng thái, quyết định, kết quả ở [CONTEXT](../CONTEXT.md) và `docs/`.

| file | nội dung |
|---|---|
| [local-env.md](local-env.md), [local-runtime.json](local-runtime.json) | máy local, env conda `nckh`, cách chạy guarded |
| [runtime.json](runtime.json) | image Kaggle đã pin (2×T4) |
| [dataset.md](dataset.md) | dataset v4 (VeReMi NextGen, 5.343 client = xe nhận): cột, fold, số dòng, phân bố client, nguồn Kaggle |
| [input_order.json](input_order.json) | thứ tự 77 cột đầu vào (71 feature + 6 cờ NaN) — đóng băng |
| [label_mapping.json](label_mapping.json) | 16 lớp ↔ id — đóng băng |
| [bert.md](bert.md) | đặc tả FeatureBERT (model client mọi phương pháp): hợp đồng vào/ra, kiến trúc, engine grouped, representation, bản tỉa LwFedNIDS |
