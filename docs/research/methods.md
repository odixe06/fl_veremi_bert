# Danh mục phương pháp: 14 baseline FL trên FeatureBERT

Nguồn: danh mục của [`fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/methods.md).
Owner 13/11 bỏ SCAFFOLD, không chạy MobiHet và centralized. Cấu hình chung: [protocol](protocol.md). Bài báo: [papers/](../../papers/README.md).
Đặc tả từng phương pháp nằm ở **docstring đầu file** `methods/<alias>.py`: công thức theo số phương trình của bài, siêu tham số, deviation.
Code method chép từ v2 (working tree đã chạy seed 41/43, 12/11). Chỉ sửa những chỗ phụ thuộc kiến trúc (cột "đổi cho BERT").

Cột "model chấm": **toàn cục** = một model, chấm trên toàn test; **client** = model riêng của mỗi xe, chấm trên test của chính xe đó.

| # | alias | bài | model chấm | trao đổi mỗi round | đổi cho BERT |
|---:|---|---|---|---|---|
| 1 | fedavg | McMahan 2017 | toàn cục | θ | — |
| 2 | fedprox | Li 2020 | toàn cục | θ | — |
| 3 | afpha | Khan 2025 (Sci. Rep.) | toàn cục | θ | — |
| 4 | fd_ids | Peng 2025 (Sensors) | toàn cục | θ (+ KD) | — |
| 5 | apfed | Sci. Rep. 2025 | client (headline) + global | θ | classifier = `head` (thay cho `head.5`) |
| 6 | lwfednids | Bouayad 2024 (Access) | toàn cục | θ đã tỉa | nhóm tỉa d / FFN / head-dim, 70 % ([bert §5](../../knowledge/bert.md)) |
| 7 | v2x_htfl | OJVT 2025 | client (headline) + client_proto | prototype | projection d×d trên CLS (d = 64) |
| 8 | canperfl | Hoang 2023 (Appl. Sci.) | client | θ (pretrain FedAvg r50) → fine-tune 10 epoch | — |
| 9 | tinyproto | Lee 2026 (AAAI) | client_proto (headline) + head | prototype thưa | s = ⌈64/10⌉ = 7 chiều/lớp; μ = 1,5e-4 (như v2 r4) |
| 10 | pfedes | Yi 2025 (AAAI) | client | extractor proxy nhỏ | G giữ nguyên trên lưới 7×11 của input (2.808 B) |
| 11 | nilm | Li 2025 (TSG) | client (headline) + proxy | proxy | cả hai model là FeatureBERT |
| 12 | fedtgp | Zhang 2024 (AAAI) | client_proto (headline) + head | prototype | K = d = 64 |
| 13 | perfedskd | Singh 2024 (TCE) | client | θ (client được chọn) | — |
| 14 | fedproto | Tan 2022 (AAAI) | client_proto (headline) + head | prototype | representation = CLS (d = 64) |

Đã loại: SCAFFOLD (owner 13/11), MobiHet/CBB (owner 13/11), centralized (owner 13/11).
Đối chiếu code với v2: [parity](v2-parity.md) (20 kiểm tra AST PASS).
Trạng thái chạy: [continue](../operations/continue.md). Kết quả: [results](results.md).
