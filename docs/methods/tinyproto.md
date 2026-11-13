# TinyProto-FP — audit r3 và sửa scale r4 (30/10/2026)

> Kế thừa từ `fl_dagsnet_veremi_v2` (DAGSNet). Cơ chế và siêu tham số giữ nguyên khi chạy trên FeatureBERT. Số liệu, SHA và run bên dưới là lịch sử của v2,
> không phải trạng thái của repo này. Repo này bỏ SCAFFOLD (owner 13/11).

Bài gốc: [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/39441/43402),
[arXiv + appendix](https://arxiv.org/html/2507.04327v1),
[code tác giả](https://github.com/regulationLee/TinyProto) (đã đọc checkout `f2aa60794273e89d03c3019fff46f7ddf2f5f430`).
Đặc tả còn lại: docstring [methods/tinyproto.py](../../methods/tinyproto.py); [protocol chung](../research/protocol.md).

## Kết quả r3 và giới hạn kết luận

Kernel `anhtruong1909/fl2-tinyproto-r3`, source `55d13cc5d8b9e4d7fec0478e97c20dd1e717aab9`, đủ 50 round;
pull 102 file chọn (không lấy `state*.pt`), 0 thiếu, 1.621 MiB mới. Artifact: `runs/pulls/fl2-tinyproto-r3/`.
[Audit](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/audits/2026-10-30/fl2-tinyproto-r3.json): 5.343 client, support nhãn khớp scan toàn test 9.347.679 dòng,
10 metric tính lại khớp; tensor snapshot có sẵn đều hữu hạn. **PASS_BOUNDED không chứng minh phương pháp ổn định hoặc đã replay model**.

| r | F1 head | F1 prototype local | train_loss | norm target |
|---:|---:|---:|---:|---:|
| 1 | 0,7906 | 0,7860 | 0,5087 | 35,49 |
| 2 | 0,8242 | 0,8165 | 0,5979 | 80,39 |
| 3 | — | — | 0,989 | 200,60 |
| 5 | 0,8319 | 0,4827 | 43,21 | 1.254,84 |
| 10 | 0,7907 | 0,3058 | 198,89 | 2.552,51 |
| 50 | 0,7881 | **0,2904** | 87,88 | 2.943,11 |

Đây không còn là lỗi lấy prototype global thưa để suy luận: pin r3 đã có dense prototype local theo Eq.12;
r1–r2 head/prototype gần nhau rồi mới suy giảm. Không kết luận TinyProto yếu về bản chất từ run này.

## Cơ chế cần sửa

Eq.10 gộp `mean_holders(n_ij * c_ij)`; Eq.11 nhân μ. Với prototype cùng hướng, hệ số sau mỗi lần gộp của lớp j là:

`a_j = μ * sum_i(n_ij) / number_of_holders(j)`.

μ cũ = `1 / mean_positive(n_ij)` = **0,0018734104697128494** (heuristic của bản dựng, không phải luật chọn μ của bài).
Scan đầy đủ cột client_id/label của train cho a_benign=3,427; DoS=2,593; Sybil=2,099.
[Bằng chứng đếm](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/audits/2026-10-30/tinyproto-scaling.json). Hệ số >1 cho phép vòng lặp target→feature→target khuếch đại.
Đây là cơ chế đã tái hiện qua đường gộp thật; chưa chứng minh đó là nguyên nhân duy nhất của suy giảm production.

Bài gốc chọn μ riêng theo dataset, arXiv ghi grid search; README tác giả dùng `-csf 0.00015` cho CIFAR-10.
Không có căn cứ coi `1 / n̄` là giá trị mặc định đúng cho VeReMi.
Code public hiện tại còn khác Eq.12 về inference và có các vị trí indentation cần kiểm tra; không sao chép mù để thay công thức bài.

## R4 — owner đã cho chạy lại

- Chỉ thay scale mặc định bằng **μ=1,5e-4**, giá trị công bố và cấu hình mẫu tác giả; ghi HP tường minh `mu=0.00015`.
  Chuyển hằng số CIFAR-10 sang VeReMi là **deviation**, không phải HP tác giả đã xác nhận cho VeReMi.
- Giữ CPS, Eq.10, loss MSE, λ=1, inference dense local Eq.12, tất cả HP chung. Thêm log `aps_mu`, `aps_gain_max`,
  `proto_target_l2`, `proto_target_absmax`. Max a_j trên full train giảm còn khoảng 0,2744.
- Regression tái hiện 10 lần gộp prototype cùng hướng qua `Method.round`: code cũ FAIL r1 (max 1,2181), code sửa PASS;
  test 2-round method + evaluation/state cũng PASS.
- Smoke compiled+AMP trên 100 xe thật: **6-round probe LR riêng**, không phải headline hay phép so ngang r3.
  Train loss 0,575→0,624→0,379→0,280→0,222→0,199; norm target 2,060→0,846→0,632→0,564→0,532→0,524.
  μ cố định trước probe; không chọn theo test. Một GPU local chưa chứng minh hai T4 hay hội tụ 50 round.
- Chạy mới từ init r1, **50-round cosine chung**, không resume r3 sau khi đổi μ.
  Kernel `anhtruong1909/fl2-tinyproto-r4`, version 1, cap 8 h.
  Base SHA giữ `55d13cc...`; overlay `methods/tinyproto.py` SHA256
  `1bbf6b6e18f87a47f2ed52831c2709f8a3fbc001a7a2461c13a968ee62895ab4`.
  Source thật + manifest được lưu trong output `provenance/`; không Git push.

Theo dõi train_loss và norm target, không chọn round/μ theo test. Chỉ đưa kết quả r4 lên bảng chính Notion sau pull/audit đủ 50 round.
R3 giữ làm bằng chứng lịch sử trong repo, không xóa kết quả xấu. Trạng thái hiện tại: [bàn giao](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/operations/continue.md).

## Kết quả R4 hoàn tất (30/10/2026 ~23:35)

Kernel `anhtruong1909/fl2-tinyproto-r4`, version 1, hoàn tất đủ **50/50 round** mà không gặp lỗi.
Pull 141 file (gồm đầy đủ `provenance/`, không lấy `state*.pt`), 0 thiếu.
[Audit](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/audits/2026-10-30/fl2-tinyproto-r4.json): **PASS_BOUNDED** (5.343 xe, 9.347.679 dòng test support chính xác, 50 round CMs & snapshot model finite).
Overlay SHA256 khớp chuẩn `1bbf6b6e18f87a47f2ed52831c2709f8a3fbc001a7a2461c13a968ee62895ab4`.

| Metric | R3 (μ heuristic cũ) | R4 (μ = 1,5e-4 sửa) |
|---|---:|---:|
| Target L2 norm (r50) | 2.943,11 (bùng nổ) | **0,2657** (ổn định) |
| Train loss (r50) | 87,88 | **0,0218** |
| Prototype macro-F1 (headline) | 0,2904 | **0,8534** |
| Prototype accuracy | 0,5954 | **0,8866** |
| Head macro-F1 | 0,7881 | **0,8693** |
| Head accuracy | 0,8234 | **0,8994** |

Target L2 không còn bị khuếch đại, mô hình hội tụ ổn định, prototype F1 local hồi phục hoàn toàn từ 0,2904 lên 0,8534. Headline r4 được cập nhật vào [results.md](../research/results.md).
