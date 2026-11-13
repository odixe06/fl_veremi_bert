# Protocol chung: mọi phương pháp chạy đúng cấu hình này

Owner chốt 13/11/2026: **giữ nguyên cấu hình của [`fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/protocol.md)**
(protocol chốt 29/10) và chỉ đổi model client từ DAGSNet sang **FeatureBERT** ([knowledge/bert](../../knowledge/bert.md)).
Một phương pháp chỉ được khác ở **cơ chế của chính nó**: loss, thứ được trao đổi, cách gộp, và siêu tham số riêng lấy từ bài gốc.
Mọi thứ dưới đây dùng chung một bộ code (`core/`), không sửa riêng cho phương pháp nào. Đổi bất kỳ dòng nào thì phải chạy lại mọi phương pháp.

## Dữ liệu và client

| | |
|---|---|
| Dataset | v4 `odixe0502/veremi-nextgen-v4-scaled-per-vehicle` ([knowledge/dataset](../../knowledge/dataset.md)); 77 đầu vào theo [input_order.json](../../knowledge/input_order.json); không chuẩn hóa lại |
| Client | **5.343 xe nhận**, partition tự nhiên của dataset (không Dirichlet) |
| Train local | **toàn bộ** dữ liệu train của client = fold `train` + `val` (37.361.273 dòng). Không có tập chọn mô hình; headline = round cuối |
| Test | 9.347.679 dòng (20 % cuối timeline của từng xe) |
| Tham gia | **mọi client mỗi round**. Ngoại lệ: khi chọn client là cơ chế của bài gốc (vd PerFed-SKD), giữ đúng cơ chế đó và ghi rõ |

## Train

| | |
|---|---|
| Round | **50**; mỗi round, mỗi client chạy **1 epoch local** trên toàn bộ dữ liệu của nó |
| Model | **FeatureBERT FULL** (77.904 tham số, d 64, 4 head, FFN 128, 2 lớp). Mọi phương pháp và mọi client dùng cùng model. Không có cổng độ rộng |
| Batch local | **64**. Client có ít hơn 64 dòng chạy 1 bước với toàn bộ dữ liệu |
| Optimizer | AdamW, β (0,9; 0,999), ε 1e-8, weight decay 1e-4. Trạng thái optimizer khởi tạo lại mỗi round |
| LR | cosine theo round: 1e-3 ở r1 → 1e-5 ở r50 (cố định trong một round) |
| Khác | grad-norm clip 1,0 theo từng client; AMP fp16 (T4); dropout 0,1 (FeatureBERT: dropout ở residual, FFN và attention) |
| Seed | **42** (owner 13/11: bản đầu chỉ chạy seed 42) |
| Init | mọi model khởi tạo từ cùng θ⁰ (seed 42), trùng θ⁰ của BERT centralized |

**Gộp model:** mọi trung bình tham số giữa client tính theo số mẫu n_k (owner 29/10). FeatureBERT không có BN buffer.
Gộp prototype giữ đúng cơ chế của bài (TinyProto APS, FedTGP Q^c, FedProto Eq. 6).

Siêu tham số riêng của phương pháp (μ FedProx, nhiệt độ KD, λ prototype, …) lấy theo thứ tự ưu tiên:
1. **giá trị trong bài gốc**;
2. nếu bài không nêu, giá trị mặc định trong code chính thức;
3. nếu không có code chính thức, tự chọn và ghi là deviation trong docstring `methods/<method>.py`.

Giá trị phải giống v2, kể cả TinyProto μ = 1,5e-4. Không quét siêu tham số trên test.

## Phạm vi (owner 13/11)

- **14 phương pháp FL** của v2: FedAvg, FedProx, AFPHA, FD-IDS, APFed, LwFedNIDS, V2X-HtFL, CANPerFL, TinyProto, pFedES, NILM-FL, FedTGP,
  PerFed-SKD, FedProto ([danh mục](methods.md)).
- **Bỏ SCAFFOLD.** Không chạy MobiHet / CBB. Không chạy centralized (BERT centralized ep1–18 nằm ở repo veremi-bert-77, chỉ để tham chiếu).

## Đánh giá (mỗi round)

- **Model toàn cục** (kiểu FedAvg): chấm trên **toàn bộ** test.
- **Model riêng từng client** (personalized): model của client k chấm trên **test của chính xe k**; gộp 5.343 phần đúng bằng toàn bộ test.
- Báo **đủ 10 metric** trên ma trận nhầm lẫn **gộp**: accuracy; precision/recall/F1 macro, micro, weighted. Kèm F1 từng lớp.
  Kèm phân phối F1-macro theo client (trung bình, p10, trung vị, min).
- Headline: **round 50**, không chọn round theo test.
- Chi phí: byte truyền mỗi client mỗi round (lên + xuống, fp32), FLOPs/dòng ([scripts/cost.py](../../scripts/cost.py)), GPU-giây/round.

## Hạ tầng (không ảnh hưởng kết quả)

- `core/grouped.py` chạy G model client trong một bước (bmm). Bản này tương đương reference ở fp64 ([tests/test_grouped.py](../../tests/test_grouped.py)).
  Client được nhóm theo kích thước (`core/local.py`); `group_rows` 8.192.
- 2×T4: mỗi GPU giữ nửa số client (xen kẽ theo kích thước) cùng dữ liệu của chúng. Process group chỉ dùng để cộng payload gộp và ma trận nhầm.
- Kaggle image đã pin ([runtime.json](../../knowledge/runtime.json)), torch.compile có cổng so eager (`local.compile_gate`).
