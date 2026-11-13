# LESSONS: bài học kỹ thuật (dữ liệu, model, train, đánh giá, code)

Đọc sau [CONTEXT](CONTEXT.md), cùng [bài học vận hành](docs/operations/lessons.md). Bài học kế thừa, grep theo chủ đề:

- `../fl_dagsnet_veremi_v2/LESSONS.md`: engine grouped, compile, recompile do stride, luật suy luận prototype, TinyProto μ, audit PASS_BOUNDED, resume RNG;
- `../mobihet/LESSONS.md`: FL/KD/đánh giá.

## Kế thừa từ v2, áp dụng thẳng

- `torch.compile` + AMP fp16 có thể làm hỏng gradient ở đường compile mà eager không bị. Vì vậy phải giữ `local.compile_gate`:
  chạy một bước thật, so gradient compile với eager, FAIL thì quay về eager. Kèm cảnh báo `skipped_frac` mỗi round.
- Slice tensor đầu vào phải `.contiguous()` trước graph tĩnh. Stride là guard; recompile quá giới hạn thì dynamo im lặng chạy eager.
- Luật suy luận prototype phải theo đúng bài: FedProto/FedTGP dùng prototype toàn cục; TinyProto Eq. 12 dùng prototype cục bộ dense.
  View prototype thấp xa view head là dấu hiệu dùng sai luật.
- Audit CM/metric + snapshot hữu hạn chỉ đạt **PASS_BOUNDED**, không phải "audit model PASS".

## FeatureBERT (13/11)

- θ⁰ seed 42 của `core/bert.FeatureBERT` trùng từng tham số với `FeatureBERT()` của repo centralized.
  Engine grouped khớp reference ở fp64: logit, CLS và gradient lệch < 1e-14 ([tests/test_grouped.py](tests/test_grouped.py)).
- Model không có buffer, nên `Layout.B = 0`. Mọi `torch.cat` trên danh sách buffer rỗng sẽ lỗi. Các chỗ đã gặp và đã sửa:
  `compile_gate` (`lay.flat_bufs({})`) và LwFedNIDS `self.b`. Code mới phải kiểm `lay.B` trước khi nối buffer.
- Bản tỉa LwFedNIDS cần inner attention = d để vẫn là `nn.TransformerEncoderLayer`. Với d 64 / head-dim 16 / tỉa 70 %,
  ra đúng 20 = 4 × 5. Cấu hình toy nhỏ (d 16) thì không thỏa (5 ≠ 8), nên test method dùng FULL.
- Tốc độ local (RTX 3050, compile):
  - synthetic 32 client × 64 dòng/bước: BERT 44,5 ms/bước, DAGSNet 21,7 ms;
  - dữ liệu thật 100 xe, group_rows 2.048: BERT 45 s/round, DAGSNet 54 s.
  Tỉ lệ thật trên T4 phải đo từ `t_train` của run đầu, không suy từ FLOPs.
- Eager và compile gate ở group_rows 8.192 OOM trên GPU 4 GiB. Local dùng `--set group_rows=2048`; T4 (15 GB) giữ 8.192.
- LwFedNIDS head-dim 5: Inductor eval dynamic lỗi `Exponent must be non-negative` trong flash SDPA padding.
  Regression FULL → pruned tái hiện trong 17 s (`runs/regression_lw_red.log`). Giữ SDPA eager riêng head-dim < 8
  bằng `torch.compiler.disable`; phần còn lại vẫn compile. Test CUDA và logits/gradient fp64 so reference đều pass
  (`runs/regression_lw_green.log`); không đổi scale, dropout hay cấu hình tỉa.

- CANPerFL giữ nguyên `state()` của v2 (chỉ bank cá nhân). Ở full 5.343 client, snapshot nhỏ lọc bank > 10M phần tử
  nên rỗng; audit phải ghi rõ thiếu bằng chứng weight thay vì sửa method để phục vụ checker. Không suy finite weights từ CM.

## T4 thật — round đầu batch seed 42 (13/11)

- `nguyetquedo/flb-afpha/1`, source `cb193e6`, 2×T4, 5.343 client, batch 64, group_rows 8.192, compiled.
  W&B `21522798-uit/fl-veremi-bert/flb-afpha`, r1 lúc **13/11/2026 09:48:22 VN**:
  **t_train = 573.432 s**, **t_eval = 56.981 s**, tổng **630.413 s**.
  Bằng chứng live: `runs/batch-watch/first_round.json`, `runs/batch-watch/initial-health.json`.
- V2 AFPHA seed42 r1 (`../fl_dagsnet_veremi_v2/runs/pulls/fl2-afpha-r2/fl2-afpha-r2/rounds.jsonl`):
  train 338,735 s, eval 126,393 s ⇒ BERT **1,693× train**, **1,355× train+eval** ở round đầu.
  Round đầu có compile/warmup; không so với trung bình ổn định v2 (r2–50: train 258,200 s) để chốt hệ số.
  FACTOR 1,5 của plan là ước lượng trước launch, không phải số đo chung mọi method; plan đã khóa, không sửa từ một điểm warmup.
- 13 W&B config đã kiểm: world 2, clients 5.343, backend compiled, HP đúng protocol; TinyProto μ 1,5e-4;
  LwFedNIDS d20/FFN39/4 head, 10.194 tham số. Khi kiểm 09:59 VN chỉ AFPHA có metric r1; các run khác chưa có round metric.
  Đây là quan sát live, chưa thay final artifact audit/CM.
