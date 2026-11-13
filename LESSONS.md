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
