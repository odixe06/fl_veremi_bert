# Continue: bàn giao sống, 13/11/2026 08:50 VN

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md) → trang này.
Artifact, state và process thật ưu tiên hơn tóm tắt.

## Đã xác minh

- Owner nhắc lại: chỉ đổi model BERT. [Đối chiếu v2](../research/v2-parity.md): 20 kiểm tra AST PASS; CANPerFL đã trả về đúng code v2.

- FeatureBERT FULL 77.904 tham số; grouped logits/CLS/gradient khớp reference fp64; θ⁰ seed 42 trùng centralized.
- LwFedNIDS giữ tỉa 70%: d 20, head-dim 5, FFN 39, 10.194 tham số. Fix SDPA eager riêng head-dim < 8;
  regression CUDA eval dynamic tái hiện lỗi trước sửa, PASS sau sửa.
- 13 method smoke dữ liệu thật PASS: 100 xe, compile, group_rows 2.048; FedAvg 3 round, còn lại 2 round.
  Log `runs/smoke_all.log`; LwFedNIDS sửa: `runs/sm_lwfednids_codex.log`. Đã kiểm CM + state đủ round, tensor hữu hạn.
- `python -m pytest -q tests`: **34 PASS** sau sửa inventory, log `runs/pytest_prelaunch_recovery.log` (32 PASS trước đó: `runs/pytest_prelaunch_final.log`) (26 PASS trước đó: `runs/pytest_handoff.log`).
- Job `flb-handoff-verify-20261113`: succeeded 08:47:20 VN, supervisor PID 98324 đã kết thúc.
  Log `runs/codex/verify_handoff.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-handoff-verify-20261113/state.json`.
  Owner thread `01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery queue đã nhận trong phiên này.
- Chi phí: [cost](../research/cost.md). Chưa có kết quả full 50 round.

## Đang chạy / bước tiếp

GitHub public đã tạo và push: `https://github.com/odixe06/fl_veremi_bert` (commit đầu `6c5105c`).
Không còn smoke local. Controller lần đầu **failed ở plan**, chưa tạo plan hay push kernel. Nguyên nhân: placeholder listing ref rỗng gây 403 giả;
đã sửa inventory + test. Worker cũng đã sửa sys.path trước import. Đang chuẩn bị retry controller bằng job ID mới.

- Job cũ **flb-batch-s42-20261113**: FAILED, supervisor PID **108083** đã kết thúc; owner thread
  `01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery **queue** (`auto-queue`).
- Controller log `runs/batch-watch/controller.log`; durable job state:
  `/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113/state.json`.
- Worker `runs/codex/launch_batch.py`; pinned source `f57bb8e7a6f1e285ed4e1527f274a1d982baf304` đã push GitHub.
- Plan khóa `runs/batch/plan.json`; phiên + version trong `.intent`; reply + marker `.pushed` mới xác nhận push.
- Watcher sau launch: `scripts/watch_batch.py`, log `runs/batch-watch/watch.log`, state `runs/batch-watch/state.json`;
  poll **1.200 s**. Dừng đánh thức ở r1 đầu quan sát được (`first_round.json`), lỗi, hoặc ALL_TERMINAL + audit.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/batch-watch/controller.log`.
- Lần đầu không có kernel train; controller cancel trả not-needed-already-terminal. Completion lỗi đã được xử lý trong phiên này.

1. Completion controller → đọc state/log + artifact thật. Nếu plan/launch lỗi, sửa đúng bước và giữ plan/intents; không push trùng.
   R1 → ghi timing, nối watcher mới; chỉ thêm bằng chứng đo, giữ code baseline đồng nhất v2.
   Notebook PRIVATE + W&B inline theo quyền đã cấp; key tồn tại vĩnh viễn trong version history private; không public notebook.
2. Kiểm quota + slot live; preflight notebook syntax/argv, resume đủ rank, pin owner/slug/version;
   `python scripts/run_batch.py plan` rồi `launch` (13 method; CANPerFL sau FedAvg r50).
3. `scripts/watch_batch.py` qua codex-long-jobs, poll 1.200 s, completion về thread này; ghi job identity vào đây.
4. Sau r1 đầu trên T4: ghi `t_train` thật vào [LESSONS](../../LESSONS.md).
5. ERROR → collect/chẩn đoán/sửa/push phiên mới. ALL_TERMINAL → audit → [results](../research/results.md)
   (10 metric từ CM), so v2. Quota reset thứ Bảy 14/11 00:00 UTC.

## Quyền / việc tay

Owner cho phép chạy 14 baseline seed 42, giữ cấu hình v2; không SCAFFOLD/MobiHet/centralized.
Chừa **novahihi, keocaosu, emtrangtri, minhtran0601** (B20). Tối đa 2 kernel GPU/account.
Public GitHub `odixe06/fl_veremi_bert`; không đẩy data/PDF/credential; không trailer Co-Authored-By.
Không đổi global credential, không hủy job khỏe, không xóa artifact. Chưa có việc tay.
