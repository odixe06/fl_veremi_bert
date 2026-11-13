> Snapshot 13/11/2026. Live handoff: [continue](../../docs/operations/continue.md).

# Continue: bàn giao sống, 13/11/2026 09:23 VN

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](../../docs/operations/lessons.md) → trang này.
Artifact, state và process thật ưu tiên hơn tóm tắt.

## Đã xác minh

- Owner nhắc lại: chỉ đổi model BERT. [Đối chiếu v2](../../docs/research/v2-parity.md): 20 kiểm tra AST PASS; CANPerFL đã trả về đúng code v2.

- FeatureBERT FULL 77.904 tham số; grouped logits/CLS/gradient khớp reference fp64; θ⁰ seed 42 trùng centralized.
- LwFedNIDS giữ tỉa 70%: d 20, head-dim 5, FFN 39, 10.194 tham số. Fix SDPA eager riêng head-dim < 8;
  regression CUDA eval dynamic tái hiện lỗi trước sửa, PASS sau sửa.
- 13 method smoke dữ liệu thật PASS: 100 xe, compile, group_rows 2.048; FedAvg 3 round, còn lại 2 round.
  Log `runs/smoke_all.log`; LwFedNIDS sửa: `runs/sm_lwfednids_codex.log`. Đã kiểm CM + state đủ round, tensor hữu hạn.
- `python -m pytest -q tests`: **37 PASS**, log `runs/pytest_partial_batch.log` (test trước đó: `runs/pytest_prelaunch_recovery.log`, `runs/pytest_handoff.log`).
- Job `flb-handoff-verify-20261113`: succeeded 08:47:20 VN, supervisor PID 98324 đã kết thúc.
  Log `runs/codex/verify_handoff.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-handoff-verify-20261113/state.json`.
  Owner thread `01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery queue đã nhận trong phiên này.
- Chi phí: [cost](../../docs/research/cost.md). Chưa có kết quả full 50 round.

## Đang chạy / bước tiếp

GitHub public `https://github.com/odixe06/fl_veremi_bert`. Không còn smoke local; chưa push kernel Kaggle.

- Controller `flb-batch-s42-20261113` FAILED ở plan (placeholder ref rỗng → 403); đã sửa.
- Controller retry `flb-batch-s42-20261113-r2` FAILED ở plan 09:13:38 VN (429/404 trên status thật → chưa đủ account đọc được).
  Log `runs/batch-watch/controller-r2.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-r2/state.json`.
  Cả hai completion đã xử lý, không restart từ thông báo cũ. Chưa có `runs/batch/plan.json` hoặc `.pushed` lúc nhận lỗi r2.
- Fix mới: status tuần tự, 2 account survey song song, bounded retry read HTTP429; vẫn giữ UNKNOWN khi thất bại.
  Reprobe minhtriethihi thành công (9,56h, busy0). Plan chạy phần đã xác minh, giữ phần thiếu trong `deferred`;
  collector thử cấp account cho deferred mỗi 1.200 s. Không đổi code train/method.
- **Job hiện hành `flb-batch-s42-20261113-r3`: RUNNING**, supervisor PID **114634**.
  Worker `runs/codex/launch_batch.py`, source **cb193e6fd56fc2f0c6810745cc57d8242c77c366** đã push GitHub;
  import preflight PASS. Controller chạy plan → launch phần đã xác minh → watcher; chưa xác nhận kernel push.
  Log `runs/batch-watch/controller-r3.log`; durable state
  `/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-r3/state.json`.
  Owner thread `01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery **queue** (`auto-queue`).
  Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/batch-watch/controller-r3.log`.
  Không tạo controller khác, không đổi HEAD khi plan chưa khóa. Handoff có job identity mới chưa commit.
  Completion r1/r2 đã xử lý; nhận thông báo cũ không launch trùng.

1. Khóa source mới, `plan` → `launch` → watcher qua codex-long-jobs; không ghi đè plan hay intent đã có.
2. R1 đầu quan sát được → ghi timing vào [LESSONS](../../LESSONS.md), nối watcher mới. Giữ baseline đồng nhất v2.
3. ERROR → collect/chẩn đoán/sửa/push phiên mới. ALL_TERMINAL → audit → [results](../../docs/research/results.md)
   (10 metric từ CM), so v2. Quota reset thứ Bảy 14/11 00:00 UTC.

Watcher `scripts/watch_batch.py`: log `runs/batch-watch/watch.log`, state `runs/batch-watch/state.json`, poll **1.200 s**.
Completion khi có `first_round.json`, lỗi collect/train/audit, hoặc ALL_TERMINAL. Owner thread:
`01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery queue. Notebook PRIVATE + W&B inline theo quyền đã cấp;
key tồn tại vĩnh viễn trong version history private; không public notebook.

## Quyền / việc tay

Owner cho phép chạy 14 baseline seed 42, giữ cấu hình v2; không SCAFFOLD/MobiHet/centralized.
Chừa **novahihi, keocaosu, emtrangtri, minhtran0601** (B20). Tối đa 2 kernel GPU/account.
Public GitHub `odixe06/fl_veremi_bert`; không đẩy data/PDF/credential; không trailer Co-Authored-By.
Không đổi global credential, không hủy job khỏe, không xóa artifact. Chưa có việc tay.
