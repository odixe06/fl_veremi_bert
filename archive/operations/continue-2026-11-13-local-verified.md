> Snapshot 13/11/2026. Live handoff: [continue](../../docs/operations/continue.md).

# Continue: bàn giao sống, 13/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](../../docs/operations/lessons.md) → trang này.
Đây là **bàn giao hiện hành duy nhất**. Artifact, state và process thật được ưu tiên hơn tóm tắt ở đây.

## Đã xong (13/11)

- Repo dựng từ `../fl_dagsnet_veremi_v2`: `core/` + 13 method file (bỏ scaffold, mobihet), scripts, tests.
  Skill `.agents/` + `.claude/` + `AGENTS.md` chép từ `../mobihet`.
- FeatureBERT reference + engine grouped: khớp reference fp64. θ⁰ seed 42 trùng BERT centralized. 24 test pass (`python -m pytest -q tests`).
- Smoke dữ liệu thật (100 xe, 2–3 round, compile, group_rows 2.048): log ở `runs/sm_<method>.log` và `runs/smoke_all.log`.
- Chi phí đo: [cost](../../docs/research/cost.md).

## Trạng thái lúc bàn giao Claude Code → Codex (13/11 ~08:45 giờ VN)

- Chưa có commit nào. Repo GitHub chưa tạo. Chưa push kernel nào. `runs/batch/` chưa tồn tại.
- Smoke `runs/smoke_all.sh` được chạy nền từ phiên Claude và có thể đã bị dừng khi phiên đóng.
  Kết quả: fedprox, afpha, fd_ids, apfed, v2x_htfl, tinyproto **PASS**; **lwfednids FAIL**.
  Còn thiếu pfedes, nilm, fedtgp, perfedskd, fedproto: xem `runs/smoke_all.log`; nếu thiếu dòng của method nào thì chạy lại riêng method đó.
- **Lỗi LwFedNIDS** (`runs/sm_lwfednids.log`): compile gate PASS, nhưng sau đó inductor báo
  `LoweringException: ValueError: Exponent must be non-negative` ở SDPA `scale=0.4472` (head-dim 5 của bản tỉa).
  Cần sửa trong engine (vd. tự tính attention bằng matmul/softmax khi head-dim < 8, hoặc chỉ chạy eager cho đường attention đó),
  thêm test, rồi smoke lại. Không đổi cấu hình tỉa (owner đã chốt).
- FedAvg smoke 3 round OK (`runs/smoke_fedavg`).

## Đang chạy (Codex kiểm 13/11 ~08:40 VN)

- **Chưa launch Kaggle, chưa commit/push GitHub.** Mục batch đang chạy của bàn giao Claude là dự kiến, chưa có artifact xác nhận.
- Smoke Claude PID 90055 vẫn sống lúc tiếp quản; worker Codex chờ nó kết thúc, không hủy hay chạy cạnh GPU.
- Fix LwFedNIDS: SDPA eager riêng head-dim < 8; regression CUDA FULL → pruned tái hiện lỗi trước sửa,
  4 test grouped pass sau sửa (`runs/regression_lw_red.log`, `runs/regression_lw_green.log`).
- Job controller: `flb-handoff-verify-20261113`; supervisor PID **98324**; owner thread
  `01a1d288-3551-7de3-9e0d-3b0b4f8b3401`; delivery **queue** (`auto-queue`).
  Log: `runs/codex/verify_handoff.log`; state:
  `/home/odixe/.codex/long-jobs/jobs/flb-handoff-verify-20261113/state.json`.
  Worker `runs/codex/verify_handoff.py` chờ smoke cũ, chạy lại lwfednids + method thiếu/thất bại vào output mới,
  rồi pytest toàn bộ. Marker yêu cầu: `HANDOFF_VERIFY_PASS`. Test log: `runs/pytest_handoff.log`.
  Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/codex/verify_handoff.log`.
- Quota live đã đọc (23 account); chưa khóa plan. Trước launch kiểm lại quota + slot thực, chừa đúng 4 account B20.

## Bước tiếp

1. Completion job verify → đọc state/log + output smoke + pytest. PASS rồi commit main (không trailer), tạo/push repo public.
2. Đọc/sửa gate vận hành cần thiết của `scripts/run_batch.py`: quota + slot live, pin owner/slug/version,
   preflight notebook syntax/argv, resume đủ rank; `plan` rồi `launch` 13 method, CANPerFL nối FedAvg.
3. `scripts/watch_batch.py` chạy bằng codex-long-jobs, poll 1.200 s, completion về thread này. Ghi identity job vào đây.
4. Sau r1 của run đầu: đo `t_train` thật trên T4, ghi [LESSONS](../../LESSONS.md).
5. ERROR → collect/chẩn đoán/sửa/push phiên mới; ALL_TERMINAL → audit → cập nhật [results](../../docs/research/results.md)
   (10 metric từ CM), so v2. Quota reset thứ Bảy 14/11 00:00 UTC.

## Quyền / việc tay

Owner 13/11 cho phép: chạy seed 42 cho 14 method, đẩy ngay, chừa 4 account B20 (novahihi, keocaosu, emtrangtri, minhtran0601),
tạo repo public `odixe06/fl_veremi_bert` và push code. Không xóa artifact, không đổi global credential, không hủy job khỏe.
Commit không có trailer `Co-Authored-By` (quy tắc owner, [lessons](../../docs/operations/lessons.md)).
