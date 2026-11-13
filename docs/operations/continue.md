# Continue: bàn giao sống, 13/11/2026 10:00 VN

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md) → trang này.
Artifact, state, exact session/version ưu tiên hơn tóm tắt.

## Đã xác minh

- Code public GitHub `odixe06/fl_veremi_bert`; source train batch khóa **cb193e6fd56fc2f0c6810745cc57d8242c77c366**.
  [Parity v2](../research/v2-parity.md): 20 kiểm tra PASS, chỉ đổi model/phụ thuộc BERT; CANPerFL đúng code v2.
- Local smoke 13 method PASS (100 xe, compile, group_rows 2.048; 2–3 round); LwFedNIDS sửa SDPA head-dim 5.
  `python -m pytest -q tests`: **37 PASS**, log `runs/pytest_partial_batch.log`.
- Controller **flb-batch-s42-20261113-r3** succeeded 09:59:14 VN; marker `BATCHW_FIRST_ROUND`.
  Log `runs/batch-watch/controller-r3.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-r3/state.json`.
  Completion đã xử lý. Hai controller trước failed ở inventory/plan, không có training launch từ chúng.
- **13 kernel v1 đã push** (reply + `.intent` + `.pushed`); exact status 09:59:08 VN: cả 13 RUNNING.
  W&B đã có 13 run: config world2/client5343/compiled/HP đúng protocol; mới AFPHA có r1 lúc kiểm.
  Chưa pull/audit output, chưa hoàn tất 50 round.
- AFPHA r1 T4: train **573,432 s**, eval **56,981 s**. So v2 seed42 r1: **1,693× train**, **1,355× train+eval**.
  Đã ghi [LESSONS](../../LESSONS.md); bằng chứng `runs/batch-watch/first_round.json`, `initial-health.json`.

## Batch đang chạy

Plan bất biến `runs/batch/plan.json`; không replan, không push lại slug đã có intent. Deferred r3 = 0.
Owner thread **01a1d288-3551-7de3-9e0d-3b0b4f8b3401**; mọi controller delivery queue.

| method | exact kernel | budget phiên đầu (h) |
|---|---|---:|
| nilm | outlookcheck/flb-nilm/1 | 11.50 |
| apfed | outlookcheck/flb-apfed/1 | 11.50 |
| pfedes | anhtruong1909/flb-pfedes/1 | 10.49 |
| perfedskd | thuyanh316/flb-perfedskd/1 | 8.60 |
| v2x_htfl | chuchimnon/flb-v2x-htfl/1 | 8.53 |
| fedtgp | minhtriethihi/flb-fedtgp/1 | 8.31 |
| tinyproto | accanniez/flb-tinyproto/1 | 6.69 |
| fd_ids | heykewtiie/flb-fd-ids/1 | 6.60 |
| afpha | nguyetquedo/flb-afpha/1 | 5.83 |
| fedproto | khanhmay0304/flb-fedproto/1 | 5.44 |
| fedprox | minhtrit06/flb-fedprox/1 | 5.05 |
| fedavg | nguyenminhtran060113/flb-fedavg/1 | 4.73 |
| lwfednids | odixeuit/flb-lwfednids/1 | 2.45 |

**CANPerFL chưa launch**: collector tự dùng output FedAvg r50, fine-tune 10 round, cùng account nguyenminhtran060113.
Phần chưa đủ 50 round tự nối `-pN` khi quota >= 3h; reset thứ Bảy **14/11 00:00 UTC = 07:00 VN**.
Không đổi loss/optimizer/participation/HP; group_rows production 8.192 như v2.

## Watcher / bước tiếp

Watcher cũ đã kết thúc có chủ ý khi thấy r1. **Job hiện hành flb-batch-s42-20261113-watch1 RUNNING**,
supervisor PID **126921**; chỉ chạy `scripts/watch_batch.py`, không plan/launch lại.
Controller log `runs/batch-watch/controller-watch1.log`; durable state
`/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-watch1/state.json`.
Delivery **queue** (`auto-queue`) về owner thread ghi trên.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/batch-watch/controller-watch1.log`.
Nó giữ `runs/batch-watch/state.json` với `first_round` đã có, nên không đánh thức r1 lần nữa.
Log watcher `runs/batch-watch/watch.log`; poll **1.200 s**. Chỉ worker polling, không model polling.

1. Completion watcher → đọc state + log + artifact thực. ERROR: collect/chẩn đoán/sửa đúng bước, source/version mới;
   giữ kernel khỏe, không restart từ đầu nếu có checkpoint tương thích đã xác minh.
2. ALL_TERMINAL → audit → [results](../research/results.md): 10 metric từ CM, so v2; nêu giới hạn PASS_BOUNDED.
3. Cập nhật handoff hiện hành, kiểm link; không chạy lại plan/launch batch.

## Quyền / việc tay

Owner cho phép 14 baseline seed42, cấu hình v2; không SCAFFOLD/MobiHet/centralized.
Chừa **novahihi, keocaosu, emtrangtri, minhtran0601**; kiểm artifact xác nhận chưa dùng 4 account này, <=2 kernel/account.
Public code GitHub; không data/PDF/credential, không Co-Authored-By. Notebook PRIVATE có W&B key inline theo quyền đã cấp;
key ở version history private vĩnh viễn. Không đổi global credential, không hủy job khỏe, không xóa artifact. Chưa có việc tay.
