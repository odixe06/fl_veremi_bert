# Chi phí tính toán và băng thông: FeatureBERT

Số liệu đo bằng [scripts/cost.py](../../scripts/cost.py) và ghi ra [cost.json](cost.json) ngày 13/11/2026, không ước tính tay. Cách đo giống v2:

- chạy một round thật (round 2) trên dữ liệu toy CPU dưới `FlopCounterMode` (matmul/conv/bmm/SDPA);
- client FLOPs/dòng = FLOPs của round trừ bước server, rồi chia số dòng train;
- byte = `payload_bytes()` của method, fp32.

Thời gian GPU thật lấy từ `t_train`/`t_eval` trong `rounds.jsonl`.

F = forward FeatureBERT = **10,23 MFLOP/dòng** (DAGSNet W1: 4,45). P = 77.904 (DAGSNet 395.408 + 3.295 buffer).

| phương pháp | client / dòng / round | × F | byte lên / xuống / xe / round | P |
|---|---:|---:|---|---:|
| fedavg | 40,0 MFLOP | 3,91 | 311.616 B / 311.616 B | 77.904 |
| fedprox | 40,0 MFLOP | 3,91 | 311.616 B / 311.616 B | 77.904 |
| afpha | 40,0 MFLOP | 3,91 | 311.616 B / 311.616 B | 77.904 |
| fd_ids | 50,2 MFLOP | 4,91 | 311.616 B / 311.616 B | 77.904 |
| lwfednids | 5,9 MFLOP | 0,58 | 40.776 B / 40.776 B | 10.194 |
| pfedes | 100,5 MFLOP | 9,83 | 1.404 B / 1.404 B | 77.904 |
| fedproto | 40,0 MFLOP | 3,91 | 4.160 B / 4.096 B | 77.904 |
| fedtgp | 50,2 MFLOP | 4,91 | 4.096 B / 4.096 B | 77.904 |
| tinyproto | 50,2 MFLOP | 4,91 | 448 B / 448 B | 77.904 |
| nilm | 80,0 MFLOP | 7,83 | 311.616 B / 311.616 B | 77.904 |
| perfedskd | 60,5 MFLOP | 5,91 | 311.616 B / 311.616 B | 77.904 |
| apfed | 80,0 MFLOP | 7,83 | 311.616 B / 311.616 B | 77.904 |
| v2x_htfl | 50,3 MFLOP | 4,92 | 4.160 B / 4.096 B | 82.128 |

CANPerFL: pha 1 = FedAvg × 50 round; pha 2 = 10 epoch fine-tune local × FedAvg/dòng, byte 0.
PerFed-SKD: byte chỉ tính cho xe được chọn (`selected_frac` mỗi round). Server FLOPs (FedTGP, V2X-HtFL) nằm ở `server_flops_per_round_toy`.
Tổng băng thông của run = byte/xe/round × K × 50.
