# FedAvg, FedProx, SCAFFOLD — model toàn cục

> Kế thừa từ `fl_dagsnet_veremi_v2` (DAGSNet). Cơ chế và siêu tham số giữ nguyên khi chạy trên FeatureBERT. Số liệu, SHA và run bên dưới là lịch sử của v2,
> không phải trạng thái của repo này. Repo này bỏ SCAFFOLD (owner 13/11).

Cấu hình chung: [protocol](../research/protocol.md). Code: [fedavg.py](../../methods/fedavg.py), [fedprox.py](../../methods/fedprox.py),
[scaffold.py](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/methods/scaffold.py). Model chấm: θ toàn cục sau round, trên toàn bộ test.

| | bài (papers/) | cơ chế | siêu tham số riêng | deviation |
|---|---|---|---|---|
| FedAvg | McMahan 2017, Alg. 1 | θ^{t+1} = Σ (n_k/n) θ_k | — | client update dùng AdamW (bài: SGD); C = 1 (mọi client), E = 1, B = 64 |
| FedProx | Li 2020, Alg. 2 | + (μ/2)‖w − w^t‖² vào loss local ⇒ gradient + μ(w − w^t) | μ = 0,01 (tập ứng viên {0,001; 0,01; 0,1; 1} của bài, cùng giá trị build B) | không chọn μ trên test; không có straggler (mọi client chạy đủ 1 epoch) |
| SCAFFOLD | Karimireddy 2020, Alg. 1, Option II | gradient g − c_i + c; c_i⁺ = trung bình gradient thô trong epoch (= Option II dưới SGD); x ← x + Σ(n_i/n)Δy; c ← c + Σ(n_i/n)Δc | η_g = 1 (bài dùng 1 trong mọi thí nghiệm) | AdamW thay SGD ⇒ c_i⁺ tính trực tiếp bằng trung bình gradient (công thức (x − y)/(Kη) đo bước Adam, smoke F1 0,16); gộp theo n_k (protocol, owner 29/10) |

Chung: BN running stats gộp cùng trọng số với tham số; byte/client/round = 4 B × (P + B_bn) mỗi chiều (SCAFFOLD gấp đôi phần P).

So với build cũ (B: `dangvanhoang1711/FL_DAGSNet`): cùng công thức; khác dataset/partition (Dirichlet 20/50/100 → 5.343 xe tự nhiên),
batch 512 → 64, chấm cùng toàn test.
