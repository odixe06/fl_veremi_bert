# FeatureBERT cho v4 (77 đầu vào): đặc tả model client

Nguồn: repo [`dangvanhoang1711/veremi-bert-77`](https://github.com/dangvanhoang1711/veremi-bert-77), `core/central_backbones.py`
(`FeatureBERT`). Đây là bản centralized được tách từ `fl_dagsnet_veremi_v2`. Code tham chiếu ở repo này là [core/bert.py](../core/bert.py)
(`FeatureBERT`, `FULL`, `cfg_full`). Bản chạy G model song song là [core/grouped.py](../core/grouped.py). Hai bản dùng cùng tên `state_dict`.
Tính tương đương được kiểm bằng [tests/test_grouped.py](../tests/test_grouped.py).

## 1. Hợp đồng vào/ra

```
x      : (B, 77) float, đúng thứ tự cột của knowledge/input_order.json, đã chuẩn hóa sẵn (không chuẩn hóa lại)
token  : x_j · scale_j + identity_j  (scale, identity ∈ R^{77×d}), một token cho mỗi cột
seq    : [CLS] + 77 token  → (B, 78, d)
logits : head(CLS output) → (B, 16), chưa qua softmax; loss = CrossEntropy; lớp theo knowledge/label_mapping.json
```

## 2. Kiến trúc

- `L = 2` lớp `nn.TransformerEncoderLayer(d=64, heads=4, FFN=128, dropout=0.1, GELU, batch_first, norm_first=False)`.
  Đây là post-LN: `x = LN1(x + Drop(SA(x)))`, rồi `x = LN2(x + Drop(W2 Drop(GELU(W1 x))))`. Attention cũng có dropout 0,1 (MHA).
- `head = Linear(64, 16)` đặt trên vị trí CLS. Model có **77.904 tham số** và **không có buffer** (không BatchNorm).
- Init: `torch.manual_seed(seed)`, rồi dựng module. scale, identity và cls dùng N(0; 0,02). Các lớp encoder và head dùng init mặc định của PyTorch.
  Với seed 42, θ⁰ trùng từng tham số với `FeatureBERT()` của repo centralized (đã kiểm 13/11).
- Thứ tự tham số: `scale, identity, cls, blocks.0.*, blocks.1.*, head.weight, head.bias`. Classifier nằm cuối.

## 3. Engine grouped

Mỗi Linear là một `bmm` trên (G, b·78, ·), dùng trọng số riêng của từng client. Attention dùng
`F.scaled_dot_product_attention` trên (G·b, H, 78, d/H). Các dòng không trộn nhau, nên dòng đệm không ảnh hưởng dòng thật.
LayerNorm tính ở fp32. AMP dùng fp16 (T4). `forward` có cùng chữ ký với engine DAGSNet của v2 (`P, B, x, cfg, mask, train, hidden`).
Vì vậy train loop và method không phải sửa. `B` luôn rỗng.

## 4. Biểu diễn cho phương pháp prototype / representation

`hidden(x)` là đầu ra CLS sau lớp encoder cuối (d = 64, sau LayerNorm). `grouped.forward(..., hidden=True)` trả cùng tensor đó.
Extractor gồm mọi tham số trừ `head`; classifier là `head` (`Layout.mask(["head."])`). `cfg["head_hidden"] = d`.

## 5. Bản tỉa của LwFedNIDS

Owner chốt 13/11: tỉa 70 % mỗi nhóm phụ thuộc, giữ `C − int(0,7 C)` phần tử:

- residual d: 64 → 20;
- FFN: 128 → 39;
- head-dim: 16 → 5, vẫn giữ 4 head (inner 20 = d).

Sau tỉa model còn **10.194 tham số** (13,1 %) và vẫn là `nn.TransformerEncoderLayer` chuẩn. Chi tiết ở docstring `methods/lwfednids.py`.

## 6. Khác với DAGSNet (v2)

| | DAGSNet W1 (v2) | FeatureBERT (repo này) |
|---|---|---|
| tham số / buffer | 395.408 / 3.295 BN | 77.904 / 0 |
| representation | hidden ReLU 256 | CLS 64 |
| classifier | `head.5` | `head` |
| gộp BN | trung bình theo n_k | không có |
| cổng độ rộng | đã chạy, khóa W = 1 | không có, dùng FULL (owner 13/11) |
