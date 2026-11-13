# Đối chiếu code với fl_dagsnet_veremi_v2 — 13/11/2026

Owner nhắc lại: chỉ đổi model client sang FeatureBERT. Đã kiểm working tree hiện tại của `../fl_dagsnet_veremi_v2`
(HEAD `36ac7ffafcb45974ed736b4d838279728d00e884`; file ngoài HEAD có thể đã sửa). Không sửa repo v2.

`python scripts/check_v2_parity.py`: **20 kiểm tra PASS**. Hash và phạm vi từng kiểm tra: [v2-parity.json](v2-parity.json).
AST bỏ docstring và thông báo assert; giữ điều kiện assert và toàn bộ phép toán/control flow.

- Data, FL environment/participation/aggregation, optimizer, metrics, evaluator: AST giống v2.
- Method không phụ thuộc kiến trúc: AST giống v2, gồm CANPerFL (đã trả persistence về đúng v2).
- APFed: chỉ đường dẫn classifier `head.5` → `head.`. V2X-HtFL: chỉ base class/import DAGSNet → FeatureBERT.
- pFedES: class Method giống v2; proxy giữ nguyên lưới 7×11, Conv1d 7→8→7. Helper grouped conv tách khỏi engine DAGSNet,
  đã so reference ProxyExtractor bằng test unit.
- Train driver: AST giống v2 sau thay model factory `width(a.width)` → `cfg_full()` và bỏ tham số width DAGSNet.
  Protocol, loss, local epoch, batch, seed, LR, AMP, compile gate, train/eval/save/resume loop không đổi.

Khác biệt bắt buộc do model: `core/bert.py`, `core/grouped.py`, default model trong Layout, compile gate hỗ trợ buffer rỗng;
LwFedNIDS tỉa nhóm d/head-dim/FFN 70% theo quyết định owner. Grouped BERT logits/CLS/gradient so reference fp64 và
CUDA regression head-dim 5 đều PASS. Những khác biệt này không phải thay thuật toán baseline.

Hạ tầng riêng repo: GitHub/W&B project, phân account/quota/slot, notebook/collect/watch và báo cáo/audit.
CANPerFL giữ snapshot bank-only của v2; bank lớn bị lọc khỏi snapshot nhỏ, nên audit ghi rõ không có bằng chứng hữu hạn của weight đó.
