"""Grouped forward == reference FeatureBERT, per client: logits, CLS representation and gradients (fp64, dropout off);
seed-42 initialisation == the centralized repo's FeatureBERT (veremi-bert-77) parameter for parameter."""
import torch
import pytest

from core import grouped
from core.bert import FeatureBERT, cfg_full
from core.flat import Layout


def _ref(cfg, lay, flat, bufs):
    ms = []
    for g in range(flat.shape[0]):
        m = FeatureBERT(cfg).double()
        m.load_state_dict(lay.state_dict(flat[g], bufs[g]))
        ms.append(m)
    return ms


def _check(cfg, G=3, n=5):
    torch.backends.mha.set_fastpath_enabled(False)
    lay = Layout(cfg)
    flat, bufs = lay.init(42, G)
    flat = flat.double() + 0.05 * torch.randn(flat.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    x = torch.randn(G, n, 77, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    y = torch.randint(0, 16, (G, n), generator=torch.Generator().manual_seed(3))
    ms = _ref(cfg, lay, flat, bufs)
    for m in ms:
        m.eval()
    logits, h, nb = grouped.forward(lay.params(flat), {}, x, cfg, hidden=True)
    assert nb == {}
    assert (logits - torch.stack([m(x[g]) for g, m in enumerate(ms)])).abs().max() < 1e-10
    assert (h - torch.stack([m.hidden(x[g]) for g, m in enumerate(ms)])).abs().max() < 1e-10
    cfg0 = dict(cfg, dropout=0.0)
    w = flat.clone().requires_grad_(True)
    z, _ = grouped.forward(lay.params(w), {}, x, cfg0, mask=torch.ones(G, n, dtype=torch.bool), train=True)
    torch.nn.functional.cross_entropy(z.reshape(-1, 16), y.reshape(-1), reduction="sum").backward()
    ref = []
    for g, m in enumerate(_ref(cfg0, lay, flat, bufs)):
        m.train()
        torch.nn.functional.cross_entropy(m(x[g]), y[g], reduction="sum").backward()
        ref.append(torch.cat([p.grad.reshape(-1) for _, p in m.named_parameters()]))
    assert (w.grad - torch.stack(ref)).abs().max() < 1e-10


def test_grouped_matches_reference_full():
    _check(cfg_full())


def test_grouped_matches_reference_pruned():
    _check(cfg_full(d=20, ffn=39))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Inductor regression")
def test_compiled_eval_full_then_pruned():
    """LwFedNIDS evaluates full and head-dim-5 models through the same dynamic graph."""
    compiled = torch.compile(grouped.forward, dynamic=True)
    for cfg in (cfg_full(), cfg_full(d=20, ffn=39)):
        lay = Layout(cfg)
        flat, _ = lay.init(42, 1, device="cuda")
        x = torch.randn(1, 8, 77, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            expected, _ = grouped.forward(lay.params(flat), {}, x, cfg)
            actual, _ = compiled(lay.params(flat), {}, x, cfg)
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_full_model_is_the_centralized_featurebert():
    cfg = cfg_full()
    assert sum(p.numel() for p in FeatureBERT(cfg).parameters()) == 77904
    torch.manual_seed(42)
    sd = FeatureBERT(cfg).state_dict()
    assert list(sd)[:3] == ["scale", "identity", "cls"] and list(sd)[-2:] == ["head.weight", "head.bias"]
