"""Lightweight-Fed-NIDS — Bouayad et al., IEEE Access 2024 (papers/lwfednids.pdf), §III-B.

The server computes ONE data-free (zero-shot) structured pruning mask at initialisation and sends it with θ⁰ once; clients
train the pruned model, the server averages ("averaging all the clients' model weights", FedAvg). Pruning: per dependency
group of channels, importance = L1 norm of every parameter slice the channel touches (Torch-Pruning/DepGraph
GroupNormImportance with p = 1, which the paper uses), the lowest 70 % of each group removed (70 % = the paper's best
sparsity) and the pruned parameters physically dropped (DepGraph), so the model trained and exchanged is a narrower
FeatureBERT whose initial weights are the kept slices of θ⁰.
FeatureBERT dependency groups (owner 13/11): (1) the residual width d, shared by the token embedding (scale, identity, cls),
every layer's in_proj input columns, out_proj / linear2 output rows, both LayerNorms and the head input; (2) per layer, the
FFN width (linear1 rows, linear2 columns); (3) per layer and head, the head dimension (the q, k and v rows of one index
inside a head + the out_proj column of v; q and k must keep the same indices for the dot product, v shares them so the layer
stays an nn.TransformerEncoderLayer). Keep C − int(0.7 C): d 64 → 20, FFN 128 → 39, head dim 16 → 5 (4 heads kept, inner
4 × 5 = 20 = d).
Deviations: FeatureBERT instead of the paper's CNN feature extractor + classifier; uniform 70 % per group (paper: a global
rate); FedAvg sample-weighted, the protocol rule (owner 29/10; paper's footnote: plain averaging).
"""
import torch

from core.bert import FeatureBERT
from core.flat import Layout
from methods import fedavg

RATIO = 0.7


def _keep(C, ratio):
    return C - int(ratio * C)


def _top(score, n):
    return torch.sort(torch.topk(score, n).indices).values


def prune_plan(cfg, sd, ratio=RATIO):
    """-> (pruned cfg, pruned state_dict) for FeatureBERT cfg with initial state_dict sd."""
    W = {k: v.abs() for k, v in sd.items() if v.is_floating_point()}
    d, H, f, L = cfg["d"], cfg["heads"], cfg["ffn"], cfg["layers"]
    dh = d // H
    bl = lambda i, k: W[f"blocks.{i}.{k}"]
    # (1) residual width
    sc = W["scale"].sum(0) + W["identity"].sum(0) + W["cls"].reshape(-1) + W["head.weight"].sum(0)
    for i in range(L):
        sc = sc + bl(i, "self_attn.in_proj_weight").sum(0) + bl(i, "self_attn.out_proj.weight").sum(1) \
            + bl(i, "self_attn.out_proj.bias") + bl(i, "linear1.weight").sum(0) + bl(i, "linear2.weight").sum(1) \
            + bl(i, "linear2.bias") + sum(bl(i, f"norm{j}.{t}") for j in (1, 2) for t in ("weight", "bias"))
    kd = _top(sc, _keep(d, ratio))
    kf, kh = [], []
    for i in range(L):
        # (2) FFN width
        kf.append(_top(bl(i, "linear1.weight").sum(1) + bl(i, "linear1.bias") + bl(i, "linear2.weight").sum(0),
                       _keep(f, ratio)))
        # (3) head dimension, per head: q, k, v rows of index j (+ biases) and out_proj column of v
        w, b, o = bl(i, "self_attn.in_proj_weight"), bl(i, "self_attn.in_proj_bias"), bl(i, "self_attn.out_proj.weight")
        s = sum(w[q * d:(q + 1) * d].sum(1) + b[q * d:(q + 1) * d] for q in range(3)) + o.sum(0)      # (d,) inner index
        kh.append(torch.cat([h * dh + _top(s[h * dh:(h + 1) * dh], _keep(dh, ratio)) for h in range(H)]))
    new = dict(cfg, d=len(kd), ffn=_keep(f, ratio), head_hidden=len(kd))
    assert len(kh[0]) == new["d"] and new["d"] % H == 0, "pruned inner width must equal the residual width"
    out = {"scale": sd["scale"][:, kd].clone(), "identity": sd["identity"][:, kd].clone(), "cls": sd["cls"][..., kd].clone()}
    for i in range(L):
        p = f"blocks.{i}."
        qkv = torch.cat([q * d + kh[i] for q in range(3)])
        out[p + "self_attn.in_proj_weight"] = sd[p + "self_attn.in_proj_weight"][qkv][:, kd].clone()
        out[p + "self_attn.in_proj_bias"] = sd[p + "self_attn.in_proj_bias"][qkv].clone()
        out[p + "self_attn.out_proj.weight"] = sd[p + "self_attn.out_proj.weight"][kd][:, kh[i]].clone()
        out[p + "self_attn.out_proj.bias"] = sd[p + "self_attn.out_proj.bias"][kd].clone()
        out[p + "linear1.weight"] = sd[p + "linear1.weight"][kf[i]][:, kd].clone()
        out[p + "linear1.bias"] = sd[p + "linear1.bias"][kf[i]].clone()
        out[p + "linear2.weight"] = sd[p + "linear2.weight"][kd][:, kf[i]].clone()
        out[p + "linear2.bias"] = sd[p + "linear2.bias"][kd].clone()
        for j in (1, 2):
            for t in ("weight", "bias"):
                out[f"{p}norm{j}.{t}"] = sd[f"{p}norm{j}.{t}"][kd].clone()
    out["head.weight"], out["head.bias"] = sd["head.weight"][:, kd].clone(), sd["head.bias"].clone()
    return new, out


class Method(fedavg.Method):
    name = "lwfednids"

    def __init__(self, env):
        torch.manual_seed(env.hp["seed"])
        full = FeatureBERT(env.cfg)
        cfg, sd = prune_plan(env.cfg, full.state_dict(), env.hp.get("prune", RATIO))
        m = FeatureBERT(cfg)
        m.load_state_dict(sd)                                         # the pruned model is a narrower FeatureBERT
        env.cfg, env.lay = cfg, Layout(cfg)                           # every client trains / exchanges the pruned model
        self.env = env
        self.p = torch.cat([sd[k].reshape(-1).float() for k, _ in env.lay.p]).to(env.dev)
        self.b = torch.zeros(env.lay.B, device=env.dev)                # FeatureBERT: no BN buffers
        self.full_params = sum(p.numel() for p in full.parameters())
