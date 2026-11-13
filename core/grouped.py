"""G independent FeatureBERTs in one forward: the function of core/bert.FeatureBERT, vectorised over clients.

Parameters are dicts keyed by the reference state_dict names, each tensor with a leading G axis. Every Linear is one bmm
over (G, b*T, ·) with that client's weights; attention runs on (G*b, heads, T, dh) with SDPA (rows never mix, so padded
rows of a client cannot touch real ones). The model has no BatchNorm: the buffer dict is always empty, `mask` only marks
real rows for the loss. Equivalence to the reference module: tests/test_grouped.py.
"""
import torch
import torch.nn.functional as F

EPS = 1e-5                                                         # nn.TransformerEncoderLayer layer_norm_eps


def _f32(t):
    """fp16 activations -> fp32 for outputs; fp64 (tests) stays fp64."""
    return t if t.dtype == torch.float64 else t.float()


def _linear(P, name, x):
    """x (G, R, i) -> (G, R, o) with P[name.weight] (G, o, i), P[name.bias] (G, o)."""
    w, b = P[name + ".weight"], P[name + ".bias"]
    return torch.baddbmm(b[:, None, :].to(x.dtype), x, w.transpose(1, 2).to(x.dtype))


def _ln(P, name, x):
    d = x.shape[-1]
    y = F.layer_norm(_f32(x), (d,), eps=EPS)
    return (y * P[name + ".weight"][:, None] + P[name + ".bias"][:, None]).to(x.dtype)


@torch.compiler.disable
def _small_attention(q, k, v, drop):
    """Keep SDPA eager: Inductor's dynamic flash padding fails for pruned head-dim 5."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=drop)


def _layer(P, p, x, G, b, T, cfg, train):
    """Post-LN encoder layer (norm_first=False): x = LN1(x + SA(x)); x = LN2(x + FF(x)). x (G, b*T, d)."""
    d, H, drop = x.shape[-1], cfg["heads"], cfg["dropout"] if train else 0.0
    qkv = torch.baddbmm(P[p + "self_attn.in_proj_bias"][:, None, :].to(x.dtype), x,
                        P[p + "self_attn.in_proj_weight"].transpose(1, 2).to(x.dtype))           # (G, b*T, 3d)
    qkv = qkv.view(G * b, T, 3, H, d // H).permute(2, 0, 3, 1, 4)                              # (3, G*b, H, T, dh)
    if d // H < 8:
        a = _small_attention(qkv[0], qkv[1], qkv[2], drop)
    else:
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=drop)
    a = a.transpose(1, 2).reshape(G, b * T, d)
    a = _linear(P, p + "self_attn.out_proj", a)
    x = _ln(P, p + "norm1", x + F.dropout(a, drop, train))
    h = F.dropout(F.gelu(_linear(P, p + "linear1", x)), drop, train)
    x = _ln(P, p + "norm2", x + F.dropout(_linear(P, p + "linear2", h), drop, train))
    return x


def hidden_of(P, x, cfg, train=False):
    """x (G, b, 77) -> CLS output (G, b, d)."""
    G, b, n = x.shape
    dt = x.dtype if x.dtype == torch.float64 else (torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled()
                                                    and x.is_cuda else x.dtype)
    tok = x.to(dt)[..., None] * P["scale"][:, None].to(dt) + P["identity"][:, None].to(dt)    # (G, b, 77, d)
    cls = P["cls"].to(dt).view(G, 1, 1, -1).expand(G, b, 1, tok.shape[-1])
    h = torch.cat([cls, tok], dim=2)
    T = n + 1
    h = h.reshape(G, b * T, -1)
    for i in range(cfg["layers"]):
        h = _layer(P, f"blocks.{i}.", h, G, b, T, cfg, train)
    return h.view(G, b, T, -1)[:, :, 0]


def forward(P, B, x, cfg, mask=None, train=False, hidden=False):
    """x (G, b, 77) -> logits (G, b, 16) [, hidden (G, b, d)], new buffers ({}: no BatchNorm).

    Same signature as the DAGSNet engine of fl_dagsnet_veremi_v2 so methods and the train loop are unchanged.
    Dropout is active only when train=True."""
    h = hidden_of(P, x, cfg, train)
    logits = _f32(_linear(P, "head", h))
    return (logits, _f32(h), {}) if hidden else (logits, {})


# Entry points the train loop / evaluator call; scripts swap in torch.compile'd versions (static shapes for train,
# dynamic for eval). Plain `forward` stays the reference for tests.
TRAIN = forward
EVAL = forward


def compile_all():
    global TRAIN, EVAL
    from core import optim
    torch._dynamo.config.recompile_limit = 64                          # methods legitimately use several shapes
    TRAIN = torch.compile(forward, dynamic=False)
    EVAL = torch.compile(forward, dynamic=True)
    optim.compile_all()
