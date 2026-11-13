"""FeatureBERT — client model of every method (repo dangvanhoang1711/veremi-bert-77, core/central_backbones.py).

x (B, 77) scaled -> one token per column: x_j * scale_j + identity_j (d dims) -> [CLS] + 77 tokens -> L post-LN Transformer
encoder layers (nn.TransformerEncoderLayer: d, heads, FFN, dropout 0.1, GELU) -> head Linear(d, 16) on the CLS output.
FULL = the centralized repo's model exactly (77.904 parameters, same parameter names and seed-42 initialisation order).
Spec: knowledge/bert.md. `core/grouped.py` is the same function for G independent models at once.
"""
import torch
import torch.nn as nn

FULL = {"d": 64, "heads": 4, "ffn": 128, "layers": 2, "dropout": 0.1, "n_features": 77, "num_classes": 16}


def cfg_full(**over):
    cfg = dict(FULL)
    cfg.update(over)
    cfg["head_hidden"] = cfg["d"]                                  # representation size (prototype methods)
    return cfg


class FeatureBERT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or cfg_full()
        d, n = cfg["d"], cfg["n_features"]
        self.scale = nn.Parameter(torch.empty(n, d))
        self.identity = nn.Parameter(torch.empty(n, d))
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        for param in (self.scale, self.identity, self.cls):
            nn.init.normal_(param, std=0.02)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ffn"], cfg["dropout"], "gelu", batch_first=True)
            for _ in range(cfg["layers"])])
        self.head = nn.Linear(d, cfg["num_classes"])

    def hidden(self, x):                                           # CLS output = representation f(x)
        tokens = x.unsqueeze(-1) * self.scale + self.identity
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), tokens), dim=1)
        return self.blocks(tokens)[:, 0]

    def forward(self, x):
        return self.head(self.hidden(x))


def build_model(cfg):
    return FeatureBERT(cfg)


def n_params(cfg):
    return sum(p.numel() for p in FeatureBERT(cfg).parameters())
