"""Every method runs 2 rounds on a toy CPU environment, produces finite models and per-client/global confusion matrices,
and its state round-trips through state()/load_state()."""
import importlib

import numpy as np
import pytest
import torch

from core import evaluate, local
from core.bert import cfg_full
from core.data import Split
from core.flat import Layout

METHODS = ["fedavg", "fedprox", "pfedes", "fedproto", "fedtgp", "tinyproto", "nilm", "fd_ids", "afpha", "perfedskd", "lwfednids", "apfed", "v2x_htfl"]


class ToyEnv:
    def __init__(self, K=6, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg_full()                                    # LwFedNIDS needs d 64 (pruned inner = d)
        self.hp = {"batch": 8, "group_rows": 32, "weight_decay": 1e-4, "clip": 1.0, "amp": False, "seed": 42,
                   "lr_max": 1e-3, "lr_min": 1e-5, "rounds": 2}
        self.lay = Layout(self.cfg)
        self.dev = torch.device("cpu")
        self.world, self.rank = 1, 0

        def split(lo, hi):
            n = rng.integers(lo, hi, K)
            off = np.concatenate([[0], np.cumsum(n)])
            y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
            x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float() + 0.3 * y[:, None].float()
            return Split(x, y, torch.from_numpy(off), torch.arange(K))
        self.train, self.test = split(10, 50), split(4, 12)
        self.K = K
        self.ids = np.arange(K)
        self.all_ids = np.arange(K)
        self.n = torch.as_tensor(self.train.sizes, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(K)), self.hp["batch"], self.hp["group_rows"])
        self.gen = torch.Generator().manual_seed(0)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g)

    def eval_global(self, p, b):
        return evaluate.global_cm(self.lay, p, b, self.test, self.cfg)

    def eval_clients(self, bp, bb):
        return evaluate.own_cm(self.lay, bp, bb, self.test, self.cfg, torch.arange(self.K))


@pytest.mark.parametrize("name", METHODS)
def test_method_runs(name):
    env = ToyEnv()
    M = importlib.import_module(f"methods.{name}").Method(env)
    for r in (1, 2):
        info = M.round(r, 1e-3)
        assert np.isfinite(info["train_loss"])
    views = M.evaluate()
    for v, cms in views.items():
        assert cms.shape == (env.K, 16, 16)
        assert cms.sum() == len(env.test.Y), v
    s = M.state()
    M2 = importlib.import_module(f"methods.{name}").Method(ToyEnv())
    M2.load_state(s)
    for v in views:
        assert (M2.evaluate()[v] == views[v]).all()


def test_canperfl_finetunes_fedavg_state(tmp_path):
    env = ToyEnv()
    p, b = env.lay.init(42, 1)
    torch.save({"method": "fedavg", "cfg": env.cfg, "round": 50, "method_state": {"p": p[0], "b": b[0]}},
               tmp_path / "state.pt")
    env.hp["init"] = str(tmp_path / "*.pt")
    M = importlib.import_module("methods.canperfl").Method(env)
    before = M.evaluate()["client"]
    assert (before == M.evaluate()["global"]).all()                  # untouched clients = the FedAvg model
    assert np.isfinite(M.round(1, 1e-3)["train_loss"])
    assert not torch.equal(M.bank_p[0], p[0])
    state = M.state()
    assert torch.equal(state["p"], p[0])  # small snapshot retains FedAvg reference even with large banks
    M2 = importlib.import_module("methods.canperfl").Method(env)
    M2.load_state(state)
    for view, cm in M.evaluate().items():
        assert (M2.evaluate()[view] == cm).all()


def test_tinyproto_aligned_prototypes_do_not_amplify(monkeypatch):
    """Replay APS with perfectly aligned features: the observed runaway target must not feed itself upward."""
    from methods import tinyproto

    env = ToyEnv()
    M = tinyproto.Method(env)
    cnt = torch.zeros(env.K, 16)
    client = torch.repeat_interleave(torch.arange(env.K), env.train.off[1:] - env.train.off[:-1])
    cnt.index_put_((client, env.train.Y.long()), torch.ones(len(env.train.Y)), accumulate=True)

    def class_sums(*args, **kwargs):
        feature = torch.ones(16, env.cfg["head_hidden"]) if M.target is None else M.target
        return cnt[..., None] * feature[None], cnt

    monkeypatch.setattr(tinyproto.evaluate, "class_sums", class_sums)
    monkeypatch.setattr(tinyproto.local, "local_epoch", lambda lay, flat, *args, **kwargs: {"loss": flat.new_zeros(len(flat))})
    for r in range(1, 11):
        M.round(r, 1e-3)
        assert M.target.abs().max() <= 1.0, f"APS amplifies aligned features at r{r}: {M.target.abs().max()}"
