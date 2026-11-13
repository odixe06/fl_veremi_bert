"""Audit stored v4 round metrics/CMs and small snapshots; report missing replay evidence explicitly.

python scripts/audit_run.py runs/pulls/<slug>/<run> --support runs/test_support.npy --out <report.json>
The support file is built once from every local test parquet's client_id/label columns.
This does not certify unpulled per-client banks or re-run predictions.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data, metrics
from core.bert import cfg_full
from core.fl import cosine_lr


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_support(path):
    if path.exists():
        return np.load(path)
    import pyarrow.parquet as pq
    out = np.zeros((5343, 16), dtype=np.int64)
    for p in data.files(data.find_root(), "test"):
        t = pq.ParquetFile(p).read(columns=["client_id", "label"])
        ids, y = [t.column(k).to_numpy().astype(np.int64) for k in ("client_id", "label")]
        out += np.bincount(ids * 16 + y, minlength=5343 * 16).reshape(5343, 16)
    assert out.sum() == 9347679 and (out.sum(1) > 0).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def audit(root, support, seed=42, rounds=50):
    cfg = json.loads((root / "config.json").read_text())
    hp = cfg["hp"]
    locked = dict(rounds=rounds, batch=64, lr_max=1e-3, lr_min=1e-5, weight_decay=1e-4, clip=1.0, seed=seed, amp=True)
    assert all(hp[k] == v for k, v in locked.items()), hp
    assert cfg["clients"] == 5343 and cfg["n_params"] == (10194 if cfg["method"] == "lwfednids" else (82128 if cfg["method"] == "v2x_htfl" else 77904))  # FeatureBERT; LwFedNIDS prunes, V2X-HtFL adds the projection
    assert cfg["method"] == "lwfednids" or cfg["cfg"] == cfg_full()
    rows = [json.loads(line) for line in (root / "rounds.jsonl").read_text().splitlines()]
    assert rows and [x["round"] for x in rows] == list(range(1, len(rows) + 1))
    assert len(rows) <= hp["rounds"]
    assert sorted(p.name for p in root.glob("cms_r*.npz")) == [f"cms_r{r:03d}.npz" for r in range(1, len(rows) + 1)]
    hashes, last, views = {}, {}, None
    for row in rows:
        r = row["round"]
        for k, v in row.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), (r, k, v)
            elif isinstance(v, list):
                assert np.isfinite(v).all(), (r, k)
        assert math.isclose(row["lr"], cosine_lr(r, hp["rounds"], 1e-3, 1e-5), rel_tol=1e-12)
        p = root / f"cms_r{r:03d}.npz"
        hashes[p.name] = sha256(p)
        with np.load(p) as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            current = sorted(k for k in z.files if k != "ids")
            assert current and (views is None or current == views)
            views = current
            for v in views:
                cm = z[v]
                assert cm.shape == (5343, 16, 16) and np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
                assert np.array_equal(cm.sum(2), support), (r, v, "test support mismatch")
                s = metrics.summary(cm)
                keys = metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10", "client_f1_macro_median", "client_f1_macro_min", "f1_per_class"]
                for k in keys:
                    np.testing.assert_allclose(row[f"{v}/{k}"], s[k], rtol=0, atol=1e-12, err_msg=f"r{r} {v}/{k}")
                last[v] = s
        p = root / "models" / f"r{r:03d}.pt"
        assert p.is_file(), (r, "missing snapshot")
        snap = torch.load(p, map_location="cpu", weights_only=True)
        assert snap
        for k, t in snap.items():
            assert torch.is_tensor(t) and torch.isfinite(t).all(), (r, k)
        hashes[str(p.relative_to(root))] = sha256(p)
    for name in ("config.json", "rounds.jsonl"):
        hashes[name] = sha256(root / name)
    return dict(status="PASS_BOUNDED", method=cfg["method"], rounds=len(rows), complete=len(rows) == 50,
                checks=["locked HP/model", "contiguous rounds/LR", "exact full-test per-client class support",
                        "all ten metrics/client quantiles/per-class F1 recomputed", "finite available snapshot tensors"],
                limits=["No stored per-row predictions: cannot replay predictions into CMs",
                        "Small snapshots omit private parameter banks: cannot replay personalized inference",
                        "Full rank resume states excluded from this pull; exact resume not certified",
                        "Dataset feature-content fingerprints absent from historical config"],
                last_metrics=last, hashes=hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42, help="expected hp seed (seed 41/43 batch, 12/11/2026)")
    ap.add_argument("--rounds", type=int, default=50, help="expected hp rounds (CANPerFL fine-tune: 10)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    report = audit(a.run, test_support(a.support), a.seed, a.rounds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(report["method"], report["status"], "rounds", report["rounds"], "complete", report["complete"])


if __name__ == "__main__":
    main()
