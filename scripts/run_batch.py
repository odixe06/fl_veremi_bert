"""Seed-42 batch: 14 FL baselines on FeatureBERT (owner 13/11/2026), one private Kaggle notebook per session.

    python scripts/run_batch.py plan       # live quota -> runs/batch/plan.json (locked; never re-planned)
    python scripts/run_batch.py launch     # generate + embed W&B + push every planned first session
    python scripts/run_batch.py collect    # one pass: status, pull terminal sessions, push continuations / CANPerFL
    python scripts/run_batch.py audit      # audit every finished run + 10-metric report -> runs/batch/report.md

A run (one method) is a chain of sessions <slug>, <slug>-p2, ... on ONE account (a continuation mounts the private output of
the previous session). Each session gets --max-hours = min(CAP, live quota / MARGIN) and run_fl stops before a round would
cross it, saving resume state; `collect` pushes the next session when the last one COMPLETEd below round 50 and the account
has >= MIN_SESSION hours of quota (quota resets weekly, Saturday 00:00 UTC). CANPerFL starts on the FedAvg account once
FedAvg's chain reached round 50 (its last session's output holds state.pt). Accounts running other projects are excluded.
"""
import json
import ast
import hashlib
import math
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/batch"
SK = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
PY = sys.executable
SEED = 42
# method, extra --set, measured seed-42 train+eval hours of the DAGSNet build (fl_dagsnet_veremi_v2 runs/pulls rounds.jsonl)
METHODS = [("nilm", [], 14.4), ("apfed", [], 7.4), ("pfedes", [], 7.1), ("perfedskd", [], 5.2), ("v2x_htfl", [], 5.2),
           ("fedtgp", [], 5.05), ("tinyproto", ["mu=0.00015"], 5.05), ("fd_ids", [], 4.4), ("afpha", [], 3.9),
           ("fedproto", [], 3.85), ("fedprox", [], 3.7), ("fedavg", [], 3.4), ("lwfednids", [], 1.1)]
CANPERFL_H = 0.83
FACTOR = 1.5          # BERT / DAGSNet time, planning guess (local RTX 3050: 0.85x on real data, 2.05x synthetic) -> re-fit
SETUP, MARGIN, CAP, MIN_SESSION = 0.8, 1.15, 11.5, 3.0
EXCLUDE = {"novahihi", "keocaosu", "emtrangtri", "minhtran0601"}   # mobihet B20 (owner 13/11)
PULL = r"(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|models/r\d{3}\.pt)$)"


def slug(m, part=1):
    return f"flb-{m.replace('_', '-')}" + (f"-p{part}" if part > 1 else "")


def sh(args, **kw):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, **kw).stdout


def quota():
    out = sh([PY, str(SK / "kaggle_account.py"), "quota"])
    q = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1].endswith("h") and p[0] != "account":
            q[p[0]] = float(p[1][:-1])
    return q


def inventory(account):
    return json.loads(sh([PY, str(SK / "kaggle_as.py"), account, "--", PY,
                          "scripts/kaggle_inventory.py", account], timeout=600))


def plan():
    f = OUT / "plan.json"
    assert not f.exists(), "plan locked"
    q = {a: h for a, h in quota().items() if a not in EXCLUDE}
    live = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {a: pool.submit(inventory, a) for a, h in q.items() if h >= MIN_SESSION * MARGIN}
        for a, future in futures.items():
            try:
                live[a] = future.result()
                print("INVENTORY", a, "busy", len(live[a]["busy"]), flush=True)
            except (subprocess.SubprocessError, ValueError):
                print("SKIP inventory unknown", a, flush=True)
    # Existing projects retain their account quota as well as their occupied slots.
    q = {a: min(q[a], s["gpu_left"]) for a, s in live.items() if not s["busy"]}
    left, slots, rows = {a: h / MARGIN for a, h in q.items()}, {a: 2 for a in q}, []
    for m, extra, hrs in sorted(METHODS, key=lambda t: -t[2]):
        need = hrs * FACTOR + SETUP + (CANPERFL_H * FACTOR + SETUP if m == "fedavg" else 0)
        ok = [a for a in left if slots[a] > 0 and left[a] >= MIN_SESSION]
        assert ok, ("no account for", m)
        fit = [a for a in ok if left[a] >= min(need, CAP)]
        a = min(fit, key=lambda a: left[a]) if fit else max(ok, key=lambda a: left[a])   # tightest full fit, else largest
        budget = math.floor(min(CAP, left[a], need) * 100) / 100
        left[a] -= budget
        slots[a] -= 1
        rows.append(dict(slug=slug(m), run=slug(m), method=m, set=extra, account=a, part=1, max_hours=budget,
                         est_hours=round(need, 2)))
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"head": sh(["git", "rev-parse", "HEAD"]).strip(), "quota": q, "inventory": live, "factor": FACTOR,
                             "rows": rows}, indent=1))
    for r in rows:
        print(r["account"], r["slug"], "budget", r["max_hours"], "est", r["est_hours"])


def push(r, head):
    a = r["account"]
    assert a not in EXCLUDE
    live = inventory(a)
    assert len(live["busy"]) < 2, ("GPU slots occupied", a, live["busy"])
    assert r["slug"] not in live["versions"], ("inspect existing kernel before push", a, r["slug"])
    # Reserve budgets of other active/planned sessions from this locked batch, not just consumed GPU hours.
    _, rows = sessions()
    reserved = sum(x["max_hours"] * MARGIN for x in rows.values()
                   if x["account"] == a and x["slug"] != r["slug"] and not (OUT / f"{x['slug']}.done").exists())
    minimum = min(MIN_SESSION, r["max_hours"])
    r["max_hours"] = math.floor(min(r["max_hours"], (live["gpu_left"] - reserved) / MARGIN) * 100 + 1e-8) / 100
    assert r["max_hours"] >= minimum, ("quota changed; inspect budget", a)
    args = [PY, "scripts/gen_notebook.py", "--owner", a, "--slug", r["slug"], "--method", r["method"], "--run", r["run"],
            "--rounds", str(r.get("rounds", 50)), "--max-hours", str(r["max_hours"]), "--sha", head,
            "--set", f"seed={SEED}", *r["set"]]
    if r.get("resume"):
        args += ["--resume-kernel", f"{a}/{r['resume']}"]
    if r.get("input"):
        args += ["--input-kernel", f"{a}/{r['input']}"]
    sh(args)
    d = ROOT / "notebooks" / r["slug"]
    sh([PY, str(SK / "embed_wandb_key.py"), str(d / f"{r['slug']}.ipynb"), "--metadata", str(d / "kernel-metadata.json")])
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["id"] == f"{a}/{r['slug']}" and meta["machine_shape"] == "NvidiaTeslaT4"
    nb = json.loads((d / f"{r['slug']}.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cmd" for t in node.targets):
                    raw = node.value.func.value.value
                    argv = shlex.split(raw.split("&&", 1)[1])
                    if "--set" in argv:
                        for item in argv[argv.index("--set") + 1:]:
                            json.loads(item.split("=", 1)[1])
    marker = OUT / f"{r['slug']}.pushed"
    assert not marker.exists(), r["slug"]
    (OUT / f"{r['slug']}.intent").write_text(json.dumps(dict(r, at=time.strftime("%FT%T%z"))))
    out = sh([PY, str(SK / "kaggle_as.py"), a, "--", "kaggle", "kernels", "push", "-p", str(d)])
    (OUT / f"{r['slug']}.push-reply").write_text(out)
    match = re.search(r"Kernel version (\d+) successfully", out)
    assert match, ("push accepted but version unknown; inspect", r["slug"])
    r["version"] = int(match[1])
    (OUT / f"{r['slug']}.intent").write_text(json.dumps(dict(r, at=time.strftime("%FT%T%z"))))
    marker.write_text(out)
    print("PUSHED", a, r["slug"], r["max_hours"], out.strip().splitlines()[-1] if out.strip() else "", flush=True)


def status(r):
    try:
        out = sh([PY, str(SK / "kaggle_as.py"), r["account"], "--", "kaggle", "kernels", "status", f"{r['account']}/{r['slug']}/{r['version']}"])
    except subprocess.CalledProcessError as e:
        return "UNKNOWN:" + (e.stdout or "")[-200:]
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    if match:
        return "CANCEL" if match[1].startswith("CANCEL") else match[1]
    return "UNKNOWN:" + out[-200:]


def sessions():
    """Every pushed or planned session row: plan rows + continuation rows recorded in their .intent files."""
    p = json.loads((OUT / "plan.json").read_text())
    rows = {r["slug"]: r for r in p["rows"]}
    for f in OUT.glob("*.intent"):
        r = json.loads(f.read_text())
        rows[r["slug"]] = r
    return p, rows


def last_round(r):
    f = list((ROOT / "runs/pulls" / r["slug"]).rglob(f"{r['run']}/rounds.jsonl"))
    if not f:
        return 0
    return max(json.loads(line)["round"] for line in f[0].read_text().splitlines() if line.strip())


def launch():
    p, _ = sessions()
    for r in p["rows"]:
        if not (OUT / f"{r['slug']}.pushed").exists():
            assert not (OUT / f"{r['slug']}.intent").exists(), ("inspect ambiguous push", r["slug"])
            push(r, p["head"])


def collect():
    """One pass, no loop. Re-run until it prints ALL_TERMINAL."""
    p, rows = sessions()
    st = {s: status(r) for s, r in rows.items() if (OUT / f"{s}.pushed").exists() and not (OUT / f"{s}.done").exists()}
    for s, v in st.items():
        if v in ("COMPLETE", "ERROR", "CANCEL"):
            r = rows[s]
            dest = ROOT / "runs/pulls" / s
            rc = subprocess.run([PY, str(SK / "kaggle_as.py"), r["account"], "--", PY, str(SK / "pull_kernel_output.py"),
                                 f"{r['account']}/{s}/{r['version']}", "-p", str(dest), "--pattern", PULL], cwd=ROOT).returncode
            print("PULL", s, v, "rc", rc, flush=True)
            if rc == 0:
                (OUT / f"{s}.done").write_text(v)
    # chain heads: the latest session of every run
    head = {}
    for r in rows.values():
        if (OUT / f"{r['slug']}.pushed").exists() and r["part"] >= head.get(r["run"], {"part": 0})["part"]:
            head[r["run"]] = r
    q = None
    state = {}
    for run, r in sorted(head.items()):
        done = (OUT / f"{r['slug']}.done").read_text() if (OUT / f"{r['slug']}.done").exists() else ""
        rr = last_round(r) if done else 0
        target = r.get("rounds", 50)
        state[run] = {"session": r["slug"], "status": done or st.get(r["slug"], "?"), "round": rr}
        if done != "COMPLETE":
            continue
        manifest = ROOT / "runs/pulls" / r["slug"] / "provenance/source_manifest.json"
        if not manifest.exists():
            matches = list((ROOT / "runs/pulls" / r["slug"]).rglob("source_manifest.json"))
            assert len(matches) == 1, ("source manifest missing/ambiguous", r["slug"])
            manifest = matches[0]
        spec = json.loads(manifest.read_text())
        assert spec["base_sha"] == p["head"] and not spec["overlay"]
        for name, digest in spec["files"].items():
            source = manifest.parent / "source" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            expected = subprocess.check_output(["git", "show", f"{p['head']}:{name}"], cwd=ROOT)
            assert hashlib.sha256(expected).hexdigest() == digest, name
        nxt = None
        if rr < target:
            nxt = dict(r, slug=slug(r["method"], r["part"] + 1), part=r["part"] + 1, resume=r["slug"])
            nxt.pop("max_hours", None)
        elif r["method"] == "fedavg" and "flb-canperfl" not in head:
            nxt = dict(slug="flb-canperfl", run="flb-canperfl", method="canperfl", account=r["account"], part=1, rounds=10,
                       set=[f"init='\\\"/kaggle/input/**/{r['run']}/state.pt\\\"'"], input=r["slug"])
        if nxt is None or (OUT / f"{nxt['slug']}.intent").exists():
            continue
        q = q if q is not None else quota()
        busy = sum(1 for x in head.values() if x["account"] == r["account"] and not (OUT / f"{x['slug']}.done").exists())
        budget = round(min(CAP, q.get(r["account"], 0) / MARGIN), 2)
        if budget < MIN_SESSION or busy >= 2:
            state[run]["next"] = f"WAIT quota {q.get(r['account'], 0):.1f}h busy {busy}: {nxt['slug']}"
            continue
        nxt["max_hours"] = budget
        push(nxt, p["head"])
        q[r["account"]] -= budget * MARGIN
        state[run]["next"] = f"PUSHED {nxt['slug']}"
    (OUT / "status.json").write_text(json.dumps({"at": time.strftime("%FT%T%z"), "runs": state}, indent=1))
    for run, s in sorted(state.items()):
        print(f"{s['status']:10s} r{s['round']:02d} {run:22s} {s['session']:24s} {s.get('next', '')}")
    expected = {slug(m) for m, _, _ in METHODS} | {"flb-canperfl"}
    finished = {run for run, s in state.items() if s["status"] in ("ERROR", "CANCEL")
                or (s["status"] == "COMPLETE" and s["round"] >= (10 if run == "flb-canperfl" else 50))}
    left = sorted(expected - finished)
    print("ALL_TERMINAL" if not left else f"REMAINING {len(left)}: {' '.join(left)}", flush=True)


def audit():
    """Audit each finished run from its last pulled session (which holds rounds 1..R of the shared run dir)."""
    _, rows = sessions()
    final = {}
    for r in rows.values():
        if (OUT / f"{r['slug']}.done").exists() and r["part"] >= final.get(r["run"], ({"part": 0}, None))[0]["part"]:
            final[r["run"]] = (r, ROOT / "runs/pulls" / r["slug"])
    res, dirs, failures = {}, [], []
    for run, (r, pulled) in sorted(final.items()):
        found = [d.parent for d in pulled.rglob("rounds.jsonl") if d.parent.name == run]
        if not found:
            res[run] = "NO_RUN_DIR"
            failures.append(run)
            continue
        out = OUT / "audit" / f"{run}.json"
        out.parent.mkdir(exist_ok=True)
        cp = subprocess.run([PY, "scripts/audit_run.py", str(found[0]), "--support", "runs/test_support.npy", "--out", str(out),
                             "--seed", str(SEED), "--rounds", str(r.get("rounds", 50))], cwd=ROOT, capture_output=True, text=True)
        res[run] = (cp.stdout.strip() or cp.stderr.strip().splitlines()[-1]) if (cp.stdout or cp.stderr) else f"rc {cp.returncode}"
        print(run, "->", res[run], flush=True)
        if cp.returncode == 0:
            dirs.append(str(found[0]))
            if not json.loads(out.read_text())["complete"]:
                failures.append(run)
        else:
            failures.append(run)
    (OUT / "audit-summary.json").write_text(json.dumps(res, indent=1))
    if dirs:
        rep = sh([PY, "scripts/report.py", *dirs])
        (OUT / "report.md").write_text(rep)
        print("REPORT", OUT / "report.md")
    expected = {slug(m) for m, _, _ in METHODS} | {"flb-canperfl"}
    assert not failures and set(res) == expected, ("batch audit incomplete/failed", failures, sorted(expected - set(res)))


if __name__ == "__main__":
    {"plan": plan, "launch": launch, "collect": collect, "audit": audit}[sys.argv[1]]()
