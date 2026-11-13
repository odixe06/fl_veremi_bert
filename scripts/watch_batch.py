"""Background watcher for the seed-42 batch: `run_batch.py collect` every 20 min, `audit` once everything is terminal.

    python scripts/watch_batch.py        # log runs/batch-watch/watch.log, state runs/batch-watch/state.json

collect pulls terminal sessions and pushes continuations / CANPerFL itself. The watcher exits (waking the session) when:
the batch is ALL_TERMINAL (then audited), a session ends ERROR/CANCEL, or MAX_TICKS (~48 h) pass.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/batch-watch"
TICK, MAX_TICKS = 1200, 144


def run(cmd):
    p = subprocess.run([sys.executable, "scripts/run_batch.py", cmd], cwd=ROOT, capture_output=True, text=True, timeout=3000)
    return p.returncode, p.stdout + p.stderr


def log(msg):
    with (OUT / "watch.log").open("a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + msg.rstrip() + "\n")


def first_round():
    """Read exact run IDs, once r1 exists; live timing evidence, not final artifact audit."""
    import wandb
    api = wandb.Api(timeout=30)
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "runs/batch").glob("*.intent"))]
    for row in rows:
        if not (ROOT / "runs/batch" / f"{row['slug']}.pushed").exists():
            continue
        try:
            path = f"{api.default_entity}/fl-veremi-bert/{row['run']}"
            run = api.run(path)
            for item in run.scan_history(keys=["round", "t_train", "t_eval", "_timestamp"]):
                if item["round"] == 1:
                    assert run.config["world"] == 2
                    spec = json.loads((ROOT / "runs/batch" / f"{row['slug']}.intent").read_text())
                    return dict(item, method=row["method"], wandb=path,
                                kernel=f"{row['account']}/{row['slug']}/{spec['version']}")
        except Exception as exc:
            log(f"first-round read unavailable {row['run']}: {type(exc).__name__}")
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev = OUT / "state.json"
    # sessions already reported bad (and handled) by an earlier watcher do not stop this one again
    state = json.loads(prev.read_text()) if prev.exists() else {"bad": []}
    state["ticks"] = 0
    for tick in range(MAX_TICKS):
        state["ticks"] = tick + 1
        try:
            rc, out = run("collect")
        except subprocess.TimeoutExpired:
            rc, out = -1, "collect timeout"
        log(f"collect rc={rc}\n{out}")
        state["last"] = out[-3000:]
        if rc != 0:
            state["error"] = {"stage": "collect", "rc": rc}
            (OUT / "state.json").write_text(json.dumps(state, indent=1))
            print("BATCHW_COLLECT_ERROR", rc, flush=True)
            return 1
        sts = json.loads((ROOT / "runs/batch/status.json").read_text()).get("runs", {}) if rc == 0 else {}
        bad = sorted(r for r, s in sts.items() if s["status"] in ("ERROR", "CANCEL"))
        if set(bad) - set(state["bad"]):
            state["bad"] = bad
            (OUT / "state.json").write_text(json.dumps(state, indent=1))
            log(f"STOP non-COMPLETE terminal: {bad}")
            print("BATCHW_NON_COMPLETE", bad, flush=True)
            return 2
        if rc == 0 and "ALL_TERMINAL" in out:
            rc, out = run("audit")
            log(f"audit rc={rc}\n{out}")
            state["audit"] = {"rc": rc, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "out": out[-3000:]}
            (OUT / "state.json").write_text(json.dumps(state, indent=1))
            print("BATCHW_DONE", rc, flush=True)
            return rc
        if not state.get("first_round"):
            try:
                observed = first_round()
            except Exception as exc:
                log(f"first-round API unavailable: {type(exc).__name__}")
                observed = None
            if observed:
                state["first_round"] = observed
                (OUT / "first_round.json").write_text(json.dumps(observed, indent=1))
                (OUT / "state.json").write_text(json.dumps(state, indent=1))
                log(f"first T4 r1 timing: {observed}")
                print("BATCHW_FIRST_ROUND", observed["kernel"], flush=True)
                return 0  # wake owner thread to record measurement, then resume watcher
        (OUT / "state.json").write_text(json.dumps(state, indent=1))
        time.sleep(TICK)
    print("BATCHW_TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
