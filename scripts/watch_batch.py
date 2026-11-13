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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev = OUT / "state.json"
    # sessions already reported bad (and handled) by an earlier watcher do not stop this one again
    state = {"ticks": 0, "bad": json.loads(prev.read_text()).get("bad", []) if prev.exists() else []}
    for tick in range(MAX_TICKS):
        state["ticks"] = tick + 1
        try:
            rc, out = run("collect")
        except subprocess.TimeoutExpired:
            rc, out = -1, "collect timeout"
        log(f"collect rc={rc}\n{out}")
        state["last"] = out[-3000:]
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
            return 0
        (OUT / "state.json").write_text(json.dumps(state, indent=1))
        time.sleep(TICK)
    print("BATCHW_TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
