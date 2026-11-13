---
name: federated-training-speed
description: Make a multi-client (federated) PyTorch training harness run faster on Kaggle 2x T4 or a local GPU without changing the method -- measure where step time goes, remove host syncs, capture CUDA graphs, run one persistent worker per GPU, benchmark candidate code paths, and prove the faster code trains the same thing (bit-exact or within a stated tolerance). Use before long FL runs, when ms/step or GPU idle time is high, or when adding workers / compile / fused kernels to a harness (e.g. "tối ưu tốc độ", "code tối ưu", "chạy 2 worker", "ms/bước", "GPU rảnh", "benchmark T4"). Project-independent; not for choosing the method, model or hyperparameters.
---

# Federated training speed (code only, method fixed)

## Paths and scope

This file is the skill; `references/` and `scripts/` resolve from this directory. The skill holds no
project state: read the calling project's agent instructions and status files for its decisions --
in particular whether a speed change may break bit-exactness with results already reported.
Every number here is evidence from earlier projects (DAGSNet, 395k parameters, tabular input,
2x Tesla T4, torch 2.10). Re-measure on the target model; never quote them as targets.

## Contract: what may change

- **May change:** kernels and fusion, CUDA graphs, host synchronisation, process layout (workers),
  scheduling of clients onto GPUs, evaluation batching/sharding, I/O, logging frequency.
- **May not change:** model, clients and partition, data and preprocessing, batch size, steps per
  local epoch (tail batches included), learning-rate schedule, loss terms and weights, aggregation,
  what is communicated (payload, codec, bytes), evaluation protocol, seeds.
- Name the acceptance class **before** writing code:
  1. **Bit-exact** -- weights, predictions, communication log and metrics identical to the old code
     over a multi-epoch fixture. Keeps every earlier result comparable. Default when results already
     exist.
  2. **Tolerance-equivalent** -- per-step differences at fp16/fp32 rounding level, identical
     applied/skipped step counts, end-to-end metric inside seed-to-seed spread. The trajectory changes,
     so **controls are re-run in the new code**. Needs the project owner's go.

## Procedure

1. **Count steps and budget.** `steps/round = sum(ceil(n_k / batch))`; hours = steps x ms/step /
   GPUs x rounds. [perf-federated.md](references/perf-federated.md) §1.
2. **Measure the existing code from its own artifacts.**
   `python scripts/ms_per_step.py <run_dir>` reports ms/step per process from committed round
   records (adjust `--clients/--seconds/--steps` keys to the project's schema). Then time GPU compute
   alone (profiler or batch sweep: [local-gpu-probe.md](references/local-gpu-probe.md),
   [perf-federated.md](references/perf-federated.md) §2). Wall time well above GPU time means the
   step is CPU/launch-bound -- the usual case on 4 vCPU per 2 GPUs.
3. **Bit-exact ladder**, measured after each rung:
   - resident data on each GPU, no DataLoader ([perf-federated.md](references/perf-federated.md) §4);
   - **no host sync inside the step** -- `.item()`, `float(t)`, `GradScaler.get_scale()`,
     `assert torch.isfinite(t)`, boolean-mask index-put, `nonzero`, `unique`, printing tensors.
     Accumulate on device; check once per client epoch ([perf-federated.md](references/perf-federated.md) §5b);
   - in deterministic mode set `torch.utils.deterministic.fill_uninitialized_memory = False` (it can
     add hundreds of `fill_` kernels per step) -- only if no code reads uninitialized memory;
   - CUDA graphs on the **modules** (`torch.compile(mode="reduce-overhead")`), loss/scaler/optimizer
     outside the graph, tail batch eager, gate with dropout off on both sides
     ([perf-federated.md](references/perf-federated.md) §5);
   - **one persistent process per GPU**: [two-worker-harness.md](references/two-worker-harness.md).
4. **Benchmark on the real device before adopting** a rung or a worker count: a short no-train job
   timing ~2,000 real steps per configuration; choose by total steps/s across GPUs.
5. **Tolerance rungs** (largest wins, owner's go only): `fused=True` optimizer, `cudnn.benchmark=True`
   instead of deterministic algorithms, whole-step CUDA graph (forward + loss + backward + unscale +
   clip + optimizer). Plan the control re-runs first: [equivalence-tests.md](references/equivalence-tests.md) §2.
6. **Prove it** with [equivalence-tests.md](references/equivalence-tests.md): old vs new code,
   1 vs W workers, interrupted-and-continued vs uninterrupted run.
7. **Report** before/after (ms/step per process, total steps/s, seconds per round), the acceptance
   class, the tests that passed and the expected gain per pushed run. Running jobs do not benefit;
   the change applies from the next launch.

## Reference measurements (earlier projects, 2x T4)

| setup | ms/step | class |
|---|---:|---|
| one model, compiled reduce-overhead + AMP, fused Adam, cudnn.benchmark, batch 512 | 6.2 | tolerance |
| teacher forward + student forward/backward, same settings | 12.1 | tolerance |
| two trained models (private + shared proxy) + 2 teacher forwards, deterministic, sync-free, modules compiled | 26.5 | bit-exact |
| same harness before removing host syncs | 34.0 (local) | bit-exact |
| GPU compute share of the 22-26 ms deterministic step | ~12 | -- |

One process per GPU added +15 % total throughput over one process for both GPUs; 2-3 processes per GPU
only +3-5 % more on 4 vCPU. More processes do not fix a CPU-bound step; fewer launches do.

## Session startup (Kaggle, 2x T4)

Read the timeline from the kernel log before changing anything: its JSON rows carry `time` in seconds from session start.
Measured on one DAGSNet FL session (5,343 clients, 395k parameters), about 10.5 minutes of a 10 h run:

| phase | seconds | handling |
|---|---:|---|
| clone, imports, credentials | ~22 | nothing to gain |
| SHA256 of the mounted npy cache (7.3 GB) | 80 | keep: audits rely on it, and it warms the page cache for the data load |
| data load and bootstrap statistics | ~60 | method-dependent |
| compile gate | 110 | cached below |
| first-round train excess, 779 s vs 669 s | 110 | cached below |
| first-round eval excess, 275 s vs 19 s (dynamic shapes) | 256 | cached below |

Compile is about 80 % of startup, so it is the first target; a prebuilt npy cache already removes the far larger parquet build.
Startup is paid per session, which matters most when a run is chained across short quotas.

**Carry the Inductor cache to resume sessions.** A warm start reuses the same generated kernels: a local two-round check gave identical
CMs and history, 36 s against 140 s cold (sm86, 2 clients; not yet measured on T4).
The cache is thousands of small files (6,589 files, 190 MB), so never leave it loose in `/kaggle/working`: output listing and pulls
slow down. Keep it in temp and put one tar in the run directory, which the resume step already copies:

```python
tar = f"/kaggle/working/{run}/inductor-cache.tar"
if os.path.exists(tar): subprocess.run(["tar", "-xf", tar, "-C", "/kaggle/temp"])   # resume session only
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/kaggle/temp/inductor"
rc = subprocess.run(cmd, shell=True).returncode
subprocess.run(["tar", "-cf", tar, "-C", "/kaggle/temp", "inductor"])               # before asserting rc
```

A cache miss only recompiles, so the tar step must never fail the session. The tar must be set up in session 1: a kernel already
pushed cannot start writing it. Re-check the bit-exact comparison after a torch, driver or image change.

Time inside the run usually outweighs startup: on the same run a full state save every round cost about 46 s (about 38 min per
50 rounds) and each diagnostic round about 4 min. Both can be owner decisions or audit inputs, so report them rather than cut them.

## Pitfalls already paid for

- `multiprocessing.Connection.send(tensor)` uses torch's ForkingPickler, which shares storage back via
  `resource_sharer` and fails with AuthenticationError between independently started processes:
  send `pickle.dumps(...)` of CPU tensors or numpy with `send_bytes`.
- Per-tensor queue payloads exhaust file descriptors at many clients; flatten to one vector.
- Setup statistics computed for all clients in every process (moments, histograms) multiply host RAM
  by the process count; local tests may need fewer workers than production.
- In an FL harness, compile probes and warm-ups read only the process's own client rows.
- A progress log calling `float(loss)` every step is a sync; log every ~1,000 steps.
- Two processes may pick different cuDNN algorithms: pin each client's evaluation to one worker
  (client c on worker c % W) if per-client results must be reproducible.
- Do not compare one process's ms/step with a round's wall time that includes evaluation.
- Do not run another heavy job beside a multi-process local test; a RAM-guard kill looks like a bug.

## Optional example implementation

A complete per-GPU-worker FL harness for DAGSNet on VeReMi (global-model and per-client variants,
compile gates, tests) exists at https://github.com/odixe06/fl_dagsnet_veremi
(`lwfednids/papers/*/proj/driver.py`, `perfedskd/papers/*/proj/driver.py`). Reading it is optional;
this skill is self-contained.
