# Kaggle multi-account CLI operation

Use the existing authenticated CLI login. Read [kaggle-credentials.md](kaggle-credentials.md) for credential formats.
Keep account-specific state in the calling project's operations docs; inspect current identity/quota before acting.
MCP JSON, header synchronization and MCP reconnect are optional and are not part of this CLI workflow.

## Inspect and budget

```bash
python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py list
python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py quota
python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py health
python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py plan --gpu-hours TOTAL --session-hours CAP
```

These preserve the global login. `list` inventories saved OAuth snapshots and API-token files; a missing OAuth snapshot
is not a CLI failure. `quota/health/plan` support both types. Confirm identity by server introspection, not filename/public
metadata. Report account and error class only; exception bodies may expose secrets.
Budget the whole batch from `max(0, allowed - used - reserved)`, setup/compile/eval/I/O and margin.
Quota reads are not reservations; respect live session slots and owner authorization to launch.

## Scheduling short quotas

Start work now instead of waiting for a reset. Before planning, count both GPU session slots of every account; a second slot on a
high-quota account is free capacity, and a run does not need its whole budget in one account's current quota.

- `gpu_left` falls live while a session runs, and concurrent sessions on one account draw from it together. Size a second slot so the
  caps of both sessions stay under `gpu_left`.
- A session that reaches its cap with state saved ends COMPLETE. Set its cap from the account's quota today, then let a worker push
  `<slug>-c2` on the same owner once `gpu_left` covers the rounds left (rounds x measured round time + setup, x 1.15).
  Same-owner resume mounts the previous output, so no intermediate dataset is needed.
- Cross-account chains need a private dataset for the state (8.5 GB here) and are worth it only when the same owner cannot resume soon.
- Each extra session repeats startup; see the federated-training-speed skill, "Session startup".
- Keep full state on the Kaggle output rather than pulling it by default, and never delete those kernel versions: pull it on demand
  with `pull_kernel_output.py <ref> --pattern 'state(_rank1)?\.pt$'`. Only the latest round's state exists in each session output.

## Concurrent owner-scoped commands

```bash
python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py OWNER -- kaggle kernels status OWNER/SLUG/VERSION
```

Prefer this wrapper for status, push, output and dataset operations across accounts. It uses a saved OAuth snapshot
or API-token file in the child and leaves the global CLI login unchanged. A token file's legacy `.mcp-token` suffix does
not make it MCP-only. Never print its value or export a raw token globally.
Run one pull process per kernel with its owner; budget about32 total download connections. Pin exact versions and
require exit0 plus artifact inventory before audit. See [pull-outputs](pull-outputs.md).

## Global login changes

Switch only when necessary for an operation that cannot use the wrapper, under current owner authorization.
Announce the account, preserve the old snapshot and verify server identity after switching.
`use/ensure --confirm` require a valid target OAuth snapshot and work without MCP JSON. Optional MCP sync matters only
when explicitly configured. Prefer the wrapper to avoid these mutations.
For an existing successful login, do not request browser login again. `save USERNAME` saves its snapshot when needed.
For a new OAuth identity, `add-account USERNAME` snapshots the old login before the human browser step.
A forced login replaces a refresh token; never run it unattended. `refresh` renews the active identity without switching.
A generic network/permission error alone does not prove credential revocation.

## Inputs and continuation

The running account must read every input and own the notebook id. Group membership is not per-item access.
A saved credential does not authorize a launch. Cross-account staging needs source identity, config/data/source hashes,
verified full parent state and a require-resume gate.

## Cross-account resume

**Generator contract.** The generator must let the previous owner be named separately from
`--owner` (a `--kernel-source owner/slug` for the same account, a `--dataset-source owner/slug`
for a checkpoint dataset); a generator that derives every attachment from `--owner` cannot
split a scenario across accounts. Attaching another notebook's output alone imports nothing
unless the driver's resume gate discovers and verifies it — implement and test new-session
resume locally (crash replay, next-session resume, require-resume on an empty tree) before
attempting a handoff.

Once the actual driver's resume contract is verified:

1. Record the source owner, notebook version, last committed round, effective config, and
   artifact hashes. Confirm that the target account can read the full required resume bundle.
2. Attach notebook outputs using `kernel_sources` (checkpoint datasets use `dataset_sources`).
   Resolve a unique source recursively under `/kaggle/input`; support nested owner prefixes.
3. Require resume explicitly. Missing, ambiguous, incompatible or incomplete artifacts must
   fail before expensive preparation. Inference weights alone may omit mu, RNG or history.
4. Restore the last committed round and prove the next round is its successor. Compare all
   overlapping history and committed artifact hashes; matching a single F1 value is insufficient.
5. Preserve the authorized W&B run identity only for a verified continuation with monotonic
   steps. W&B `resume="allow"` does not load model weights. A deliberate new run or unverified
   restart needs its own identity and provenance; do not silently overwrite the original history.
6. Record which owner/session paid for which rounds and budget each continuation separately.

Monitor the exact session/version and fresh W&B step progress. Account-wide `time_reserved`
can corroborate accelerator activity but cannot tell which of several runs is alive or stopped.

## Handing a resumed run to another account

**`kernel_sources` does not cross accounts.** Measured 2026-09-09 with a dedicated CPU probe
(`minhtrit06` attaching a private notebook output owned by `khanhmay0304`):

```
The following are not valid kernel sources and could not be added to the kernel: [...]
Kernel version 1 successfully pushed.
```

Kaggle prints one warning line, **reports the push as successful**, and starts the kernel with an
**empty `/kaggle/input`**. There is no exception, no non-zero exit, nothing an exit-code check
would see. A multi-session chain that crosses an account boundary this way looks correct right up
until the session starts training from round 1 — or, with a resume guard, dies after the queue
wait for no visible reason.

Two consequences for planning:

* Decide the account boundary from *where the payload is smallest*, not from where the quota runs
  out. Moving the boundary one session earlier can cut the handoff from 43 rounds to 16.
* Verify the boundary with a **CPU probe before** the chain reaches it. GPU quota is the expensive
  resource; a probe that mounts the source and reads a few bytes costs none of it and answers the
  question hours or days before the chain gets there.

### The route that does work: a checkpoint dataset

Upload the previous session's run tree as a dataset **owned by the account that will run the next
session** — then nothing has to be shared or made public, and the run keeps its private weights
private. `find_import_source`-style discovery (`rglob("config.json")` filtered by run name and
fingerprint) does not care whether the tree arrived from a kernel output or a dataset mount.

Two defaults of `kaggle datasets create` will silently corrupt such a dataset:

| Default | What it does | Fix |
|---|---|---|
| `--dir-mode skip` | **ignores every directory**, uploading nothing but loose top-level files | `-r zip` |
| tabular conversion is ON | rewrites `*.csv` (and other tabular files) into Kaggle's own CSV | `-t` / `--keep-tabular` |

The second is the dangerous one: a rewritten `client_log/round_007.csv` still looks like a CSV, so
the upload and the mount both succeed, and the failure surfaces much later as an integrity-hash
mismatch mid-resume. Pass `-t` and then **compare sizes against the local originals** before
trusting the dataset.

Kaggle re-extracts the uploaded zip on its side and **drops one directory level**: a local
`runs/<run_name>/…` arrives as `<run_name>/…`. Harmless for rglob-based discovery; not harmless
for any code that pins the parent directory's name. The mechanism (seen 2026-09-14, a sibling rebuild, 100c):
`-r zip` turns each top-level directory of `-p <dir>` into `<that-dir>.zip` and Kaggle unpacks the
zip's *contents* at the dataset root — so the top-level directory's own name is what disappears.
Uploading `<dir>/<run_name>/…` therefore lands `weights/`, `complete/` … at the root with **no
`<run_name>` anywhere in the mount path**, and a discovery filter such as
`run_name in p.parts` finds nothing. Always stage one level deeper (`<dir>/runs/<run_name>/…`)
and check `kaggle datasets files` for the run name before pushing a GPU kernel. A wrong layout is
fixed with `kaggle datasets version -p <dir> -r zip -t -m "..."` on the same slug (new version
replaces the file set); a CPU probe of the gate is generated by the project's
`scripts/gen_ckpt_probe.py` — it embeds `proj/ckpt.py` and asserts the imported round.

**Measured end to end, 2026-09-11** (two run trees of 510 MB and 308 MB, `-r zip -t`): the
trees compressed about 5:1, the datasets were `ready` about a minute after upload, every file
matched the local tree by name *and* size (compare **all pages** of `datasets files`, not the
first; the CLI pages at 200), and the mount path was
`/kaggle/input/datasets/<owner>/<slug>/<run_name>/…` — note the extra `datasets/<owner>/`
level compared with the older `/kaggle/input/<slug>/`. Code that pins either layout breaks;
`rglob` for the per-round completion marker with the run name in the path survives both.

**Probe the gate itself, and assert the round, not just "found something".** A resume gate that
only checks `last is not None` has a hole: a dataset with one unreadable round in the middle
imports up to it, passes the gate, and the GPU session *retrains* the missing rounds under the
same run name — W&B drops the now non-monotonic steps silently, and only a later byte-level
merge notices. The CPU probe is where that is caught for free: run the production import
against the mounted dataset with the cfg from the run's own manifest, and **assert
`last == the round the previous session committed`** before pushing the GPU kernel. Keep the
probe as a generated notebook next to the training notebook's generator, embedding the same
checkpoint module the training notebook uses, so it cannot drift from the gate it tests. It
costs about half a minute per kernel and no GPU quota.

### The cheaper route: a handoff bundle (last round + sha chain)

The full-tree dataset above re-uploads every round the owner already holds locally, and the
next session's output then re-carries them, so every hop costs one upload and one download
of the whole history (a per-client method at 100c: 2.2 GB after 14 rounds, ~8 GB by round 50;
a global-model method is ~0.2 MB a round). Measured
2026-09-16: a full-tree upload ran at ~1.4 MB/s from the owner's machine.

What a continuation actually needs is one round: the weights it resumes from, that round's
resume/metrics/confusion/marker, and proof that the round is the product of the history it
claims. The projects here (`proj/ckpt.py`, `HANDOFF`) carry that proof as
`reports/handoff.json`: the sha256 of `weights/round_001..r.pt` taken from the full tree,
which the owner verifies locally (`last_complete_round` walks the `prev_sha` chain from
round 1) *before* exporting. The resume gate then anchors round r two ways — its bytes must
hash to `chain[r]` and its `prev_sha` must equal `chain[r-1]` — and refuses a bundle whose
chain, round or fingerprint does not match, or that would be spliced onto a tree with its own
rounds. Every session's output still holds a contiguous, verifiable tree; it just starts at r.

```
scripts/stage_ckpt_dataset.py <full run dir> --owner A --slug S --out DIR --last-only
kaggle_as.py A -- kaggle datasets create -p DIR -r zip -t        # ~35 MB (20c) / 170 MB (100c)
scripts/gen_ckpt_probe.py --owner A --dataset A/S --run-name R --expect-round r --out ...
gen_notebook.py --owner A ... --session N --require-resume --dataset-source A/S
```

Nothing is lost for the final record: the owner pulls every session and
`scripts/merge_sessions.py` unions the per-round files (byte-identical on every overlap, or it
refuses), rebuilds `history.csv`/`clients.csv` keyed by round, keeps each session's
`handoff.json` under `logs/sessions/<n>/`, and the merged directory reads as one run —
`verify_run --require-rounds 50` re-walks the whole chain there. Test:
`tests/test_ckpt_verify.py` cases 7–10 (bundle import, tampered chain, splice refusal, merge).

The same-account `--kernel-source` route keeps importing the whole attached tree; a project
that wants the smaller commit there too can stage a bundle for itself the same way.
