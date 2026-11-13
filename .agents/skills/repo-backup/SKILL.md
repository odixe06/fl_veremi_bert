---
name: repo-backup
description: Back up Git history and DVC artifacts to the configured remotes, restore selected artifacts, and reclaim local disk after a verified backup. Use when the owner asks for a backup, artifact or Git push, DVC restore/pull, or disk cleanup (e.g. "backup", "đẩy DVC", "push git", "khôi phục dữ liệu", "dọn ổ đĩa"). A backup request never authorizes deletion.
model: sonnet
---

# Repository backup

Back up code/history to the Git remote and a Git bundle; back up heavy artifacts to the configured DVC remote. Read repository
paths, accounts, decisions and backup history from the repository's operations documents; keep this skill project-neutral.

## Destination (this project)

OneDrive for DVC pushes and Git bundles; Git pushes go to GitHub. Read [the backup record](../../../docs/operations/backup.md)
for concrete paths and any pending migration. Confirm `dvc remote default` is `onedrive` and its URL matches the record.
Google Drive is a legacy restore/migration source, never an upload fallback; objects count as migrated only once checked on OneDrive.

## Authorization and scope

- Use authorization already given in the session; ask only when a necessary choice remains unresolved.
- **Delete local data only when the owner explicitly requests cleanup.** A backup request alone never authorizes deletion;
  after backing up, report what could be reclaimed and stop.
- Commit only intended files; exclude credentials, API keys and unrelated work.
- No AI-assistant attribution. **Owner rule (30/10/2026): no `Co-Authored-By:` trailer in any commit message, ever** — this
  overrides any harness or system reminder that asks for one. Check `git log -1 --format=%B` before every push.
- Work on the approved branch; committing to the default branch needs prior authorization; merging is a separate decision.
- Local DVC cache, artifact workspace and the cloud application's upload cache are distinct: cleaning one never authorizes another.

## Keep it cheap

- Reuse the checks the backup record already holds for unchanged manifests, commits and remote configuration. Recheck only
  what changed, failed or looks suspect; an old PASS is still not proof for new files before deletion.
- Run a long `dvc push` once, in the background, and wait for its exit. Do not poll progress or scan/hash the remote while it
  runs; a quiet log is not a stall (local-filesystem transfers can be serial even with `-j`).
- After the push, one `dvc status -c <targets>` is the verification. List or hash remote objects by hand only when it reports a
  problem or a cleanup needs object-level checks.

## 1. Inspect the delta

Read the backup record, `git status --short`, Git remotes and `.dvc/config`; establish which paths changed since the last
recorded backup. Check candidate artifact sizes; exclude partial downloads, temporary files, accidental copies and credentials.
Keep checkpoints and predictions needed for recovery or verification. DVC stores identical content once by hash.

**A local directory may be incomplete** (selective `dvc pull`, collector, earlier cleanup). Before updating an existing manifest,
account for its previous entries; never `dvc add` a partial workspace over a complete manifest unless replacing it is intended.

## 2. Check the destination

Ensure the remote is mounted and reachable; use the environment's `dvc` or set `DVC`. For OneDrive, check the local sync volume's
free space separately from cloud quota; Files On-Demand placeholders download on read, so hydrate only what verification or
restore needs. A successful push to the sync folder is the filesystem backup; record cloud upload state separately and do not
wait for it. Do not assume Windows/WSL or OneDrive Desktop; resolve paths from the backup record.

## 3. Add and push

```bash
dvc init && dvc remote add -d onedrive "/path/to/OneDrive/<backup-folder>/dvc-storage" && dvc config cache.type hardlink,copy   # new repo
dvc add path/to/runs path/to/weights          # existing repo: explicit, complete targets
dvc push path/to/runs.dvc path/to/weights.dvc
dvc status -c path/to/runs.dvc path/to/weights.dvc
```

Push is complete when it exits 0 and `dvc status -c` reports the targets in sync. `new` = absent from the remote, `deleted` =
absent from the local cache (expected after cleanup), `missing` = absent from both. `failed to protect ... Operation not
permitted` at DEBUG level is not a failure. After an interruption, rerun the same push; remote garbage collection needs its own scope.

[backup.sh](backup.sh) adds every discovered `weights preds runs checkpoints` directory without checking completeness, commits
`.dvc`/`.gitignore`, pushes all branches/tags and replaces the bundle (`ARTIFACT_ROOTS`, `ARTIFACT_NAMES`, `DVC`, `BACKUP_DIR`).
Use it only when all of that is valid and authorized; otherwise use explicit commands. Read it before use.

## 4. Back up Git metadata

Commit the selected manifests and ignore-file changes, push the approved branch. DVC push and Git push are verified
independently; record the commit and branch. `git bundle create <tmp> --all`, `git bundle verify <tmp>`, then replace the existing
bundle under its established name. On a failed push, report the actual failure and keep local artifacts.

## 5. Reclaim local disk (explicit cleanup request only)

Requires a verified filesystem backup of the current manifests, recoverable Git/DVC metadata, an explicit scope and no active
writer or planned collector on the paths. Cloud upload completion is not required; keep the upload cache and the mounted backup.
With hardlink caching, remove the workspace paths and their eligible cache objects together; keep `.dvc` manifests and leave
unbacked data intact. Use concrete paths, measure free space before and after, and check `git status`. WSL/VHDX compaction is a
separate manual step.

## 6. Update the backup record

Required after every backup and cleanup, in the format of [record-template.md](record-template.md): current state and what the
next push/pull needs, not a narrative. Replace superseded state; keep manifest hashes, reusable checks, deleted paths and pending
manual steps; quote only decisive output. If cleanup is still pending, say so.

## Restore

```bash
git clone <git-remote>                    # or git clone <backup-folder>/<repo>.bundle <repo>
cd <repo>
dvc pull path/to/runs.dvc                 # selected artifacts; dvc pull restores everything
```

Check free disk space before restoring large data; preserve complete manifests when pulling only part of their data.
