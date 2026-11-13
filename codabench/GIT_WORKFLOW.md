# Code through Git; trained artifacts through NAS

Codabench source is part of the parent ML repository. The shared research
branch is `feat/codabench-track2` on the `personal` remote. The `org` remote
is a second copy and may lag; it is not the default source for updates.

## Retrieve code

From the ML checkout, on Windows or Mac:

```bash
python scripts/codabench_git.py fetch
python scripts/codabench_git.py pull
```

On a Mac without an active environment, use `python3` instead of `python`.
`fetch` downloads Git objects and updates the remote branch reference only.
`pull` also updates this checkout if it can fast-forward. It refuses tracked
local edits and divergent history; it never stashes or resets your work.
Local commits that already contain the fetched code are preserved.

To start a fresh code-only checkout:

```bash
git clone --branch feat/codabench-track2 https://github.com/Aria-Doroodchi/Step-Type-Prediction---EEG.git ML
```

Until the Mac setup branch is merged into the research branch, its setup
tools are available on the `org` remote's `codex/codabench-mac-setup` branch:

```bash
git clone --branch codex/codabench-mac-setup https://github.com/Ali-Dor-SRI/Step-Type-Prediction---EEG.git ML
```

These commands
retrieve committed and pushed code; Git cannot retrieve uncommitted files
from another computer.

## Publish code from either machine

Install the source-check hook once per checkout:

```bash
git config --local core.hooksPath .githooks
```

Commit the intended source/configuration/documentation paths, then push the
branch you are working on. For the existing Windows research branch:

```bash
git add codabench
python scripts/codabench_git.py check
git diff --cached --stat
git commit -m "codabench: describe the change"
git push personal HEAD
```

Review the staged list before committing. The existing ignore rules keep
submission archives, logs, datasets, environments and caches out of this
command. Existing explicitly tracked results under logs remain tracked;
add an individual new concise results document with `git add -f` only
after reviewing it. Do not force-add weights or arrays.

The pre-push hook detects untracked source in the main Codabench source
folders, missing tracked files, Python syntax errors, tracked model/data
artifacts, and tracked Codabench files larger than 10 MiB. It does not
automatically commit or push anything. The `Codabench source` GitHub Actions
job repeats the source checks and tests safe synchronization on pushes and
pull requests; it can also be started manually in GitHub Actions.

## NAS artifacts

Use the existing NAS root:

```text
Windows: Z:\Projects\codabench
UNC:     \\192.168.50.157\Beast_PC\Projects\codabench
Mac:     /Volumes/Beast_PC/Projects/codabench
```

Public data lives under `neural_compet/`; the PROTEUS window cache is
`proteus_cache/`. Windows-trained submissions and experiment artifacts are
archived under `windows-transfer/codabench/`, retaining their original
`submissions/` and `logs/sprint_*/` paths. Model weights (`.pt`, `.joblib`),
feature/probability arrays (`.npy`, `.npz`), and submission ZIPs belong on
NAS. The frozen B3 archive is not downloadable from Git.

Use `MAC_SETUP.md` for the native environment and data-validation commands.

The November 13 transfer is restored and checksum-verified; see
`RESUME_STATUS.md`. To connect its NAS artifacts to a fresh Mac/Linux checkout:

```bash
python3 scripts/link_codabench_artifacts.py
```

Use `--artifact-root /path/to/windows-transfer/codabench` for a different mount
location. The command reads the NAS import manifest, preserves tracked files,
backs up existing local artifacts before linking, and is safe to repeat.
