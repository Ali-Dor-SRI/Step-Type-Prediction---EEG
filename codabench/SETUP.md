# Setup — the CPU-only environment on this machine

How the competition environment is built, where everything lives, and how to
rebuild it. Built 2026-09-23.

## The machine

- Windows 11 desktop, i7-12700K (20 logical cores), 64 GB RAM,
  **no NVIDIA GPU** (Intel UHD 770 only), so training is CPU-only. That's
  allowed: training compute is uncapped, and scoring happens on the
  organisers' GPU.
- Power: AC sleep and hibernate are "never" (checked 2026-09-23), so overnight
  jobs aren't killed by idle sleep.

## Why WSL (Ubuntu) and not native Windows

`neuralbench/main.py` does `import resource`, a **Unix-only** module, so
NeuralBench training can't run on native Windows. The competition's
scoring image is Linux too. Everything runs inside **WSL2 Ubuntu 26.04**
(user `ali_d`).

## Where things live

| What | Where | Why there |
|---|---|---|
| This folder (docs, our solvers, scripts, config) | `C:\Users\Ali D\Documents\ML\codabench\` = WSL `~/codabench` (symlink) | tracked in git; edit from Windows or WSL |
| Official competition code | `codabench/2026-competition/` (git clone, not tracked by our repo) | benchopt runs from here |
| Python env | WSL `~/neuralbench/.venv` (Python 3.12) | a venv must sit on the Linux disk (fast, correct permissions) |
| Competition data (benchopt) | WSL `~/neuralbench/benchopt_data/neural_compet/` | large; kept off OneDrive and out of git |
| NeuralBench data / cache / results | WSL `~/neuralbench/{data,cache,results}` | same |
| NeuralBench config | `codabench/config/neuralbench_config.json`, symlinked from WSL `~/.neuralbench/config.json` | one tracked copy |
| Logs | `codabench/logs/` (gitignored) | |

From Windows Explorer, the WSL side is at `\\wsl.localhost\Ubuntu\home\ali_d\`.

**Don't** move the data or the venv into the OneDrive-synced `Documents` folder.
The datasets run to tens of GB (Stieger 2021 is large), OneDrive would try to sync
them, and file access across the Windows/WSL boundary is several times slower.

## Daily use

Open **Ubuntu** (Start menu) and run:

```bash
source ~/codabench/env.sh
```

That activates the venv, sets `NEURALBENCH_CONFIG` and `BENCHOPT_DATA_HOME`,
and moves into `~/codabench`. Then work from `2026-competition/`, e.g.:

```bash
cd 2026-competition
benchopt run tracks/bci_decoding -d Simulated -o "BCI-decoding[training=True]"
```

## What's installed (venv `~/neuralbench/.venv`)

| Package | Version | Note |
|---|---|---|
| Python | 3.12.14 | via `uv` (Ubuntu's system Python is 3.14) |
| torch / torchvision / torchaudio | 2.14.0+cpu / 0.29.0+cpu / 2.11.0+cpu | **CPU wheels** from the PyTorch CPU index |
| neuralbench / neuralset / neuralfetch | 0.3.1 | = the competition's pinned versions |
| exca | 0.5.29 | = pinned |
| benchopt | 1.10.0 | the competition runner |
| moabb | 1.7.2 | EEG datasets (pulls in `pyriemann`) |
| braindecode | 1.8.1 | |
| mne | 1.13.2 | |
| scikit-learn | 1.9.1 | |

## Rebuild from scratch

```bash
# 1. uv + Python 3.12 venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/neuralbench && cd ~/neuralbench && uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. CPU torch FIRST, so neuralbench doesn't pull the multi-GB CUDA build
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. the competition stack (pins from 2026-competition/requirements.txt)
uv pip install neuralbench==0.3.1 neuralset==0.3.1 neuralfetch==0.3.1 exca==0.5.29 \
               "benchopt>=1.10" moabb mne braindecode pandas scikit-learn

# 4. config + links
mkdir -p ~/.neuralbench ~/neuralbench/{data,cache,results}
ln -sfn "/mnt/c/Users/Ali D/Documents/ML/codabench" ~/codabench
ln -sfn ~/codabench/config/neuralbench_config.json ~/.neuralbench/config.json

# 5. competition code (see "symlink fix" below)
cd ~/codabench && git clone https://github.com/neural-interfaces26/2026-competition.git
```

Check: `python -c "import torch; print(torch.__version__)"` should end in `+cpu`.

## Gotchas hit during setup

1. **Symlinks in the competition repo (important).** The repo stores
   `tracks/*/benchmark_utils` and `codabench/phases/warmup/*/benchmark` as git
   symlinks. Cloned with *Windows* git (`core.symlinks=false`), they become
   one-line text files, and benchopt fails with
   `ModuleNotFoundError: No module named 'benchmark_utils'`. Fix, run in WSL:
   ```bash
   cd ~/codabench/2026-competition
   for t in tracks/*/; do rm -f "$t/benchmark_utils"; ln -s ../../benchmark_utils "$t/benchmark_utils"; done
   for f in codabench/phases/warmup/*/benchmark; do [ -L "$f" ] || { t=$(cat "$f"); rm "$f"; ln -s "$t" "$f"; }; done
   ```
   Re-run this after any `git pull` that touches those paths, or pull from WSL
   git, which keeps them as real symlinks.
2. **Stale weights across datasets.** Every local run of a solver shares
   `tracks/bci_decoding/outputs/<SolverName>/`. The upstream baselines load any
   `weights.pt` found there, even in a *training* run, so switching datasets
   (different channel count) crashes with `Error(s) in loading state_dict`.
   Our solvers skip saved weights when training. For the upstream ones,
   delete the folder first.
3. **HTTP/3 warning on downloads** (`MustDowngradeError ... HttpVersion.h3`).
   Harmless: urllib3 retries over HTTP/2 and the download proceeds.
4. **A single failed file breaks a dataset, and retries don't fix it.**
   On 2026-09-23 OSF returned a transient `500 Server Error` for Dreyer
   subject 59. moabb logged `Error reading subject 59` and moved on, then
   NeuralBench refused the dataset: `Dataset Dreyer2023Large is corrupted,
   expected 520 timelines but found 516`. Re-running `benchopt prepare`
   does **not** re-download (NeuralBench treats the download step as done). Fix:
   fetch the missing subject by hand, then prepare again:
   ```bash
   D=~/neuralbench/benchopt_data/neural_compet/moabb/Dreyer2023Large/download/MNE-dreyer2023-data
   cd $D && curl -sSL --retry 5 -o sub-59.zip "<the OSF URL from the 'Error reading subject' line>"
   python -c "import zipfile; zipfile.ZipFile('sub-59.zip').extractall('.')"
   bash ~/codabench/scripts/prepare_track2_data.sh dreyer2023
   ```
   The URL is printed in the prepare log (`grep "Error reading subject" ~/codabench/logs/prepare_track2_*.log`).
5. **Disk space is C:'s, not WSL's.** `df` inside WSL shows the virtual disk
   (~1 TB) but its file lives on C: (~106 GB free after Dreyer). Stieger
   2021 (399 GB) doesn't fit; `prepare_track2_data.sh` refuses it below
   450 GB free.
6. **`benchopt install`** would try to create a conda env. Don't use it here;
   the venv already has everything (install extras with `uv pip install`).
7. Don't pass `--config` or `NEURALBENCH_CONFIG` pointing at a missing file:
   check the first log lines of any run.

## Updating the competition code

```bash
cd ~/codabench/2026-competition && git pull   # WSL git keeps symlinks intact
```

Our solvers live outside the clone (`codabench/solvers/`), so a pull never
touches them. Re-check `requirements.txt` after a pull: if the pinned versions
change, match them in the venv.

## 2026-10-13: the 0.4.0 environment for the PROTEUS release (side by side with the one above)

The official training data (PROTEUS, NEMAR `nm000290`) was released on
2026-10-13 with neuralbench 0.4.0 and a new, streamed contract. A second
environment was set up next to the frozen 0.3.1 one; the setup above is kept
as it is (the old regression gate `scripts/train_sealed.sh` still runs there).
Entry point: `source ~/codabench/env040.sh` (prints the neuralbench version and
the working directory as proof of the environment). What exists, how to run it
and the gates: `PROTEUS_SETUP.md`.

| | Old (frozen) | New (use this for PROTEUS) |
|---|---|---|
| venv | `~/neuralbench/.venv` (neuralbench 0.3.1) | `~/neuralbench/.venv040`: Python 3.12.14 (uv 0.12.18), neuralbench / neuralset / neuralfetch / neuraltrain 0.4.0, benchopt 1.10.0, torch 2.14.1+cpu, scikit-learn 1.9.1, pyriemann 0.12, numpy 2.5.2, pandas 3.0.6, mne 1.13.2, moabb 1.7.2, exca 0.5.29. Full list: `config/venv040_freeze.txt` (`uv pip freeze`, 191 packages) |
| upstream code | `~/codabench/2026-competition` (clone, `main` at 1ff8ce3, 48 commits behind) | `~/neuralbench/competition-040`: a **detached git worktree of that clone** at `origin/main` 442edcd ("update image docker"). Its git data lives in the old clone (`git -C ~/codabench/2026-competition worktree list` shows both): deleting or re-cloning the old clone breaks the new setup. Never `git checkout / pull / reset / stash / fetch` in the worktree: a checkout turns the hand-made symlinks (below) back into text files |
| data home | `~/neuralbench/benchopt_data` | `~/neuralbench/benchopt_data_040`: `neural_compet/Dreyer2026Proteus/` (BIDS, 500 Hz EDF, 1.5 GB) and `neural_compet/cache/` (the 120 Hz extraction cache, 730 MB). Do not change loader settings: a new filter or rate re-extracts everything and no longer matches the server |
| entry | `source ~/codabench/env.sh` | `source ~/codabench/env040.sh` |
| task / dataset | `eeg/motor_imagery`, batches of 64 | `eeg/_motor_imagery_stream`, `-d "BCI[study=dreyer2026proteus]"`, test streamed one window at a time |
| scoring image | `lionelyneuro/neural-compet:v2` | `v3` (`codabench/competition_bci.yaml` upstream) |

What was run on 2026-10-13 (16:41 to 16:47 EDT; the brief's Appendix A, LOG.md "2026-10-13"):

1. `git fetch origin` in the old clone: `origin/main` = 442edcd.
2. `git worktree add --detach ~/neuralbench/competition-040 origin/main`;
   `uv venv --python 3.12 ~/neuralbench/.venv040`; CPU torch from the PyTorch
   CPU index; then `uv pip install -r requirements.txt` (the worktree's file).
   8 s, wheels cached.
3. `benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/training.yml`
   failed with `No module named 'benchmark_utils'`: the worktree was created by
   a git with `core.symlinks=false`, so its 8 git symlinks (`tracks/*/benchmark_utils`,
   `codabench/phases/warmup/*/benchmark`) were text files. Repaired with the
   loop below (same gotcha as item 1 above), in the worktree only.
4. The same prepare with `BENCHOPT_DATA_HOME=~/neuralbench/benchopt_data_040`:
   1.48 GB downloaded in 19 s, 112 runs extracted in 1 min 45 s
   (`~/neuralbench/logs040/prepare_proteus.log`).
5. Loader check: 41 x 480 windows at 120 Hz; train / val / test = 1,521 / 379 / 755.
6. Baseline smoke `benchopt run ... --config training.yml -s MeanLogReg --no-plot`:
   cell mean 0.395863 on 8 cells. It wrote `tracks/bci_decoding/outputs/benchopt_run_2026-10-13_16h46m29.parquet`
   and `tracks/bci_decoding/outputs/MeanLogReg/` inside the worktree (still there).

Symlink repair, worktree only (never in the old clone):

```bash
cd ~/neuralbench/competition-040 && git ls-files -s | awk '$1=="120000"{print $4}' | while read f; do [ -L "$f" ] || { t=$(cat "$f"); rm "$f"; ln -s "$t" "$f"; }; done
```

Checked on 2026-10-15: all 8 symlinks are real symlinks; `origin/main` has
since moved to a66bc23 (the worktree stays at 442edcd on purpose).
