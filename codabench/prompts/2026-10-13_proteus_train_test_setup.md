# Setup session: a verified training and test configuration on PROTEUS

You are starting a fresh session on the Codabench Track 2 work in `codabench/`.
Time frame: **7 hours from the moment you start** (read the start with `date` in
WSL and write the absolute deadline into § 7 of the saved copy of this brief).
The user has asked for exactly this focus and has approved it: **configure the
training and test sets on the newly released official data and prove that every
piece of code around them works.** This is a setup and verification session.
It is not a modelling session: no tuning, no new features, no accuracy or speed
work. The user may be reachable, but do not wait for answers: collect questions
under "needs you" in the final summary. Work autonomously; decide each next
step from the previous result with the rules below; never let a job go stale
(§ 6). Read the whole brief before starting.

If you use the project-sprint skill, this text **is** the brief: do not write
another one, and do not let the value check re-scope the work towards model
performance. The skill's `references/project-map.md`, `CLAUDE.md`'s key
context, the `MEMORY.md` index line and `codabench/README.md` still describe
the old plan and the old environment. Where they disagree with § 1, § 1 wins.

Paths: `$COMPET_ROOT/…` is the organisers' code (`~/neuralbench/competition-040`);
`~/codabench/…` is our folder (the repo's `codabench/`). Both have a
`codabench/` sub-folder: do not mix them up.

## 1. Context (verified 2026-10-13, 16:40–17:15 EDT; re-checked by four reviewers 17:18–17:45)

**The official data is out and the contract changed.** Sources, at upstream
`origin/main` 442edcd: `$COMPET_ROOT/codabench/pages/scientific_faq_bci.md`,
`$COMPET_ROOT/tracks/bci_decoding/README.md` and `objective.py`,
`$COMPET_ROOT/benchmark_utils/streaming.py`.

- **Data:** PROTEUS, NEMAR `nm000290`, CC-BY. 10 people (P01–P10), 14 sessions,
  84 cued runs, **2,655 windows**. P04 and P10 have sessions 1–3; the other
  eight have session 1 only. Every session has 6 cued runs: runs 1–2 are
  calibration runs (sham feedback), runs 3–6 online runs. Each run uses one
  interface, Graz or BrainHero, in an order that differs between sessions.
- **Windows per session**, in the order P01, P02, P03, P04/1, P04/2, P04/3, P05,
  P06, P07, P08, P09, P10/1, P10/2, P10/3:
  190, 189, 189, 189, 189, 188, 190, 187, 189, 189, 189, 192, 186, 199.
- **Classes:** 0 = `mi` (885), 1 = `sub` (886), 2 = `word` (884).
- **What a model receives:** one tensor `(B, 41, 480)`, float32, **microvolts**,
  120 Hz, 4 s starting 1 s after the cue, notch 50/60 Hz, 0.1–75 Hz.
  **No scaling and no clamping**: 925 windows (34.8 %) exceed 200 µV in
  absolute value somewhere; the extremes are −4,417.8 and +3,511.5 µV. EEG
  only: no EMG/EOG, ever.
- **Channel order** (41): C2, CP2, P2, Pz, CPz, Cz, C1, CP1, P1, P3, PO7, P5, TP7,
  CP5, CP3, C3, PO8, P6, P4, TP8, CP6, CP4, C6, C4, FC6, FC4, FC2, F6, F4, F2, AF4,
  AF3, F5, F3, F1, Fz, FC5, FC3, FC1, FCz, C5.
- **Evaluation is streamed.** `predict` is called on **one window at a time**
  (`B = 1`), in recording order (runs by run number, then time), one stream per
  person-session. Each stream runs on a fresh `copy.deepcopy` of the model,
  whose optional `reset_state()` is called first. The model gets the tensor
  only: no person, session, run or interface.
- **Adapting inside a session without labels is allowed.** Nothing may cross a
  session boundary.
- **Ranking metric:** balanced accuracy within each person × session ×
  interface cell, averaged over cells (`objective_balanced_accuracy_cell_mean`).
  Pooled balanced accuracy and accuracy are diagnostics.
- **Sealed test (Oct 28 – Nov 21):** later sessions of these 10 people **plus 10
  people absent from the release**. No calibration data for anyone. 1 upload
  per day, 10 in total; the ranking takes the best of a team's last five
  sealed uploads; 3,600 s per run; the top 3 are audited by re-running
  training. Phase ends are 23:59:59 UTC in `competition_bci.yaml`.
- **Warm-up (ends Oct 25, 19:59 EDT; 5 uploads/day; does not rank):** now
  scored on PROTEUS with the organisers' split: train P01 P02 P03 P06 P07 P10
  (1,521 windows), validation P05 P08 (379, never given to a solver), **test
  P04 × 3 sessions + P09 (755 windows, 4 streams, 8 cells)**. The test labels
  are public. The server runs `BCI[study=dreyer2026proteus,subset=full]` with
  seed 42 on the image `lionelyneuro/neural-compet:v3` (it was `v2` at our
  2026-10-02 deployment test).

**Two environments exist side by side in WSL. Use only the new one.**

| | Old (frozen; do not use, do not edit) | New (use this) |
|---|---|---|
| venv | `~/neuralbench/.venv` (neuralbench 0.3.1) | `~/neuralbench/.venv040` (neuralbench 0.4.0, benchopt 1.10.0, scikit-learn 1.9.1, pyriemann 0.12, torch 2.14.1+cpu) |
| upstream code | `~/codabench/2026-competition` (1ff8ce3, 48 commits behind) | `~/neuralbench/competition-040` (a detached git worktree of the old clone, at 442edcd) |
| data home | `~/neuralbench/benchopt_data` | `~/neuralbench/benchopt_data_040` (PROTEUS 1.5 GB + 730 MB cache) |
| entry | `source ~/codabench/env.sh` | the block below; Phase 1 turns it into `codabench/env040.sh` |
| task / dataset | `eeg/motor_imagery`, batches of 64 | `eeg/_motor_imagery_stream`, `-d "BCI[study=dreyer2026proteus]"` |

The activation recipe for everything except git (one Bash-tool call per block;
`exec 2>&1` first):

```bash
wsl.exe bash -s <<'EOF'
exec 2>&1
export PATH="$HOME/.local/bin:$PATH"
source ~/neuralbench/.venv040/bin/activate
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore::FutureWarning MNE_LOGGING_LEVEL=ERROR
export NEURALBENCH_CONFIG="$HOME/codabench/config/neuralbench_config.json"
export BENCHOPT_DATA_HOME="$HOME/neuralbench/benchopt_data_040"
export BENCHOPT_DEBUG=true
export COMPET_ROOT="$HOME/neuralbench/competition-040"
export PROTEUS_CACHE_ROOT="$HOME/neuralbench/proteus_cache"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$COMPET_ROOT"
# commands
EOF
```

**Read first (10 minutes at most).** Ours: `~/codabench/DIRECTIONS.md` § 1–2
and 5–6; `~/codabench/SETUP.md` (gotchas); `~/codabench/datasets/mock_sealed.py`
(the template for a benchopt `Dataset` of our own);
`~/codabench/prompts/2026-10-01_dual_lda_strict_wcv.md` only as the model for
how this repo logs and hands off. Theirs: `$COMPET_ROOT/codabench/pages/scientific_faq_bci.md`
and `participate.md`, `$COMPET_ROOT/tracks/bci_decoding/objective.py` and
`datasets/bci_studies.py`, `$COMPET_ROOT/benchmark_utils/streaming.py`,
`data.py`, `nb_task.py`, `base_solver.py`. `RELEASE_DAY.md`, `SEALED_RECIPE.md`
§ 1/3/4, `TRACK2_BCI.md`, `SUBMISSION_GUIDE.md` and `COMPETITION.md` describe
the old contract: do not follow them.

**Already measured, do not repeat:**
- `benchopt prepare` for PROTEUS is done. Nothing needs downloading.
- The official loader: `load_task` about 5 s on the warm cache, a pass over all
  windows about 3 s.
- The organisers' `MeanLogReg` with the Phase 1 command: cell mean
  **0.395863**, session mean 0.395465, pooled 0.395672, accuracy 0.396026
  (299/755), `n_cells` 8. Run three times on 10-13: bit-identical, 14 s.
  Phase 1 re-runs it once so that its export and parquet sit in our log
  folder (G1); that is the only repeat.
- **`MeanLogReg` depends on the order of the training rows** (lbfgs stops at
  its tolerance). Ten row orders gave cell means 0.393175–0.397165; `--seed 0`
  gives 0.397165; cache order gives 0.395821. So 0.395863 belongs to the
  Phase 1 command only. Never compare two trainings made through different
  loaders: exact gates compare **one exported model** across routes.
- The Riemann solver on 120 PROTEUS windows, `personal="pooled"`,
  `align="none"`: fit 5.9 s on microvolt input (4,646 features), a repeated
  fit bit-identical, `B = 1` against one batch 2.4e-13, about 8.5 ms per
  window at `B = 1`.

**Verified in the code (2026-10-13), with what each one means for you:**
1. **Our solver's inference already runs one window at a time.**
   `RiemannSealedModel` (`~/codabench/solvers/bci_decoding/riemann_sealed.py:1134`)
   is deep-copyable and `predict` on `(1, C, T)` returns a `(1,)` int64 tensor.
   It has no `reset_state()`. With `adapt="none"` it keeps no state that
   affects predictions (only the `_pred` timing counters).
2. **It was never fitted on the full PROTEUS data.** Every existing joblib and
   zip comes from the old loader (other channel sets, scaled input) and is
   unusable here.
3. **Its person-specific paths are wrong for unseen people, silently.** With
   the file's defaults (`personal="blend"`, `align="subject"`) the router
   sends an unseen person to some training person and applies that person's
   whitening and personal LDA. No error, no warning (`:1204-1212`,
   `:1625-1634`). With `personal="pooled"` and `align="none"` the router still
   runs, but its result cannot change the output (checked: difference 0.0).
4. **Codabench runs a solver file with its default parameters.** Settings that
   differ from the defaults must be baked into a candidate copy; `predict`
   takes `personal`, `adapt`, `buffer` and a numeric `blend_w` from the
   defaults, not from the joblib (`:1670-1701`).
5. **Its `fit` reads `info["subject_id"]`** (`:1255`, `:1294`). The organisers'
   loaders give `info` = `subject_id`, `record_id`, `onset` (float seconds),
   `stream_id`, `context_id`. Upstream's `benchmark_utils.data.make_loader`
   gives no `subject_id`, casts `onset` to int64, and needs a `record_id` that
   is unique over the whole loader (a bare run number raises "Recording …
   resumes").
6. **Every existing script is wired to the old environment.**
   `scripts/sealed_lib.sh:9`, `train_sealed.sh:137`,
   `test_track2_solvers.sh:25`, `prepare_track2_data.sh:18` and
   `install_xsess_overlays.sh:8` source `env.sh`; `analysis/xsess_cache.py:55`
   and `xsess_lib.py:56` put the old clone on `sys.path`. Calling any of them
   from the new environment silently switches back. The old harness also
   mis-scores PROTEUS without failing (wrong default split, 14 cells instead
   of 28, batches only): do not use it.
7. **The integer ids in `info` are not stable keys.** `context_id` and
   `record_id` are numbered by first appearance in each loader (0 = Graz in
   train and validation, 0 = BrainHero in test); `stream_id` is 3, 4, 5, 10 in
   the official test loader. **Stable keys** are the trigger-table columns:
   `subject` (`'Dreyer2026Proteus/P04'`), `session` and `run` (strings: use
   `int()`), `task` (maps to the interface with `_PROTEUS_INTERFACES` in
   `$COMPET_ROOT/tracks/bci_decoding/datasets/bci_studies.py:50-55`) and
   `start` (the cue; the window starts 1.0 s later and `info["onset"]` is that
   start). (`subject`, `session`, `run`, `start`) is unique over all 2,655
   windows.
8. **How to read a loader with its keys:** `ds = loader.dataset`; read item `i`
   as `ds[i]` for `i in range(len(ds))`, not by iterating the loader (the
   train loader is shuffled). Its key is
   `ds.seg_ds.triggers.iloc[int(ds.index[i])]`: positional, because the
   table's index labels do not start at 0. For validation and test this order
   is the order in which the loader yields windows.
9. **benchopt exposes scores only.** A run prints no score and no data shape;
   the six `objective_*` scalars are in a parquet. Without `--output` the
   parquet lands inside `$COMPET_ROOT/tracks/bci_decoding/outputs/`. Without
   `--seed` an arbitrary seed is used. An inference-only run whose
   `COMPET_SUBMISSION_DIR` holds no weights makes `MeanLogReg` predict class 0
   without a warning (cell mean 0.333333).
10. **A `Dataset` file of ours can be passed by path with parameters:**
    `-d "$HOME/codabench/datasets/proteus_protocols.py[protocol=lopo,fold=3]"`
    (verified on benchopt 1.10.0; `protocol` arrives as a string, `fold` as an
    int; a list such as `fold=[0,1]` runs both in one call).
11. **Upstream helpers you can call directly:** `benchmark_utils.data.make_loader`,
    `benchmark_utils.streaming.predict_streams`,
    `benchmark_utils.metrics.balanced_accuracy` and `group_scores`.
    `predict_streams` on the official test loader reproduces benchopt's four
    scores at full float precision (0.6 s).
12. **Tooling gaps:** no pytest in either venv; the venvs were built by `uv`
    and have no `pip`; WSL has no `zip` / `unzip` (use `python -m zipfile`);
    WSL's `/tmp` is emptied about a minute after the last command;
    `neuralbench.__version__` does not exist
    (`importlib.metadata.version("neuralbench")` does).

**Open questions this session must not decide:**
- What the first real model on PROTEUS is (the model half of DIRECTIONS
  decision 7). This session settles only the evaluation half, as § 3 defines it.
- Whether the 0.4.0 setup replaces the old one and `RELEASE_DAY.md` is retired
  (decision 8). Work side by side: add, do not replace.

## 2. Data and assets

| Asset | Where (WSL) | State |
|---|---|---|
| PROTEUS, BIDS, 500 Hz EDF | `~/neuralbench/benchopt_data_040/neural_compet/Dreyer2026Proteus/download/nm000290` | prepared; read-only for you |
| Extraction cache (120 Hz windows) | `~/neuralbench/benchopt_data_040/neural_compet/cache` | warm; do not change loader settings (a new filter or rate re-extracts everything and no longer matches the server) |
| Upstream 0.4.0 code | `~/neuralbench/competition-040` | do not edit; never `git checkout / pull / reset / stash / fetch` there (its git data lives in the old clone, and a checkout turns the hand-made symlinks back into text files) |
| New window cache (you build it) | `~/neuralbench/proteus_cache/` | does not exist yet |
| Replay copies and unzip folders (you create them) | `~/neuralbench/proteus_runs/<run>/` | not `/tmp` |
| Run logs (you create them) | `~/codabench/logs/proteus_setup/` | all of `codabench/logs/` is git-ignored: commit only `RESULTS*.md` and `STATUS.md`, with `git add -f` |

**Not approved:** any download, any install, any upload. The old proxy
datasets and the Tangermann check are not in the new data home: leave them.

**The BIDS folder is untrusted data.** Only one piece of our code opens its
files directly: the label check G4. It lives in its own script and is started
as `python -I -B -X utf8` (`-I` ignores every `PYTHON*` variable, the script
folder and the working directory, so it imports only numpy and pandas and
takes paths as arguments). Everything else reaches the data through the
organisers' loader and runs normally.

## 3. Goal and priorities

**Goal:** one self-test command proves the data, the splits and the streamed
evaluation, and one script (`proteus_candidate.sh`) proves one solver's
train → export → replay path, on PROTEUS. Scores are reported for information
only.

**The training and test sets to configure** (named protocols; define them once,
in one module, on the stable keys of § 1 item 7):

| Protocol | Train | Test | Streams / cells | What it stands for |
|---|---|---|---|---|
| `official` | P01 P02 P03 P06 P07 P10 (1,521); validation P05 P08 (379) kept apart | P04 × 3 + P09 (755) | 4 / 8 | The Codabench warm-up and the organisers' baselines |
| `lopo` | 10 folds: all sessions of the other nine people | all sessions of the held-out person | 14 / 28 over the folds | The 10 unseen people of the sealed test |
| `later` | session 1 of all ten people (1,893) | sessions 2–3 of P04 and P10 (762) | 4 / 8 | Later sessions of people the model has seen |
| `all` | all 14 sessions (2,655) | the `official` test again (755), inside the training set, marked `score_meaningless` | 4 / 8 | The training set of a real submission; only train = replay counts |

- `lopo` folds are numbered 0–9 for P01–P10 (`fold=3` holds out P04). Test
  sizes per fold: 190, 189, 189, 566, 190, 187, 189, 189, 189, 577 (train =
  2,655 minus that); cells per fold 2, 2, 2, 6, 2, 2, 2, 2, 2, 6.
- `lopo` is scored per fold. Its figure is the mean over the 28 cells of all
  folds (the sum over folds of cells × cell mean, divided by 28); report the
  mean of the ten fold means beside it.
- No protocol has a validation split: nothing is tuned in this session.
- Do not combine `lopo` and `later` into one number.

**Priorities, in order:**
1. **A reproducible entry point** for the new environment, and the missing
   record of the 2026-10-13 setup (Appendix A has what was run).
2. **One window cache** built from the official loader, proven identical to it.
3. **The four protocols**, proven leak-free and correctly ordered.
4. **A streamed local evaluation** that calls the organisers' own evaluator and
   metric, proven equal to the benchopt route.
5. **One solver through the whole path** (fit, export, read-only replay, zip)
   on PROTEUS, in its simplest valid configuration.
6. **A reference table** for the four protocols and the docs that let the next
   session start from here.

**Deliverables** (new files unless marked):
- `codabench/env040.sh`, `codabench/config/venv040_freeze.txt`
- `codabench/analysis/proteus_cache.py`, `proteus_splits.py`,
  `proteus_eval.py`, `proteus_g4_events.py`, `proteus_selftest.py`
- `codabench/datasets/proteus_protocols.py` (benchopt `Dataset`, used by path)
- `codabench/scripts/proteus_candidate.sh` (bake → train → replay → equal → zip)
- `codabench/solvers/bci_decoding/riemann_sealed.py` (edited: `reset_state()`
  only) and one baked candidate copy next to it
- `codabench/PROTEUS_SETUP.md` (what exists, how to run it, the gate table)
- `codabench/logs/proteus_setup/RESULTS.md`, and the doc updates of Phase 7

**Not this session:** tuning of any kind; new features or preprocessing
(including amplitude clipping); online re-centring design; speed-ups of the
solver; EEGNet or foundation models; porting the old ablation harness
(`sealed_personal.py`, `sealed_run.py`, `release_*`); rewriting the stale
docs; uploading.

## 4. Rules

- **Git runs on the Windows side.** Run every git command for our repo from
  the Bash tool directly (Git Bash, repo root `C:\Users\Ali D\Documents\ML`),
  never inside a `wsl.exe` block: WSL git lists about 430 CRLF-only "modified"
  files there and cannot push. Branch `feat/codabench-track2`; subject prefix
  `codabench:`; add paths explicitly; push to `personal` after every commit
  (OneDrive has rolled `.git` back before). No push to `org`, no tags, merges
  or PRs. If you use sub-agents, only the main loop commits.
- **Not yours to stage:** the edit in `codabench/README.md`,
  `codabench/MODEL_ARCHITECTURE.md`, `codabench/analysis/arch_figures.py`,
  `codabench/figures/` (another session's work). Never `git add -A`.
  The uncommitted `codabench/DIRECTIONS.md` edit **is** ours: commit it in Phase 0.
- **Frozen:** `codabench/submissions/`, `solvers/bci_decoding/eegnet_steptype_wu1.py`,
  `env.sh`, the old venv, the old clone, `~/neuralbench/benchopt_data`,
  `~/neuralbench/xsess_cache`, and the new worktree
  `~/neuralbench/competition-040` (no edit, and no `git checkout / pull /
  reset / stash / fetch` there; only the read-only queries of Phase 0).
  Existing solver and script defaults stay bit-identical: new behaviour goes
  into new files or is opt-in.
- **Never upload.** A finished zip gets a "ready, not uploaded" row in
  `SUBMISSIONS.md`. Uploading is the user's action.
- **Prove the environment in every run.** `env040.sh` prints the neuralbench
  version and the working directory (never source it with `>/dev/null`). Our
  own scripts and our `Dataset` print the protocol, the fold and the training
  and test shapes in their first line. For a bare benchopt run on the
  organisers' dataset the evidence is the parquet's `dataset_name` and
  `objective_n_cells`.
- **benchopt, every run:** `--seed 42`; `--no-cache --no-plot --no-html
  --no-display`; `--output <absolute path under the run's log folder>.parquet`;
  solver and dataset files by absolute path; a unique `Solver.name` per
  variant; one solver and one fold per call. **Training:** a fresh, absolute
  `COMPET_SUBMISSION_DIR` (never the folder that holds the solver file being
  trained). **Inference-only:** `COMPET_SUBMISSION_DIR` points at the exported
  folder or its read-only copy, and the log must show that the weights were
  used: `[loading] … weights.joblib into MeanLogReg` for `MeanLogReg`; for
  `Riemann-Sealed`, its `[Riemann-Sealed] predict: … calls` lines (without its
  joblib that model raises on the first window). **The one exception** to
  these flags is the Phase 1 reference command: run it exactly as written.
  The line `fatal: No names found` in every log is harmless.
- **Gates are exact comparisons**, not score thresholds. A gate that fails is
  a finding: give it 20 minutes of diagnosis, then either fix the cause or
  record FAIL with what you learned and stop only the work that depends on it.
  Do not loosen a gate to make it pass.
- **Numbers are labelled.** Every score in a log or doc says which protocol,
  which model, which loader and seed trained it, and "for information only".
- **Tests are plain-assert Python scripts** (house style:
  `analysis/xfeat_selftest.py`). New scripts start with
  `sys.path.insert(0, os.environ["COMPET_ROOT"] + "/tracks/bci_decoding")` and
  the path of `~/codabench/analysis`.
- **Running jobs:** while a background job runs, do not edit a script, module
  or solver file it uses; apply fixes after it ends, then re-run what they touch.
- **Shell:** scripts keep LF endings (inside WSL `grep -c $'\r' <file>` prints
  0; on the Windows side use `grep -cP '\r' <file>`); write
  files with the Write tool, not heredocs with non-ASCII text; never drive WSL
  from PowerShell; never `nohup … &`.

## 5. Phases and gates

Before each phase: state the estimate. After it: time-log it from the log
stamps, record each gate as PASS or FAIL with the measured value in
`logs/proteus_setup/RESULTS.md` (a gate with several checks gets one row per
check), append a `LOG.md` entry, update the state table in `HANDOFF.md`,
commit and push.

### Phase 0: orient (est. 15 min)

- Save this brief as `codabench/prompts/2026-10-13_proteus_train_test_setup.md`
  (it may already be there, uncommitted; the pasted text wins) and write the
  absolute deadline into § 7.
- Windows side: `git status --short` should show ` M codabench/DIRECTIONS.md`,
  ` M codabench/README.md`, `?? codabench/MODEL_ARCHITECTURE.md`,
  `?? codabench/analysis/arch_figures.py`, `?? codabench/figures/` and this
  brief; `git log --oneline -1` is 2765619 unless the user committed since.
- WSL: nothing of ours running
  (`pgrep -af 'benchopt|train_sealed|proteus_' || echo none`); the local
  upstream head is 442edcd (`git -C ~/neuralbench/competition-040 log
  --oneline -1`); whether upstream moved
  (`git -C ~/neuralbench/competition-040 ls-remote origin refs/heads/main`: a
  read-only query; if it no longer starts with 442edcd, note it under "needs
  you" and continue on 442edcd); the symlink
  `~/neuralbench/competition-040/tracks/bci_decoding/benchmark_utils` exists.
- Add a top section to `HANDOFF.md` that points at this brief. Commit the
  brief, `HANDOFF.md` and the pending `codabench/DIRECTIONS.md` edit.

**Done when:** pushed.

### Phase 1: entry point and reference run (est. 30 min)

- Write `codabench/env040.sh` from the block in § 1. It also prints
  `python -c "import importlib.metadata as m, os; print('neuralbench', m.version('neuralbench'), 'cwd', os.getcwd())"`.
  From here on every WSL call starts with `source ~/codabench/env040.sh`
  (the one exception is G11).
- Re-run the organisers' baseline as the environment's reference:
  ```bash
  REF=$HOME/codabench/logs/proteus_setup/ref_meanlogreg; mkdir -p $REF
  COMPET_SUBMISSION_DIR=$REF/submission benchopt run tracks/bci_decoding \
      --config tracks/bci_decoding/training.yml -s MeanLogReg \
      --no-cache --no-plot --no-html --no-display --output $REF/train.parquet
  ```
  (`training.yml` sets seed 42, the PROTEUS dataset and `training=True`.)
- Save `uv pip freeze` of `.venv040` as `codabench/config/venv040_freeze.txt`
  (a read, not an install).
- Write the 2026-10-13 setup into `SETUP.md` as a new section, from Appendix A
  and from what can be read back (`pyvenv.cfg`, the freeze, `git -C
  ~/codabench/2026-competition worktree list`, the symlinks). Say that the old
  clone owns the worktree: deleting or re-cloning it breaks the new setup.
  Back-fill one `LOG.md` entry for that setup with the times of Appendix A.

- **G0 (environment):** `importlib.metadata.version("neuralbench") == "0.4.0"`;
  `benchmark_utils.__file__` resolves under `competition-040`; `sys.prefix`
  ends with `.venv040`; `BENCHOPT_DATA_HOME` ends with `benchopt_data_040`.
  This is also the first check of the self-test.
- **G1 (reference):** the `objective_*` columns of `$REF/train.parquet` are
  cell mean 0.395863, session mean 0.395465, pooled 0.395672, accuracy
  0.396026 (to 6 decimals) and `n_cells` 8, and `$REF/submission/` holds
  `submission.py` and `weights.joblib`.

### Phase 2: the window cache (est. 45 min)

Build one cache of all 2,655 windows by calling the official loader exactly as
`bci_studies.py` does (`load_task("eeg", "_motor_imagery_stream", dataset=None,
data_dir=…/neural_compet, device="cpu", batch_size=64, seed=0, num_workers=0,
subset="full", target_transform=lambda y: y.argmax(-1),
context_of=_proteus_interface, order_of=_proteus_run)`) and reading the three
datasets item by item (§ 1 item 8). Store per window: the tensor, the label,
the stable keys (person, session, run, interface, task, cue time, window
start), the organisers' split, and the SHA-1 of the tensor. Store in
`meta.json`: channel names, units, sampling rate, the neuralbench version, the
upstream commit, the task name, the build time. Do not extend `xsess_cache.py`.

- **G2 (content):** 2,655 windows of shape (41, 480), float32, no NaN; classes
  885 / 886 / 884; 10 people, 14 sessions, 84 runs; the per-session counts of
  § 1; 28 person × session × interface cells (Graz cells have exactly 96
  windows, BrainHero cells 90–103); every cell and every run holds all three
  classes; the organisers' split 1,521 / 379 / 755; channel names equal to § 1
  in order; window start = cue + 1.000 s; the 2,655 SHA-1 hashes are distinct;
  unit check on absolute values: 925 of 2,655 windows have |x| > 200
  somewhere; minimum −4,417.8 and maximum 3,511.5 (to 1 decimal).
- **G3 (equal to the loader):** for every item of every official dataset, the
  cached tensor found by its key is identical (maximum absolute difference
  0.0) and the label is equal; `info["onset"]` equals `start + 1.0`; the label
  equals the table's `code`.
- **G4 (labels against the raw files, both directions):** every window's label
  agrees with the `trial_type` of its cue row (`mi` → 0, `sub` → 1, `word` → 2)
  in that run's BIDS `*_events.tsv`, matched on `round(onset × 1000)` ms; and
  every cue row of the 84 run files has exactly one window (2,655 = 2,655).
  Own script, `python -I -B -X utf8` (§ 2). In those files `end_trial` rows
  with value 32770 are run-end markers and BrainHero runs have no beep rows.

### Phase 3: the protocols (est. 45 min)

Implement the four protocols of § 3 as functions that return, for a protocol
(and a fold for `lopo`), the training indices and the test indices in stream
order: streams one after another, inside a stream by run number, then cue
time. An unknown protocol or an out-of-range fold raises.

- **G5a (sizes):** every number in the § 3 table and its bullets.
- **G5b (no leaks):** training and test index sets are disjoint in `official`,
  `later` and every `lopo` fold. `official` and `lopo`: no test person in
  training. `later`: no (person, session) pair on both sides, and every test
  person has session 1 in training. `all` is the stated exception: assert that
  its 755 test indices are all inside its training set and that it carries
  `score_meaningless`.
- **G5c (order):** in every protocol and fold, each test stream is ordered by
  (run number, cue time). For `official` the whole test key sequence equals
  the organisers' test dataset order, element by element, and the P05 and P08
  streams equal the organisers' validation dataset order.
- **G5d (ids and cells):** the integer ids we hand to an evaluator come from
  the stable keys, in one function: `stream_id` = rank of (person, session),
  `record_id` = rank of (person, session, run), `context_id` = 0 Graz, 1
  BrainHero. They are never copied from an official loader's `info`. Grouping
  a test set by (`stream_id`, `context_id`) equals grouping it by (person,
  session, interface), checked both ways; the cell counts are 8, 28 (2 or 6
  per fold), 8 and 8.

### Phase 4: streamed evaluation, two routes that must agree (est. 90 min)

**Route A, the harness** (`analysis/proteus_eval.py`). Wrap test rows with
`make_loader(X, y, batch_size=1, shuffle=False, record_id=…, onset=…,
stream_id=…, context_id=…)`, where `onset` is the window start as a sample
index (`round(start_s × 120)`), and score a model with the organisers'
`predict_streams`, `balanced_accuracy` and `group_scores`. Do not re-implement
them. Map results back to cache rows by position, never by a returned `onset`.

**Loading a submission folder the way the platform does** (also in
`proteus_eval.py`; this is where per-window predictions come from, since
benchopt gives scores only): import `<folder>/submission.py` by path, set
`COMPET_SUBMISSION_DIR=<folder>`, `solver = mod.Solver.get_instance()` (the
file's default parameters, as on Codabench),
`solver.set_objective(train_loader=None, **meta)` with `meta` = the dataset's
dict without the two loaders, `model = solver.get_result()["model"]`, then
`predict_streams(model, loader)`. Save `y_pred` as `.npy`. In-process, the
organisers' own `Dataset.get_data()` cannot be called (it needs a running
benchmark): take their loaders and `meta` from `load_task` as in Phase 2 and
build the dict as `bci_studies.py:157-165` does
(`n_classes = int(meta.pop("raw_target_shape")[-1])`, drop `target_shape`).

**Route B, benchopt** (`datasets/proteus_protocols.py`). A `Dataset` that
serves any protocol and fold from the cache, selected as in § 1 item 10, with
`mock_sealed.py` as the template. Requirements:
- Both loaders carry the organisers' five `info` keys: `subject_id` (one
  integer per person), `record_id`, `onset`, `stream_id`, `context_id`. Use a
  small Dataset class of our own for this (as `_MockWindows` and `_collate`
  in `mock_sealed.py:73-97`); `make_loader` cannot serve the training side.
- Test loader: batch size 1, unshuffled. Train loader: batch size 64, shuffled
  by `torch.Generator().manual_seed(seed)`, with the seed read as
  `mock_sealed.py:146-149` does (`try: seed = self.get_seed()` /
  `except Exception: seed = 0`). Outside a benchopt run `get_seed()` raises
  (`run_context was not initialized`), and G5e, G6 and G7 call
  `Dataset.get_instance(protocol=…, fold=…).get_data()` in-process.
- The other returned values equal the organisers' `get_data()` exactly:
  `n_classes` 3, `sfreq` 120.0, `ch_names`, `chs_info`, `n_chans` 41,
  `n_times` 480, `device`.
- `get_data` prints the protocol, the fold and both shapes in its first line.

Gates:
- **G5e (what the loaders serve), every protocol and fold, in-process on route
  B's `get_data()`:** one pass over both loaders. The set and count of SHA-1
  hashes served equal the protocol's index sets; every served label equals the
  cache label of its hash; training and test hashes are disjoint (for `all`:
  the test hashes are all inside training); the test loader's hash sequence
  equals the protocol's test order; the other values `get_data()` returns
  (`n_classes`, `sfreq`, `ch_names`, `chs_info`, `n_chans`, `n_times`,
  `device`) equal the organisers'.
- **G6 (routes agree), with the `MeanLogReg` exported in Phase 1:**
  - through the CLI, inference-only, twice:
    `COMPET_SUBMISSION_DIR=$REF/submission benchopt run tracks/bci_decoding -o "BCI-decoding[training=False]" -d "<dataset>" -s $REF/submission/submission.py --seed 42 --no-cache --no-plot --no-html --no-display --output <abs>.parquet`
    with `<dataset>` = `BCI[study=dreyer2026proteus,subset=full]` and then
    route B `official`. Both give G1's four scores and `n_cells` 8. (A cell
    mean of 0.333333 means the weights were not loaded.)
  - in-process: the `y_pred` arrays from the organisers' test loader, route
    B's test loader and route A's loader are equal element by element.
- **G7 (the evaluator behaves), in-process on route A and on route B's test
  loader, every protocol and every `lopo` fold, with four test doubles:**
  - a **constant** model scores exactly 0.333333;
  - a **look-up** model that returns the true label of each test window (found
    by the SHA-1 of the tensor) scores exactly 1.000000: windows and labels
    stay aligned through the whole path;
  - a **cell-marker** model, the look-up returning the true label on Graz
    windows and (label + 1) mod 3 on BrainHero windows: every cell scores
    exactly 1.0 or 0.0, half of each, and the cell mean is exactly 0.500000:
    the cell assignment is right;
  - a **counter** model that returns the number of windows it has seen in this
    stream (no modulo; 9 of the 14 session lengths are multiples of 3):
    `y_pred` equals the concatenation of `1 … n_s` over the test streams, with
    the lengths `n_s` taken from the cache keys (`official` and `all`: 189,
    189, 188, 189; `later`: 189, 188, 186, 199), and the base model's counter is still 0
    afterwards: every stream runs on a fresh copy.
- **G8 (no shared windows):** a model that memorises the *training* rows of a
  protocol (hash → label, class 0 for an unknown tensor) and is scored on its
  test rows recognises 0 windows in `official`, `later` and every `lopo` fold
  (cell mean exactly 0.333333) and all 755 in `all` (exactly 1.000000). This
  catches a window present on both sides under different keys.

### Phase 5: one solver through the whole path (est. 90 min)

The vehicle is the existing `Riemann-Sealed` solver in its simplest valid
configuration for people it has never seen: `personal="pooled"`,
`align="none"`, `adapt="none"`, everything else at the file's defaults. This
is a plumbing vehicle, not a candidate model: change no feature,
regularisation or other setting to improve its score or its speed.

- **The one solver edit.** Add to `RiemannSealedModel`:
  `def reset_state(self): self._buf = {}; self._pred = {"calls": 0, "windows": 0, "seconds": 0.0}; self._warned_w = False`
  (the `__init__` values of lines 1168 and 1178–1179; do not `.clear()`
  `_pred`, `predict` would raise `KeyError: 'calls'`). No other edit to that
  file, with two exceptions that each re-run G9, G10 and G11: the § 8
  contingency "the solver's fit fails", and a defect that the Phase 6 review
  confirms.
- **`scripts/proteus_candidate.sh <protocol> [fold]`**, modelled on
  `scripts/sprint1001_deploytest.sh`, with these differences:
  - do not source `sealed_lib.sh` (it sources `env.sh`). Use the skill's
    runner: `LOGDIR=$HOME/codabench/logs/proteus_setup/<run>`,
    `ENV_SETUP='source ~/codabench/env040.sh'`, `THREADS_PER_JOB=6`, then
    `source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"`;
  - bake: copy the solver, give it its own `name`, change two defaults with
    sed (`"personal": ["blend"]` → `["pooled"]`, `"align": ["subject"]` →
    `["none"]`), then `grep -F` three lines in the copy: `"personal":
    ["pooled"]`, `"align": ["none"]`, `"adapt": ["none"]`;
  - train through benchopt (`-o "BCI-decoding[training=True]"`, `--seed 42`);
  - copy the exported folder to `~/neuralbench/proteus_runs/<run>/replay`,
    make it read-only, replay inference-only from the copy (create each copy
    in the same step that uses it);
  - zip with `python -m zipfile -c <zip> <folder>/submission.py
    <folder>/<weights>` (files at the zip root), extract with
    `python -m zipfile -e` into an empty folder, replay once more from there;
  - the zip stays in the run's log folder, with its size and sha256.
- **Which runs:**
  - `official`: the whole chain on the organisers' dataset, exactly as the
    server runs it (`-d "BCI[study=dreyer2026proteus,subset=full]"`). Then
    replay the same export once through route B `official`.
  - `all`: the whole chain, training through route B `all`; replay on the
    organisers' dataset as well (its test set is the same 755 windows).
  - `later` and each `lopo` fold: training with evaluation only, through
    route B. No copy, no replay, no zip. Measure fold 0 before launching the
    rest (§ 6).

Gates:
- **G9 (fit and exact replay), on `official`:** the fit log line reads
  `[Riemann-Sealed] fitting on X=(1521, 41, 480), subjects=6, …`; the benchopt training run, the read-only
  replay, the unzipped replay and the route B replay have equal `objective_*`
  values and `n_cells` 8; loaded as in Phase 4, the exported folder, the
  read-only copy and the unzipped folder give identical `y_pred`, and the
  loaded model has `personal == "pooled"` and `align == "none"`. A second
  training run with seed 42 into a fresh folder gives identical `y_pred` (the
  audit re-runs training).
- **G9b (the other protocols):** `all`: the fit log reads
  `X=(2655, 41, 480), subjects=10`; its training run, read-only replay and
  unzipped replay through route B, and its replay on the organisers' dataset,
  have equal `objective_*` values and `n_cells` 8. `later`:
  `X=(1893, 41, 480), subjects=10`. `lopo` fold k:
  `X=(2655 − n_k, 41, 480), subjects=9`. Each run's parquet has the
  `objective_*` columns with the expected `n_cells`.
- **G10 (streaming contract of the model), on the fitted `official` model:**
  `B = 1` against one batch gives the same class on every window (record the
  largest probability difference and flag it above 1e-9); `copy.deepcopy`
  works; after `reset_state()` the predictions are unchanged; every prediction
  has shape `(1,)` and is an integer in {0, 1, 2}; the 20 highest-amplitude
  windows of the cache give finite outputs; after a reload,
  `model._pred["calls"] == 0` and `model._buf == {}`, and the same right after
  one in-process `fit` on the `official` training rows (done once in this
  phase; not part of the self-test).
- **G11 (old world intact).** Because `riemann_sealed.py` was edited, run the
  old regression once, in its own Bash-tool call that does **not** source
  `env040.sh` (`train_sealed.sh` sources `env.sh` itself; about 2 minutes):
  ```bash
  wsl.exe bash -s <<'EOF'
  exec 2>&1
  R=regress_$(date +%F_%H%M)
  SUBMISSION_DIR=$HOME/codabench/logs/$R/submission RUN_NAME=$R bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess
  grep -E "START train|END train|gate:" ~/codabench/logs/$R/STATUS.md
  EOF
  ```
  Expected: `START train` and `END train` rows from this run (a `skip train`
  row is a stale pass) and `gate: train 0.770000 replay 0.770000 EQUAL;
  harness blend_calib pooled 0.778333 (w=0.75), gap 0.0083 OK`. Any further
  edit to `riemann_sealed.py` needs a new run. The run writes two parquets
  into the old clone's `tracks/bci_decoding/outputs/` and a zip into
  `logs/$R`: expected, and the only writes this session makes in the old setup.
- **G12 (the zips):** files at the zip root only; the unzipped replay equals
  the training scores. Add two "ready, not uploaded" rows to `SUBMISSIONS.md`,
  each with path, size, sha256 and the local scores: the `official` Riemann
  zip, and a zip of the Phase 1 `MeanLogReg` export (expected 0.395863 /
  0.395672 / 0.396026). The server shows three numbers (cell mean, pooled
  balanced accuracy, accuracy) and the duration; if the user uploads either
  zip to the warm-up, those three must match. Note in the rows that the
  scoring image changed (`v2` → `v3`) since our last server test, so a
  finished run also re-proves that a pyriemann / scikit-learn joblib loads
  there. The `all` zip gets no row (it was trained on the warm-up test
  people): record its path, size and sha256 in `RESULTS.md` only.
- **Timing (report only):** milliseconds per window at `B = 1` for the fitted
  41-channel model, and the projected total for 106 sessions × 199 windows (an
  upper bound inferred from 20 people × 6 sessions minus the 14 public ones;
  the real size is unknown). If the projection exceeds 1,200 s, report it and
  the two known hot spots (`riemann_sealed.py:1655-1659`, `:282-289`) under
  "needs you". Make no speed change.

### Phase 6: reference table and independent review (est. 45 min)

- Fill a table in `RESULTS.md`: protocol × {constant, `MeanLogReg`, the Riemann
  vehicle} with cell mean, `n_cells`, fit time and ms per window, each row
  naming the loader and seed that trained it. The constant rows come from G7.
  The `MeanLogReg` rows of `later`, `all` and the ten `lopo` folds need 12
  training runs through route B that no earlier phase makes (one per call,
  `--seed 42`, a fresh `COMPET_SUBMISSION_DIR` each, about 15 s each): run
  them first in this phase. Head the table "for information only: untuned
  plumbing models" and mark the `all` rows "tested on its own training
  windows: score meaningless". State under it: on the 8-cell protocols the cell
  mean of a label-blind model has a standard deviation of about 0.025, so
  differences below about 0.05 mean nothing, and `later` tests 2 people. A
  `MeanLogReg` retrained through route B will not read 0.395863 (§ 1). Draw no
  conclusion about models.
- `analysis/proteus_selftest.py` runs G0, G2–G8 and G10 in under 5 minutes and
  trains nothing. It reads the window cache, the official loader,
  `$REF/submission` and the `official` export of Phase 5; G6's two CLI runs
  are subprocesses with `--output` under `logs/proteus_setup/selftest/`; it
  starts the G4 script with
  `subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", …])`. It prints
  one PASS / FAIL line per check and exits non-zero on any failure.
- **Independent review:** have a fresh sub-agent that has seen none of your
  reasoning read the new code and try to break it: a leak between training and
  test in any protocol, a label or cell misalignment, state that survives a
  stream, an integer id from `info` used as a key, a path that still reaches
  the old environment, a gate that passes when the thing it guards is broken.
  Fix what it confirms, re-run the self-test, and list what it could not break.

**Done when:** the self-test passes from a clean shell with only
`source ~/codabench/env040.sh` before it.

### Phase 7: docs and hand-off (est. 45 min)

- `codabench/PROTEUS_SETUP.md`: the environment, the cache, the protocols
  table, the commands (self-test, a protocol run, the candidate script), the
  gate table with measured values, and the traps of this brief that proved real.
- `LOG.md`: one entry in the house table format (Time | Step | Estimate |
  Actual | Result) per phase. `HANDOFF.md`: the top section complete and
  resumable from that file alone. `SUBMISSIONS.md`: the rows of G12.
- `DIRECTIONS.md`: § 1, § 2, § 5, § 6 and one change-log line. Decision 7:
  the evaluation half is done; for the model half, name the protocols and the
  comparison to run next, and do not cite the plumbing scores as evidence for
  or against a model. Decision 8 stays the user's, with a recommendation.
  Bring the file back towards 150 lines (it is at 188) by trimming only § 3
  (three lines and the link to `MODEL_ARCHITECTURE.md` instead of the
  diagram), § 4 (rows no longer relevant go to a link to `SEALED_RECIPE` § 2),
  § 5 items 3–6 (the old plan) and change-log lines older than 2026-10-02.
  Never remove an open decision row.
- A one-line banner at the top of `RELEASE_DAY.md`, `SEALED_RECIPE.md`,
  `TRACK2_BCI.md`, `SUBMISSION_GUIDE.md` and `COMPETITION.md`: "describes the
  pre-2026-10-13 contract; for PROTEUS see PROTEUS_SETUP.md". Do not rewrite
  them. Do not touch `README.md`.
- The skill's `.claude/skills/project-sprint/references/project-map.md`: the
  new environment, the new external facts of § 1, the new traps. The memory
  file `neuralbench-challenge-setup.md` and its `MEMORY.md` index line.

**Done when:** committed and pushed, and the final summary is sent in this
order: outcome in two sentences; what needs the user, each with a
recommendation; each phase with its gates (PASS / FAIL and the value);
deviations and incidents; what is running and where everything is; next steps
ranked.

## 6. Keeping jobs alive and never stale

- Most steps here take seconds to minutes: run them in the foreground with a
  `timeout`. Anything you expect to exceed 10 minutes (the `later` + `lopo`
  runs of the vehicle) runs in the foreground of a WSL shell inside a
  background Bash-tool call (`run_in_background: true`), resumable by step,
  with an ETA row written before launch.
- At most two heavy jobs at once, each with the thread caps of `env040.sh`.
- Past 1.5 × an estimate: read the live log, check for a stall or a fall-back
  to the old environment, fix the cause. Do not wait passively.
- Kill by PID, never `pkill -f`. Before a long run read the sleep settings
  from the Bash tool with `MSYS_NO_PATHCONV=1 powercfg /query SCHEME_CURRENT
  SUB_SLEEP`; do not change them.

## 7. Time budget and order

Deadline: **2026-10-16 00:38:51 EDT** (T = start = 2026-10-15 17:38:51 EDT, WSL `date`). Checkpoints: Phase 2 by 19:09, Phase 4 by 21:24, Phase 5 by 22:54, Phase 6 by 23:54; `later` + `lopo` launch only before 22:24; wrap-up reserve from 23:54.

Estimates sum to 6 h 45 min against 7 h. Order and checkpoints: Phase 0 → 1 →
2 (done by T + 1 h 30) → 3 → 4 (done by T + 3 h 45) → 5 (done by T + 5 h 15) →
6 (done by T + 6 h 15) → 7. The `later` + `lopo` background job of Phase 5 is
launched only if that is before T + 4 h 45.

- **If Phase 4 ends after T + 3 h 45:** Phase 5 shrinks to `reset_state()`,
  the `official` chain (G9, G10, G12) and G11.
- **Drop order when late:** the vehicle's `lopo` runs, its `later` run, its
  `all` chain, the `MeanLogReg` runs and `lopo` rows of the reference table,
  the stale-doc banners, the project-map
  and memory updates (list what was dropped under next steps).
- **Never drop:** a gate of Phases 1–4; G11 once `riemann_sealed.py` is edited
  (or revert the edit); the independent review; `HANDOFF.md`; the
  `DIRECTIONS.md` update.
- **Launch cutoff:** no job starts unless 1.5 × its estimate fits before the
  deadline minus 45 minutes. **Wrap-up reserve:** the last 45 minutes are
  Phase 7 only.

## 8. Contingencies

- **The upstream symlinks are broken** (`No module named benchmark_utils`):
  repair them in the new worktree only, never in the old clone:
  `cd ~/neuralbench/competition-040 && git ls-files -s | awk '$1=="120000"{print $4}' | while read f; do [ -L "$f" ] || { t=$(cat "$f"); rm "$f"; ln -s "$t" "$f"; }; done`
- **Upstream has moved past 442edcd:** record it under "needs you"; do not
  move the worktree.
- **G0 or G1 fails:** nothing else can be trusted. Find the cause first (wrong
  environment, weights not loaded, a changed cache, a run with neither
  `--seed 42` nor `--config …/training.yml`).
- **A `KeyError: 'subject_id'` in a fit** is a fault of our `Dataset`, not of
  the solver: fix `proteus_protocols.py`.
- **G3 or G6 cannot be made to pass:** stop building on the cache, record FAIL
  with what you learned, and finish Phases 6–7 on what still holds. List the
  alternative under next steps: serving the protocols through
  `load_task(..., overrides={"study.split": {"=replace=": True, "name":
  "PredefinedSplit", …}})`, which is valid as a config but has never been run.
- **The solver's fit fails on PROTEUS for a reason inside the solver:** fix
  only what blocks the fit, as an opt-in, and re-run G11. If the fix needs a
  modelling choice, stop Phase 5 there, finish Phases 6–7 with `MeanLogReg` as
  the only vehicle, and put the choice under "needs you".
- **Anything that needs an approval this brief does not give** (an install, a
  download, an upload, a change to the old setup): skip it and list it under
  "needs you".
- **You find that a fact in § 1 is wrong:** correct it in an addendum at the
  end of the saved brief (leave the original text), and continue from the
  corrected fact.

## Appendix A: what the 2026-10-13 setup session ran (for `SETUP.md` and `LOG.md`)

All in WSL, 16:41–16:47 EDT. Times are from that session's own output, not
from log stamps, and those marked "about" are approximate.

| Time | Step | Command (abridged) | Result |
|---|---|---|---|
| 16:41 | Fetch upstream | `git fetch origin` in `~/codabench/2026-competition` | `origin/main` = 442edcd, 48 commits ahead of the old clone |
| 16:42:23–16:42:31 | Worktree and venv | `git worktree add --detach ~/neuralbench/competition-040 origin/main`; `uv venv --python 3.12 ~/neuralbench/.venv040`; `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`; `uv pip install -r requirements.txt` (the worktree's file) | 8 s (wheels were cached) |
| about 16:42:50 | First prepare | `benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/training.yml` | failed: `No module named 'benchmark_utils'` (symlinks checked out as text files, `core.symlinks=false`) |
| about 16:43 | Symlink repair | the loop of § 8, without the `[ -L ]` guard | every git symlink of the worktree re-made |
| about 16:43:10–16:45:40 | Prepare | the same `benchopt prepare`, with `BENCHOPT_DATA_HOME=~/neuralbench/benchopt_data_040`; log `~/neuralbench/logs040/prepare_proteus.log` (no time stamps) | 1.48 GB in 19 s, extraction of 112 runs in 1 min 45 s |
| 16:46:03–16:46:10 | Loader check | `load_task("eeg", "_motor_imagery_stream", …)` | 41 × 480 at 120 Hz; 1,521 / 379 / 755 windows |
| 16:46:24–16:46:32 | Baseline smoke | `benchopt run tracks/bci_decoding --config tracks/bci_decoding/training.yml -s MeanLogReg --no-plot` | 0.395863, 8 cells; wrote `tracks/bci_decoding/outputs/benchopt_run_2026-10-13_16h46m29.parquet` and `tracks/bci_decoding/outputs/MeanLogReg/` (`submission.py`, `weights.joblib`) inside the worktree; both are still there |

Nothing was committed. PROTEUS is not mirrored to `Z:`.

Start with Phase 0 now.
