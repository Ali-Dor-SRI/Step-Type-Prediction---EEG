# PROTEUS setup: what exists, how to run it, what was proven

Built and verified in the setup session of 2026-10-15 (brief
[prompts/2026-10-13_proteus_train_test_setup.md](prompts/2026-10-13_proteus_train_test_setup.md);
measured values in [logs/proteus_setup/RESULTS.md](logs/proteus_setup/RESULTS.md);
timings in [LOG.md](LOG.md) "2026-10-15"). Everything here runs on the released
training data (PROTEUS, NEMAR `nm000290`) under the organisers' 0.4.0 contract:
streamed evaluation, one window at a time, balanced accuracy averaged over
(person, session, interface) cells. **Every score in this file is for information
only**: the models are untuned plumbing vehicles. The gates are exact comparisons.

## 1. The environment

```bash
source ~/codabench/env040.sh      # inside WSL; prints "neuralbench 0.4.0 cwd /home/ali_d/neuralbench/competition-040"
```

| | |
|---|---|
| venv | `~/neuralbench/.venv040`: Python 3.12.14, neuralbench / neuralset / neuralfetch / neuraltrain 0.4.0, benchopt 1.10.0, torch 2.14.1+cpu, scikit-learn 1.9.1, pyriemann 0.12 (full list `config/venv040_freeze.txt`) |
| upstream code | `~/neuralbench/competition-040` = `$COMPET_ROOT`: a detached git worktree of the old clone at `origin/main` 442edcd. Never `git checkout / pull / reset / stash / fetch` there (its 8 symlinks were re-made by hand; a checkout turns them back into text files). `origin/main` has moved on (a66bc23 on 10-15): moving the worktree is the user's call |
| data home | `~/neuralbench/benchopt_data_040` = `$BENCHOPT_DATA_HOME`: the BIDS release (1.5 GB) and the 120 Hz extraction cache (730 MB). Do not change loader settings |
| thread caps | `OMP/OPENBLAS/MKL_NUM_THREADS=6` (two jobs at once fit on 20 cores) |
| old setup | `env.sh`, `~/neuralbench/.venv` (0.3.1), `~/codabench/2026-competition`, `~/neuralbench/benchopt_data`: frozen; only `scripts/train_sealed.sh` still runs there, as the regression gate G11 |

The print is the proof of the environment: never source `env040.sh` with
`>/dev/null`. Every script of ours prints the protocol, the fold and the training
and test shapes in its first line; for a bare benchopt run on the organisers'
dataset the evidence is the parquet's `dataset_name` and `objective_n_cells`.
`SETUP.md` "2026-10-13" records how this environment was built.

## 2. The window cache (`~/neuralbench/proteus_cache`, 209 MB)

Built by `python ~/codabench/analysis/proteus_cache.py build` (11 s) from the
official loader called exactly as `bci_studies.py` does, reading the three
official datasets item by item. `X.npy` (2655, 41, 480) float32 microvolts in
**cache order** = (person, session, run, cue); `rows.parquet` with the stable keys
`person`, `session`, `run`, `interface`, `task`, `cue` (s), `start` (= cue + 1.0 s),
`label` (0 mi / 1 sub / 2 word), the organisers' `split`, `loader_pos`, `sha1`;
`keys.csv` for the isolated events check; `meta.json` (channel names, units, 120 Hz,
neuralbench version, upstream commit, build time). Proven: identical to the loader
(max difference 0.0 on every item, G3), labels equal to the raw BIDS `*_events.tsv`
both ways (G4, run as `python -I -B -X utf8`: the BIDS folder is untrusted data).
The official loaders' `info` integers (`record_id`, `stream_id`, `context_id`,
`subject_id`) are numbered by first appearance in each loader and are never used as
keys.

## 3. The four protocols (`analysis/proteus_splits.py`)

| Protocol | Train | Test | Streams / cells | Stands for |
|---|---|---|---|---|
| `official` | P01 P02 P03 P06 P07 P10 (1,521); validation P05 P08 (379) kept apart | P04 × 3 sessions + P09 (755) | 4 / 8 | the Codabench warm-up and the organisers' baselines |
| `lopo` | fold k = 0..9: all sessions of the other nine people | all sessions of P(k+1); 190 189 189 566 190 187 189 189 189 577 | 14 / 28 over the folds (2 2 2 6 2 2 2 2 2 6) | the 10 unseen people of the sealed test |
| `later` | session 1 of all ten (1,893) | sessions 2–3 of P04 and P10 (762) | 4 / 8 | later sessions of seen people |
| `all` | all 14 sessions (2,655) | the `official` test again (755), inside the training set, `score_meaningless` | 4 / 8 | the training set of a real submission; only train = replay counts |

`split(protocol, fold)` returns cache row indices, the test set in stream order
(streams one after another, runs by number, windows by cue). `ids_of(rows, idx)`
is the one function that makes the evaluator's integers from the keys:
`stream_id` = rank of (person, session), `record_id` = rank of (person, session,
run), `context_id` = 0 Graz / 1 BrainHero, `subject_id` = rank of person, `onset`
= window start in samples. `lopo` is scored per fold; its figure is the mean over
the 28 cells (`lopo_summary`), with the mean of the ten fold means beside it. Do not
combine `lopo` and `later` into one number. No protocol has a validation split.

## 4. The two evaluation routes

- **Route A, the harness** (`analysis/proteus_eval.py`): wraps a protocol's test
  rows with the organisers' `make_loader` (B = 1, unshuffled) and scores a model
  with their `predict_streams`, `balanced_accuracy` and `group_scores`; nothing is
  re-implemented. `load_submission(folder, meta)` loads a submission folder the way
  the platform does (`submission.py` by path, `COMPET_SUBMISSION_DIR`, the file's
  default parameters) and is where per-window predictions come from, since
  benchopt gives scores only.
- **Route B, benchopt** (`datasets/proteus_protocols.py`): a `Dataset` serving any
  protocol and fold from the cache with the organisers' five `info` keys; test
  loader B = 1 unshuffled, train loader batch 64 shuffled by benchopt's seed. Used
  by path: `-d "$HOME/codabench/datasets/proteus_protocols.py[protocol=lopo,fold=3]"`.
- Both routes reproduce the organisers' own run of the Phase 1 `MeanLogReg` export
  to 6 decimals and give identical per-window predictions (G6).

## 5. Commands

```bash
source ~/codabench/env040.sh

# the self-test: G0, G2-G8, G10; 349 checks, about 1 min, trains nothing
python ~/codabench/analysis/proteus_selftest.py

# a protocol run through benchopt (route B), one solver, one fold per call
REF=$HOME/codabench/logs/proteus_setup/my_run; mkdir -p $REF
COMPET_SUBMISSION_DIR=$REF/submission benchopt run tracks/bci_decoding \
    -d "$HOME/codabench/datasets/proteus_protocols.py[protocol=lopo,fold=3]" \
    -s $HOME/codabench/solvers/bci_decoding/riemann_sealed_proteus_pooled.py \
    -o "BCI-decoding[training=True]" --seed 42 --no-cache --no-plot --no-html --no-display \
    --output $REF/train.parquet          # scores are in the parquet, not on screen

# per-window predictions and scores of an exported folder on a protocol (route A)
python ~/codabench/analysis/proteus_eval.py score $REF/submission lopo 3 --out $REF/eval

# one solver through the whole path: bake -> train -> replay -> equal -> zip
bash ~/codabench/scripts/proteus_candidate.sh official      # also: all | later | lopo <fold>
```

`proteus_candidate.sh` is resumable by step (`logs/proteus_setup/cand_<run>/`,
`STATUS.md`); it bakes the vehicle's settings into
`solvers/bci_decoding/riemann_sealed_proteus_pooled.py` (Codabench runs a solver
file with its defaults), trains with a fresh `COMPET_SUBMISSION_DIR`, replays
inference-only from a read-only copy under `~/neuralbench/proteus_runs/<run>/`,
zips with `python -m zipfile -c` (files at the zip root), replays from the unzipped
folder, and for `official` adds a route B replay and a second training run. Rules
for every benchopt run: `--seed 42`, `--no-cache --no-plot --no-html --no-display`,
absolute `--output`, solver and dataset files by absolute path, a unique
`Solver.name` per variant, one solver and one fold per call; for inference-only runs
the log must show the weights were used (`[loading] … weights.joblib into
MeanLogReg`; `[Riemann-Sealed] predict: … calls` lines).

## 6. The gate table (measured 2026-10-15; details in RESULTS.md)

| Gate | What it proves | Result |
|---|---|---|
| G0 | neuralbench 0.4.0, `benchmark_utils` under `competition-040`, `.venv040`, `benchopt_data_040` | PASS |
| G1 | the organisers' `MeanLogReg` with `--config training.yml`: 0.395863 / 0.395465 / 0.395672 / 0.396026, 8 cells (12 s) | PASS |
| G2 | cache content: 2,655 × (41, 480) float32, no NaN, 885/886/884, 10 people / 14 sessions / 84 runs, per-session counts, 28 cells (Graz 96 each, BrainHero 90–103), every cell and run holds 3 classes, split 1521/379/755, channel names, start = cue + 1.000 s, 2,655 distinct hashes, 925 windows > 200 uV, min −4417.8 / max 3511.5 | PASS 22/22 |
| G3 | every official item = its cached row (max diff 0.0), label = table `code`, onset = start + 1.0 | PASS 12/12 |
| G4 | labels = raw BIDS events both ways (2655 = 2655), isolated `-I` | PASS 4/4 |
| G5a–d | protocol sizes, no leaks, stream order (official = the organisers' test and validation order element by element), ids from the keys, cell counts 8 / 28 / 8 / 8 | PASS 80/80 |
| G5e | what route B serves = the protocol's index sets, labels, order, meta | PASS 78/78 |
| G6 | routes agree: CLI inference on the organisers' dataset and on route B = G1; in-process `y_pred` organisers' loader = route B = route A | PASS 6/6 |
| G7 | constant 0.333333, look-up 1.000000, cell-marker 0.500000 with cells exactly 1 / 0, counter 1..n_s per stream on fresh copies | PASS 107/107 |
| G8 | a memoriser of the training rows recognises 0 test windows (all: 755) | PASS 26/26 |
| G9 | `official` Riemann chain: train = read-only replay = unzipped = route B = 2nd training, 0.406751 (8 cells); `y_pred` identical; `personal='pooled'`, `align='none'` | PASS |
| G9b | `all` (1.000000, meaningless, 4 runs equal), `later`, `lopo` 0..9: expected shapes, subjects and cells | PASS |
| G10 | B = 1 = batch on every window (max prob diff 7.9e-12), deepcopy, `reset_state()`, shapes, extreme windows finite, counters 0 after reload and after a fit | PASS |
| G11 | the old regression after the one solver edit: `gate: train 0.770000 replay 0.770000 EQUAL …` | PASS |
| G12 | zips at the root; unzipped replay = training; two "ready, not uploaded" rows in SUBMISSIONS.md | PASS |
| self-test | 349 / 349 from a clean shell, 55 s on a quiet machine (219 s with another job on the CPU) | PASS |
| review | a fresh sub-agent tried to break the code: 8 findings, none changing a number, all fixed and re-run (RESULTS.md) | PASS |

Timing (report only): 8.2–9.7 ms per window at B = 1; the largest plausible sealed
test (106 sessions × 199 windows) projects to about 3 minutes of the 3,600 s
allowed. Fits: 27 s (1,521 windows) to 49 s (2,655).

## 7. Traps that proved real in this session

- `objective_name` starts with `objective_` and reads `training=True` / `False`:
  comparing "all `objective_*` columns" across a training run and a replay says
  DIFFERENT. Compare the six scalars.
- Batch and single-window probabilities differ by ~1e-11 (chunked BLAS): exact
  equality tests must use the same call shape; classes are identical.
- A `MeanLogReg` retrained through route B does not read 0.395863 (lbfgs stops at
  a tolerance; the row order differs): compare one exported model across routes.
- ms-per-window numbers taken while two jobs share the CPU are 2–4× too high.
- benchopt prints no scores and no shapes; read the `--output` parquet; the
  `[loading]` line is the only proof that weights were used.
- `grep -c` with zero matches exits 1 and breaks a `&&` chain; long shell heredocs
  with quotes fail in the Bash tool: write patch scripts with the Write tool.
- Outside a benchopt run `Dataset.get_seed()` raises: the route B train loader then
  shuffles with seed 0.
- The organisers' `get_instance()` builds the solver with the file's defaults: a
  setting that is not baked into the candidate file is not what the server runs.
- A step runner that only logs a failure lets a script end in `ALL DONE` (found by
  the review): `proteus_candidate.sh` now exits 1 with `DONE WITH FAILURES`, and a
  fresh train run removes every downstream `.done` so stale replays are never
  compared with a new export.
- Every CLI entry of the PROTEUS tooling refuses to run outside `env040.sh`
  (`proteus_cache.require_env040()`): the old `env.sh` gives neuralbench 0.3.1.

## 8. Where everything lives

| What | Where |
|---|---|
| entry point, freeze | `env040.sh`, `config/venv040_freeze.txt` |
| cache, protocols, harness, checks | `analysis/proteus_cache.py`, `proteus_splits.py`, `proteus_eval.py`, `proteus_g4_events.py`, `proteus_selftest.py` |
| route B dataset | `datasets/proteus_protocols.py` |
| candidate script, vehicle | `scripts/proteus_candidate.sh`, `solvers/bci_decoding/riemann_sealed_proteus_pooled.py` (baked from `riemann_sealed.py`, which gained `reset_state()` on 10-15) |
| gate values, reference table | `logs/proteus_setup/RESULTS.md` (committed); run folders `logs/proteus_setup/<run>/` (git-ignored) |
| exports, zips | `logs/proteus_setup/ref_meanlogreg/` (MeanLogReg zip), `cand_official/` (Riemann zip), `cand_all/` (zip, no upload row) |
| replay copies | `~/neuralbench/proteus_runs/<run>/replay` (read-only), `unzipped` |
| the window cache | `~/neuralbench/proteus_cache/` |
