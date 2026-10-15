# PROTEUS setup session 2026-10-15: gate results

Brief: `prompts/2026-10-13_proteus_train_test_setup.md`. Start 17:38:51 EDT,
deadline 2026-10-16 00:38:51 EDT. Every score here is **for information only**
(untuned plumbing models); every gate is an exact comparison. Environment:
`source ~/codabench/env040.sh` (neuralbench 0.4.0, `~/neuralbench/.venv040`,
upstream worktree `~/neuralbench/competition-040` @ 442edcd, data home
`~/neuralbench/benchopt_data_040`).

## Gate table

| Gate | Check | Measured | Result | Where |
|---|---|---|---|---|
| G0 | `importlib.metadata.version("neuralbench")` | 0.4.0 | PASS | `env040.sh` |
| G0 | `benchmark_utils.__file__` under `competition-040` | `~/neuralbench/competition-040/benchmark_utils/__init__.py` | PASS | |
| G0 | `sys.prefix` ends with `.venv040` | `/home/ali_d/neuralbench/.venv040` | PASS | |
| G0 | `BENCHOPT_DATA_HOME` ends with `benchopt_data_040` | yes | PASS | |
| G1 | reference `MeanLogReg`, `--config training.yml` (seed 42, official loader), 17:40:35–17:40:47 (12 s) | cell mean 0.395863, session mean 0.395465, pooled 0.395672, accuracy 0.396026, `n_cells` 8 | PASS | `ref_meanlogreg/train.parquet`, `train.log` |
| G1 | `ref_meanlogreg/submission/` holds `submission.py` + `weights.joblib` | 959 B + 2,853 B | PASS | |
| G2 | window cache `~/neuralbench/proteus_cache` built from the official loader (17:45:55–17:46:06, 11.0 s): 22 checks (`cache_check.log`) | (2655, 41, 480) float32, 0 NaN; classes 885 / 886 / 884; 10 people, 14 sessions, 84 runs (runs 1..6 in every session); per-session counts 190 189 189 189 189 188 190 187 189 189 189 192 186 199; 28 cells, Graz 96 each, BrainHero 90..103; every cell and run holds 3 classes; split 1521 / 379 / 755 with the expected people; channel names = brief § 1; 120 Hz, microvolts; start = cue + 1.000 s (max diff 0.0); 2,655 distinct SHA-1 and keys; 925 windows with abs > 200 uV; min −4417.8 / max 3511.5 | PASS (22/22) | `analysis/proteus_cache.py check` |
| G3 | every item of the official train / val / test datasets vs the cache row found by its key (1521 + 379 + 755 items, 6.8 s) | max abs X difference 0.0 in all three; label = cache = table `code`; `info["onset"]` = `start` + 1.0; split and loader position recorded | PASS (12/12) | same |
| G4 | labels vs the raw BIDS `*_events.tsv`, isolated `python -I -B -X utf8` (`g4_events.log`, 1 s) | 84 files; 0 duplicate cue onsets; 2655/2655 windows matched to a cue row on round(onset × 1000) ms, 2655 labels agree (mi 0 / sub 1 / word 2); 2655 cue rows in the files, 0 without a window | PASS (4/4) | `analysis/proteus_g4_events.py` |

## Phase time log

| Phase | Estimate | Start | End | Note |
|---|---|---|---|---|
| 0 | 15 min | 17:38 | 17:41 | under |
| 1 | 30 min | 17:39 | 17:44 | under (the run 12 s) |
| 2 | 45 min | 17:44 | 17:50 | under (build 11 s, checks 8 s) |
