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

## Phase time log

| Phase | Estimate | Start | End | Note |
|---|---|---|---|---|
| 0 | 15 min | 17:38 | 17:41 | under |
| 1 | 30 min | 17:39 | 17:54 | under (the run 12 s) |
