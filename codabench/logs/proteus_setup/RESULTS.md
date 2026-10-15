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
| G5a | sizes of the four protocols (`analysis/proteus_splits.py`, 17:48:31–17:48:41, `splits_check.log`) | official 1521 / val 379 / 755, 4 streams, 8 cells; later 1893 / 762, 4 / 8; all 2655 / 755, 4 / 8; lopo test sizes 190 189 189 566 190 187 189 189 189 577, train = 2655 − test, cells 2 2 2 6 2 2 2 2 2 6 (28); fold k holds out P(k+1); unknown protocol, missing / out-of-range fold and a fold on a fold-less protocol raise | PASS (12/12) | `analysis/proteus_splits.py` |
| G5b | no leaks | official, later, every lopo fold: 0 shared rows; official and lopo: no test person in training; official: the 379 validation rows apart from both sides; later: no (person, session) on both sides, P04 and P10 have session 1 in training; all: its 755 test rows inside training, `score_meaningless` | PASS (26/26) | same |
| G5c | order | every protocol and fold: test streams contiguous, ordered by (run, cue); official test key sequence = organisers' test dataset order (755 items) and official val = their validation order (379 items), element by element | PASS (15/15) | same |
| G5d | ids from the stable keys (`ids_of`) | in every protocol and fold (stream_id, context_id) <-> (person, session, interface) both ways; cells 8 / 8 / 8 and 2 2 2 6 2 2 2 2 2 6; context 0 Graz / 1 BrainHero; record_id <-> (person, session, run); subject_id <-> person | PASS (26/26) | same |
| G5e | what route B (`datasets/proteus_protocols.py`) serves, in-process `get_data()`, every protocol and fold (`eval_gates.log`, 17:53:09–17:53:45, 3.2 s) | train and test hash sets and counts = the protocol's index sets; every served label = cache label; train / test hashes disjoint (all: test inside train); test hash sequence = the protocol's test order; `n_classes` 3, `sfreq` 120.0, `ch_names`, `chs_info`, `n_chans` 41, `n_times` 480, `device` cpu = organisers' | PASS (78/78) | `analysis/proteus_eval.py gates` |
| G6 | routes agree, Phase 1 `MeanLogReg` export: CLI inference-only on `BCI[study=dreyer2026proteus,subset=full]` (8.1 s) and on route B `official` (6.9 s); in-process `y_pred` from the organisers' test loader, route B's test loader and route A | both CLI runs: 0.395863 / 0.395465 / 0.395672 / 0.396026, 8 cells, `[loading] … weights.joblib into MeanLogReg` in the log; the three `y_pred` (755) equal element by element; in-process `predict_streams` on the organisers' loader = G1; route A = route B = organisers' scores as exact floats | PASS (6/6) | `g6/g6_official_loader.parquet`, `g6/g6_route_b_official.parquet`, `g6/*.log`, `g6/g6_y_pred_*.npy` |
| G7 | the evaluator behaves: four test doubles on route A and route B, every protocol and fold (3.9 s) | constant exactly 0.333333; look-up exactly 1.000000; cell-marker: cells exactly 1.0 / 0.0, half each, mean 0.500000; counter = 1..n_s per stream with n_s from the cache keys (official / all 189 189 188 189; later 189 188 186 199) and the base model's counter 0 afterwards | PASS (107/107) | same |
| G8 | no shared windows: memoriser of the training rows scored on the test rows, routes A and B (1.5 s) | official, later, every lopo fold: 0 recognised, cell mean exactly 0.333333; all: 755 recognised, exactly 1.000000 | PASS (26/26) | same |
| G11 | old world intact after the `reset_state()` edit: `train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess` in the old environment (`env.sh`, its own Bash call), 17:56:37–17:58:25 (1 min 48 s), run `regress_2026-10-15_1756` | `START train` 17:58:00 / `END train rc=0` 17:58:16 from this run; `gate: train 0.770000 replay 0.770000 EQUAL; harness blend_calib pooled 0.778333 (w=0.75), gap 0.0083 OK` | PASS | `g11_regress_2026-10-15_1756.log`, `logs/regress_2026-10-15_1756/STATUS.md` (two parquets in the old clone's `outputs/`, a zip in that log folder: expected) |
| G9 | `official` chain (`scripts/proteus_candidate.sh official`, 17:59:11–18:03:16; `cand_official/`): fit log line; benchopt training run = read-only replay = unzipped replay = route B replay = second training run (seed 42, fresh folder); in-process `y_pred` of the export, the read-only copy, the unzipped folder and the second export (`g9_g10_rerun.log`, 18:10:27–18:11:19) | `[Riemann-Sealed] fitting on X=(1521, 41, 480), subjects=6, classes=[0, 1, 2], features=4646, align=none/riemann, personal=pooled` (fit 26.8 s); all five parquets 0.406751 / 0.406827 / 0.405717 / 0.405298, `n_cells` 8, **EQUAL**; `y_pred` replay / unzipped / export2 == export: 0 of 755 differ; loaded model `personal='pooled'`, `align='none'`, `adapt='none'` | PASS (train, replay, replay_unzip, replay_cross, train2 all rc 0; 5/5 in-process checks) | `cand_official/RESULTS.md`, `*.parquet`, `*.log`, `g9_y_pred_*.npy` |
| G9b | the other protocols | `all` (`cand_all/`, 18:03:42–18:06:00): `X=(2655, 41, 480), subjects=10` (fit 49.3 s); training = read-only replay = unzipped replay (route B all) = replay on the organisers' dataset: 1.000000 on all four scores, 8 cells, **EQUAL** (its test is inside its training set: score meaningless). `later` (`cand_later/`): `X=(1893, 41, 480), subjects=10`, 8 cells, 0.492382. `lopo` fold k (`cand_lopo_f<k>/`, 18:00–18:10, 33–49 s per fit): `X=(2655 − n_k, 41, 480), subjects=9` for k = 0..9 (2465, 2466, 2466, 2089, 2465, 2468, 2466, 2466, 2466, 2078), cells 2 2 2 6 2 2 2 2 2 6; every parquet has the `objective_*` columns | PASS | `cand_*/RESULTS.md`, `train.parquet`, `train.log` |
| G10 | streaming contract of the fitted `official` model (`g9_g10.log` 18:01:29–18:03:14 with the in-process fit; re-run `g9_g10_rerun.log` after the test-code fix below) | B = 1 vs one batch: same class on all 755 windows, largest probability difference 7.9e-12 (not flagged); `copy.deepcopy` works and the copy predicts the same; predictions unchanged after `reset_state()`; `reset_state()` leaves `_buf == {}` and `_pred` zeroed; every prediction shape (1,), integer in {0, 1, 2}; the 20 highest-amplitude windows (4,417.8 .. 480.3 uV) give finite outputs; after a reload `_pred['calls'] == 0` and `_buf == {}`; the same right after one in-process fit on the route B `official` training rows (fit 28.9 s) | PASS (9 checks + the fit check) | `cand_official/g9_g10*.log` |
| G12 | the zips | `official` Riemann zip `cand_official/riemann_sealed_proteus_pooled_official_2026-10-15.zip`: 562,449 B, sha256 `cd9f6a193a3286e73368c811284d00bdfe1f6987e6b09b5616e24f650458c742`, entries `submission.py`, `riemann_sealed.joblib` at the root; unzipped replay = training scores. `MeanLogReg` zip `ref_meanlogreg/meanlogreg_official_2026-10-15.zip`: 3,022 B, sha256 `5b31484b87541c826589a6803ec3615b0930c5ebf6996db805181b50086d6583`, `submission.py` + `weights.joblib` at the root; unzipped replay 0.395863 / 0.395465 / 0.395672 / 0.396026 = training. Both rows added to `SUBMISSIONS.md` ("ready, not uploaded"; image `v2` → `v3` noted). `all` zip (no SUBMISSIONS row): `cand_all/riemann_sealed_proteus_pooled_all_2026-10-15.zip`, 676,458 B, sha256 `640a14fbc93e04de8c18e9f6db43046adeb730ad6444d26a2735773e0a232ef5`, files at the root, unzipped replay = training (1.000000, meaningless) | PASS | `*/RESULTS.md`, `zip.log` |
| timing | report only: ms per window at B = 1 for the fitted 41-channel model | 8.2 ms (in-process loop, quiet machine), 8.6–9.7 ms (route A `predict_streams`, all 13 exports), 8.9 ms (benchopt replay log); projected 106 sessions × 199 windows = 21,094 windows: **173–205 s**, far below 1,200 s (no "needs you"). Values of 20–35 ms seen in some benchopt logs were measured while two jobs shared the CPU | — | `reftable.md` |
| self-test | `analysis/proteus_selftest.py` from a clean shell with only `source ~/codabench/env040.sh` before it: 18:12:47–18:13:43 before the review fixes, 18:41:26–18:45:07 after them | G0 4, G2 22, G3 12, G4 5, G5a–d 80, G5e–G8 217, G10 9 = **349 / 349 PASS** both times; 55.3 s on a quiet machine, 218.9 s with another session's sweep jobs sharing the CPU (G10 163 s; still under 5 min) | PASS | `selftest/selftest.log` |
| review | independent review by a fresh sub-agent (18:12–18:30; findings and fixes below) | 8 findings, none making a reported number wrong; 8 fixed (3 in `proteus_candidate.sh`, 5 in the harness) and everything re-run: 13 candidate runs ALL DONE, G9/G10 PASS 15/15 in `g9_g10.log` and STATUS.md, self-test 349/349 | PASS | this file, `cand_*/STATUS.md` |

## Phase time log

| Phase | Estimate | Start | End | Note |
|---|---|---|---|---|
| 0 | 15 min | 17:38 | 17:41 | under |
| 1 | 30 min | 17:39 | 17:44 | under (the run 12 s) |
| 2 | 45 min | 17:44 | 17:50 | under (build 11 s, checks 8 s) |
| 3 | 45 min | 17:47 | 17:52 | under (checks 10 s) |
| 4 | 90 min | 17:52 | 17:56 | under (gates 25 s + two CLI runs 15 s) |
| 5 | 90 min | 17:56 | 18:12 | under: official chain 4 min, all chain 2.5 min, later + 10 lopo 11 min (two jobs at once), G11 1 min 48 s |
| 6 | 45 min | 18:04 | 18:45 | on estimate: 12 MeanLogReg runs 2 min, table 1 min, self-test 1 min, review 18 min (fresh sub-agent), fixes and re-runs 15 min |
| 7 | 45 min | 18:14 | 18:52 | under (docs written while the review ran) |
| total | 6 h 45 min | 17:38 | 18:52 | 1 h 14 min of the 7 h |

## Reference table: for information only: untuned plumbing models

Protocol × {constant, `MeanLogReg`, the Riemann vehicle}: cell mean, `n_cells`, fit time and ms per window, each row naming the loader and seed that trained it (`reftable.py` of the session scratchpad, 18:11:19–18:12:17; `reftable.json`). The constant rows come from G7. The ms/window column gives the route A measurement on a quiet machine and, after `;`, the last `[Riemann-Sealed] predict:` line of the benchopt log (20–35 ms where two jobs shared the CPU).

| protocol | model | cell mean | n_cells | fit (s) | ms / window at B = 1 (route A; benchopt log) | trained through |
|---|---|---|---|---|---|---|
| official | constant (class 0) | 0.333333 | 8 | 0 | 0.0 | no training |
| official | MeanLogReg | 0.395863 | 8 | — | 0.3 | official loader (`BCI[study=dreyer2026proteus]`), seed 42 via training.yml (= G1) |
| official | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.406751 | 8 | 26.8 | 8.6; 8.9 | official loader (`BCI[study=dreyer2026proteus,subset=full]`), seed 42 |
| later | constant (class 0) | 0.333333 | 8 | 0 | 0.1 | no training |
| later | MeanLogReg | 0.428786 | 8 | — | 0.3 | route B `ProteusProtocols[protocol=later]`, seed 42 |
| later | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.492382 | 8 | 33.5 | 8.6; 24.2 | route B `ProteusProtocols[protocol=later]`, seed 42 |
| all | constant (class 0) | 0.333333 **tested on its own training windows: score meaningless** | 8 | 0 | 0.1 | no training |
| all | MeanLogReg | 0.466050 **tested on its own training windows: score meaningless** | 8 | — | 0.3 | route B `ProteusProtocols[protocol=all]`, seed 42 |
| all | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 1.000000 **tested on its own training windows: score meaningless** | 8 | 49.3 | 8.6; 20.5 | route B `ProteusProtocols[protocol=all]`, seed 42 |
| lopo[fold=0] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=0] | MeanLogReg | 0.400202 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=0]`, seed 42 |
| lopo[fold=0] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.469926 | 2 | 43.2 | 9.0; 23.9 | route B `ProteusProtocols[protocol=lopo,fold=0]`, seed 42 |
| lopo[fold=1] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=1] | MeanLogReg | 0.530858 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=1]`, seed 42 |
| lopo[fold=1] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.370139 | 2 | 46.0 | 8.6; 9.0 | route B `ProteusProtocols[protocol=lopo,fold=1]`, seed 42 |
| lopo[fold=2] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=2] | MeanLogReg | 0.672155 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=2]`, seed 42 |
| lopo[fold=2] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.477005 | 2 | 39.4 | 8.6; 8.7 | route B `ProteusProtocols[protocol=lopo,fold=2]`, seed 42 |
| lopo[fold=3] | constant (class 0) | 0.333333 | 6 | 0 | 0.1 | no training |
| lopo[fold=3] | MeanLogReg | 0.371304 | 6 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=3]`, seed 42 |
| lopo[fold=3] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.369825 | 6 | 40.9 | 8.4; 35.2 | route B `ProteusProtocols[protocol=lopo,fold=3]`, seed 42 |
| lopo[fold=4] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=4] | MeanLogReg | 0.416499 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=4]`, seed 42 |
| lopo[fold=4] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.389449 | 2 | 48.8 | 9.1; 9.5 | route B `ProteusProtocols[protocol=lopo,fold=4]`, seed 42 |
| lopo[fold=5] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=5] | MeanLogReg | 0.455287 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=5]`, seed 42 |
| lopo[fold=5] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.558445 | 2 | 36.1 | 9.3; 8.5 | route B `ProteusProtocols[protocol=lopo,fold=5]`, seed 42 |
| lopo[fold=6] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=6] | MeanLogReg | 0.422334 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=6]`, seed 42 |
| lopo[fold=6] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.456720 | 2 | 37.0 | 9.1; 8.0 | route B `ProteusProtocols[protocol=lopo,fold=6]`, seed 42 |
| lopo[fold=7] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=7] | MeanLogReg | 0.438441 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=7]`, seed 42 |
| lopo[fold=7] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.477957 | 2 | 36.1 | 9.7; 8.2 | route B `ProteusProtocols[protocol=lopo,fold=7]`, seed 42 |
| lopo[fold=8] | constant (class 0) | 0.333333 | 2 | 0 | 0.1 | no training |
| lopo[fold=8] | MeanLogReg | 0.481501 | 2 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=8]`, seed 42 |
| lopo[fold=8] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.409483 | 2 | 43.5 | 9.6; 9.8 | route B `ProteusProtocols[protocol=lopo,fold=8]`, seed 42 |
| lopo[fold=9] | constant (class 0) | 0.333333 | 6 | 0 | 0.1 | no training |
| lopo[fold=9] | MeanLogReg | 0.422459 | 6 | — | 0.3 | route B `ProteusProtocols[protocol=lopo,fold=9]`, seed 42 |
| lopo[fold=9] | Riemann-Sealed-Proteus-Pooled (pooled / align none / adapt none) | 0.380160 | 6 | 32.7 | 9.6; 8.1 | route B `ProteusProtocols[protocol=lopo,fold=9]`, seed 42 |
| lopo (all 10 folds) | MeanLogReg | **0.442755** over the 28 cells (mean of the 10 fold means 0.461104) | 28 | — | — | route B, seed 42, one fold per call |
| lopo (all 10 folds) | Riemann | **0.418506** over the 28 cells (mean of the 10 fold means 0.435911) | 28 | — | — | route B, seed 42, one fold per call |

On the 8-cell protocols the cell mean of a label-blind model has a standard deviation of about 0.025, so differences below about 0.05 mean nothing, and `later` tests 2 people. A `MeanLogReg` retrained through route B does not read 0.395863 (its lbfgs fit depends on the training-row order, brief § 1). The MeanLogReg fit time is not logged by the solver: each route B training run took 7–11 s wall time. Draw no conclusion about models from this table.

## Independent review (Phase 6): what it confirmed, what was fixed, what it could not break

A fresh sub-agent that had seen none of this session's reasoning read the new
code (the five `analysis/proteus_*.py` files, `datasets/proteus_protocols.py`,
`scripts/proteus_candidate.sh`, the `reset_state()` edit, `env040.sh`) and ran
its own read-only checks. **No finding makes a reported number wrong.** Findings,
most severe first, each fixed before the final re-runs:

1. `proteus_candidate.sh` printed `ALL DONE` and exited 0 after a G9/G10 failure
   or a DIFFERENT comparison (confirmed on the first `cand_official` run). Fixed:
   a `FAILED` flag; `ALL DONE` only when everything passed, else
   `DONE WITH FAILURES` and exit 1.
2. The recorded `cand_official` / `cand_all` results came from my manual re-runs
   after the two test-code fixes, while `STATUS.md` still showed the first run's
   failures and the lopo / later `RESULTS.md` had the old column set. Fixed by
   re-running the resumable script on all 13 runs (18:31–18:41): only G9/G10 and
   the comparisons re-executed; every STATUS.md now ends in `ALL DONE`, `g9_g10.log`
   holds the PASS (15 checks), every RESULTS.md has the same columns.
3. Resumption could validate a stale export (a re-run train with old `.done`
   files skips the replays; a date-stamped zip name changes across days; a
   missing `--zip` was silently omitted). Fixed: a real train run removes every
   downstream `.done`; the zip name is persisted in `zip.name`; `equal --zip`
   fails when the zip is missing.
4. The G7 cell-marker double did not assert the number of cells. Fixed: it now
   requires 8 / 8 / 8 / `LOPO_CELLS[fold]` cells (indirectly covered before by G5d
   and the counter double).
5. The G6 flag `weights_loaded` was misnamed for Riemann (its predict lines are
   printed whether loaded or freshly fit). Renamed `inference_evidence`, with the
   reasoning from the brief (without its joblib the model raises on the first window).
6. `proteus_eval.py score --official-loader` with a non-official protocol would
   mislabel its output. Fixed: refused.
7. No environment guard outside the self-test. Fixed: `proteus_cache.require_env040()`
   at every CLI entry (verified: under the old `env.sh` it exits with
   `wrong environment … neuralbench 0.3.1 != 0.4.0`).
8. `equal_parquets` compared only the first parquet's columns. Fixed: the union,
   a missing column counts as different.

Could not break (its own re-derivations): (a) leaks: every protocol re-derived
from (person, session) alone equals `split`; row and SHA-1 intersections 0 except
`all` (755, flagged); val disjoint from both sides; no window under two keys (2,655
distinct hashes at full precision, at 1 µV rounding and by coarse fingerprint; 84
distinct run-level hashes; no overlapping windows). (b) alignment: `X[i]` hash =
`rows.sha1[i]`; route A and route B serve hash, label and ids by position equal to
the cache and `ids_of`; the solver's own `_table_row_mismatch` accepts route B's
trigger table; `scores_from` is the objective's arithmetic line by line. (c) state:
both routes score through the organisers' `predict_streams` (deepcopy +
`reset_state()` per stream); the solver defines no copy hooks; its only test-time
state is `_buf`, cleared by `reset_state()`; the counter double would fail on a
reused object. (d) no `info` integer is used as a key. (e) nothing in the new code
reaches the old environment; `~/codabench/analysis` shadows no upstream module.
(f) every FAIL reaches a non-zero exit; G6 compares against the Phase 1 parquet,
not against itself; G8's hit count is independent of the model. (g) no silent
retrain (the objective passes no train loader under `training=False`; a missing
joblib raises); the replay copy is read-only before benchopt runs; both zips have
their two files at the root; `equal` compares the right parquets; `fold=3` reaches
route B as an int. (h) the benchopt route B seed differs from the in-process 0 and
changes only the train shuffle order, which the Riemann solver ignores and G5e
compares as sets.

## Deviations and incidents

- Two defects in my own test code were found by the first `official` chain and
  fixed after the jobs ended (brief § 4: no edit while a job runs): `equal`
  compared `objective_name`; the deepcopy check mixed call shapes. Both re-run.
- The review's eight findings above (the script's success reporting, the audit
  trail of the first run, resumption staleness, five smaller items): fixed, re-run.
- Upstream `origin/main` moved past 442edcd (a66bc23): left for the user.
- Another session started sweep jobs (`scripts/proteus_w1016_p*.sh`,
  `analysis/proteus_sweep.py`, lanes A and B) in this folder at about 18:30, while
  this session was still running: not this session's, left running. The second
  self-test timing (219 s) and some benchopt log timings reflect that contention.
- No gate failed. Nothing was dropped. Nothing was uploaded.
