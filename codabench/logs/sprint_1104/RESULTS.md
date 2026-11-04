# Sprint 2026-11-04: train as streamed — RESULTS

Brief `prompts/2026-11-04_train_as_streamed.md`. **Outcome: no gain. X4f stays the sealed model.**

## Phase 0 (17:17–17:25, passed)
zhou `EQUAL`; X4f re-run equal (lopo fold 0 0.695397, later 0.673546); stateful gates ALL PASS;
every arm > 0.4 on later (tsf 0.6601, tsc 0.6711, tscf 0.6616 vs X4f 0.6735). Logs `logs/sprint_1104/p0/`.

## Phase 1 (17:28:50–18:19:40, estimate ~90 min, actual 51 min, under)
Data line checked in every lane (train=2465 test=190 chans=41). Rules v2 vs X4f
(adopted: Δ lopo ≥ +2.0, CI lower > 0, Δ later > 0; lean: ≥ +1.0, CI lower > 0).
(Row `K2` is another arm stored in the K2 lane; ignore it. X4f's rows are from K2.jsonl.)

Base: **X4f**. Cell-averaged balanced accuracy; diff = arm - base (points ×100); CI = paired bootstrap over people (10,000); up = people whose mean cell score rose.

| arm | lopo (28) | Graz | BrainHero | calib / online runs | Δ lopo [95 % CI] (up/10) | later (8) | Δ later [95 % CI] (up/2) | fit s |
|---|---|---|---|---|---|---|---|---|
| C2f+tscf | 0.6030 | 0.6131 | 0.5929 | 0.6299 / 0.5876 | -2.45 [-3.36, -1.38] (2/10) | 0.6409 | -3.26 [-5.29, -1.23] (0/2) | 114 |
| K2 | 0.5576 | 0.5506 | 0.5647 | 0.5587 / 0.5562 | -6.98 [-8.97, -4.97] (1/10) | 0.5892 | -8.43 [-10.15, -6.72] (0/2) | 68 |
| X4f | 0.6275 | 0.6302 | 0.6248 | 0.6539 / 0.6125 | — | 0.6735 | — | 69 |
| X4f+C0.003 | 0.6309 | 0.6310 | 0.6308 | 0.6539 / 0.6177 | +0.34 [-0.05, +0.78] (7/10) | 0.6736 | +0.00 [-0.53, +0.54] (1/2) | 108 |
| X4f+C0.03 | 0.6222 | 0.6235 | 0.6209 | 0.6491 / 0.6071 | -0.53 [-0.88, -0.22] (1/10) | 0.6749 | +0.14 [+0.00, +0.27] (2/2) | 113 |
| X4f+tsc | 0.6246 | 0.6235 | 0.6257 | 0.6536 / 0.6082 | -0.29 [-1.43, +0.71] (5/10) | 0.6711 | -0.24 [-0.54, +0.05] (1/2) | 136 |
| X4f+tscf | 0.6158 | 0.6168 | 0.6148 | 0.6376 / 0.6036 | -1.17 [-2.56, +0.21] (5/10) | 0.6616 | -1.19 [-2.15, -0.23] (0/2) | 137 |
| X4f+tsf | 0.6170 | 0.6183 | 0.6156 | 0.6356 / 0.6066 | -1.05 [-2.11, +0.42] (4/10) | 0.6601 | -1.34 [-2.66, -0.03] (0/2) | 122 |


C2f+tscf vs C2f (0.6083 lopo / 0.6424 later): −0.53 lopo / −0.15 later.

## Decision
- **tsf −1.05, tsc −0.29, tscf −1.17 lopo** (all CIs include 0 or are negative; later negative): replaying
  training sessions through the causal estimator does **not** help; it slightly hurts. The
  hypothesis (train/test mismatch of the centring) is not supported: the full-session centring
  gives cleaner training features and the classifier transfers to noisy streams fine.
- **C 0.003 +0.34 [−0.05, +0.78], C 0.03 −0.53 [−0.88, −0.22]**: below the lean bar; C = 0.01 stays.
- Nothing adopted or lean: **Phases 2, 3, 4 skipped** per the brief. No new zips. Sealed model unchanged:
  X4f all `60a0148c`.
- Other measured-today results (no gain): online class-balance correction, log-prob averages.
