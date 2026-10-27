# Sprint 2026-10-27: sealed-phase accuracy over F64T

Time frame: 2026-10-27 11:00 → **16:00 EDT** (the user's "until 4:00 PM today").
The user is reachable at the start; work autonomously after. Chosen by the value
check: the user asked for "perfect accuracy to move up the leaderboard", but the
0.988 they saw is the all-trained zip scored on the warm-up test windows it was
trained on (memorisation; the warm-up does not rank). The user then asked Claude to
pick the task that translates best to the sealed phase: **raise lopo / later**, the
two local measures that mirror the sealed test's halves (10 unseen people / later
sessions of known people).

## 1. Context (verified 2026-10-27)

- Workstream: Codabench Track 2 (PROTEUS), `codabench/`, branch
  `feat/codabench-track2`. State of record: `HANDOFF.md` top, `DIRECTIONS.md`,
  `SUBMISSIONS.md`, `LOG.md`.
- Environment: WSL, `source ~/codabench/env040.sh`; scripts via
  `wsl.exe bash -lc 'bash ~/codabench/scripts/<x>.sh'`. 20 cores, 31 GB.
  Sleep is off on AC and DC (checked 11:08).
- Candidate: **F64T** = F64 (stream covariance + feature re-centring, pooled shrinkage
  LDA, bpt4) with 1-thread pools. **lopo 0.5672, later 0.6005** (cell-averaged
  balanced accuracy, chance 0.333). Warm-up: exact match, 19.0 s.
- Harness: `analysis/proteus_sweep.py` (route B fit + route A streamed B = 1 score,
  one JSON row per (config, protocol, fold), resumable) and
  `analysis/proteus_w1016_summary.py` (paired people bootstrap, 10,000). Rows go to
  `logs/proteus_w1016/<lane>.jsonl`; base F64 rows are lane B.
- **Already measured, do not repeat** (LOG 2026-10-16, RESULTS_p2/p3/p7b/p8):
  no xDAWN −1.5, CAR +0.1, clipping −0.3, 1–3 s crop −3.8, Euclidean means −1.4,
  router + personal LDA −4.9, EEGNet / EEGNet ensemble ≈ 0, buffers 16 / all and
  K 2 / 32 worse, icoh on F64 **+1.65 [+0.43, +2.85]**, later +2.35 (below the bar).
  **Checked offline this morning (saved probabilities):** causal per-stream class
  balancing: no gain (lopo 0.5627–0.5639); ensembles F64+FI64 +1.31, F64+F16 +0.35,
  F64+F16+F0 −0.23, F64+FI64+F16 +1.25: none beats FI64 alone.
- Machine change for this sprint: the sweep now scores with every thread pool at 1
  (`one_thread()`), which makes scoring ~30× faster and changes no prediction
  (shown for F64/F64T 10-27; re-checked in Phase 0).

## 2. Goal and priorities

Deliverable: `logs/sprint_1027/RESULTS.md` (all arms, paired CIs), LOG entry,
DIRECTIONS update, and — only if a candidate clears the rules — a baked, thread-pinned
zip pair (official + all) ready for a sealed upload, labelled "pending nested check".

Priority (highest first):
1. **fscale** — per-group / per-stream feature *scaling* on top of F64's re-centring.
   The re-centring was the biggest lever (+8.3 lopo); the second moment is the
   natural next step for unseen people.
2. **Delta / time-course features** — the class signal peaks in the second after
   feedback onset and is as strong in delta as theta (EDA 10-26). Arms: `dbpt8`
   (delta band power, 8 bins, added to bpt4), `bpt8` (finer bins instead of bpt4),
   `slow_block=True` (0.1–4 Hz waveform bins, existing code, never run on PROTEUS).
3. **logreg0.01** — standardised L2 logistic regression instead of shrinkage LDA,
   one fixed C (no tuning on lopo).
4. **Combination** of what passes (+ icoh), then a bake.

## 3. Rules

- Commit on `feat/codabench-track2`, prefix `codabench:`, push to `personal` and
  `org` after every commit. No uploads, tags, merges, PRs. `logs/` is ignored except
  `RESULTS*.md` (force-add).
- New solver options are opt-in; defaults bit-identical (zhou regression must read
  `train 0.770000 replay 0.770000 EQUAL`; F64 re-run must equal the 10-16 rows).
- Evidence: paired people bootstrap over the 10 lopo people; later has 2 people
  (report, never decide on it alone). One fixed seed (deterministic fits).
  No hyper-parameter is chosen on lopo/later beyond the pre-registered arm list.
- ≤ 2 heavy jobs, 10 threads each. Thread pools 1 during scoring.

## 4. Phases and decision rules

### Phase 0: harness + opt-ins smoke (estimate 8 min) — `scripts/sprint1027_p0.sh`
F64 re-run (lopo fold 0 + later) under the edited solver and harness; each new arm on
`later`; zhou regression. **Done when:** F64r equals the 10-16 rows to 6 dp, every
smoke arm finishes with a sane score (> 0.4), zhou EQUAL. Otherwise fix first.

### Phase 1: single arms vs F64 (estimate 50–60 min, two lanes) — `scripts/sprint1027_p1.sh`
Arms (F64 + one change): `fscale`, `dbpt8` (xblocks bpt4_dbpt8), `bpt8` (xblocks
bpt8), `slow` (slow_block=True), `logreg` (clf=logreg0.01). lopo 0–9 + later.
**Decision (rules v2):** an arm is **adopted** if Δ lopo ≥ +2.0 points with the
95 % CI lower bound > 0 **and** Δ later > 0. It is **lean** if Δ lopo ≥ +1.0 with the
CI lower bound > 0. Otherwise "no gain". icoh counts as lean (10-16 result).

### Phase 1b (pre-registered 11:40, after fscale / logreg, before any 1b number): delta in the filter bank
`scripts/sprint1027_p1b.sh`, lane G, starts when lane E frees (estimate 35 min). Arms:
`fbd` = filter-bank bands 1to4 + the four standard ones; `fb6` = 1to4, 4to8, 8to13,
13to20, 20to30, 30to45 (new opt-in `fbands`; bpt / icoh blocks unchanged). Same
decision rule as Phase 1. fscale and logreg came out "no gain" (CI crosses 0).

### Phase 2: combination (estimate 20–40 min)
If anything is adopted or lean: `C1` = F64 + every adopted/lean change except icoh;
`C2` = C1 + icoh. **Decision:** the new candidate is the simpler of C1/C2 that has
Δ lopo ≥ +2.0 vs F64 with CI lower bound > 0 and Δ later > 0; C2 must beat C1 by
≥ 1.0 lopo to be preferred. If no combination passes but one single arm was
adopted, that arm is the candidate. If nothing passes: "no gain", F64T stays.

### Phase 3: bake (estimate 35–45 min) — only with a candidate
Bake the candidate's settings into a solver copy with the 1-thread pin (as F64T),
train official + all, export, unzip-replay equality, stateful gates, timing. Zips
labelled **pending the nested check** (3.5–4 h, does not fit today).

## 5. Keeping jobs alive
Phase scripts use the skill's `step_lib.sh` (`.done` markers, STATUS.md,
per-lane jsonl). Launched attached in a background Bash call; Monitor on
`monitor_loop.sh <logdir> 60–300`. SLOW (> 1.5× ETA) or STALLED → read the step log,
check the process and the data-shape line, fix the root cause.

## 6. Time budget
P0 11:10–11:20 · P1 11:30–12:30 · P2 12:40–13:30 · P3 13:30–14:20 ·
docs/wrap-up from 15:20. **Wrap-up reserve 40 min. Launch cutoff:** no job starts
unless 1.5 × its ETA ends before 15:20.

## 7. Contingencies
- Sealed phase opens / organiser news: note it, keep running; uploads are the user's.
- A smoke arm fails: fix it if < 15 min, otherwise drop that arm and say so.
- P1 runs > 1.5× late: let finished arms decide P2; report the rest.
