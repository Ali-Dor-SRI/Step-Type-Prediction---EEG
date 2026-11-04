# Sprint 2026-11-04 (overnight): train as streamed

Frame: 2026-11-04 17:20 → 2026-11-05 08:00 EST (the user's "overnight"). Wrap-up
reserve 06:45–08:00. The user uploads **X4f all** (`60a0148c`) to the sealed phase at
18:01; this sprint feeds sealed uploads 2–10 (1/day to Nov 21).

## 1. Verified context (17:00–17:20)

- X4f (lopo 0.6275 / later 0.6735; nested 0.6210) is the sealed model; C2f fallback.
  Every number: `logs/sprint_1027/RESULTS.md`, `logs/sprint_1028/RESULTS.md`.
- **Already measured, not repeated:** task-driven blocks (roi / asym / asymabs / burst,
  10-28), finer bins, slow waveform, delta bins, buffers 16 / all and K 2 / 32 (on F64),
  EEGNet and its ensembles, router + personal LDA, CAR, clipping, crop.
- **Probes today (no refit, from the saved streamed probabilities,
  `analysis/sprint1104_posthoc.py`, `analysis/sprint1104_probe2.py`; baselines
  reproduce 0.6275 / 0.6735 exactly):**
  - online label-free class-balance correction (divide by the running mean of the
    model's own probabilities): **no gain**, −0.66 to +0.15 lopo over 6 settings;
  - log-probability averages of existing arms: **no gain** (X4f + X4f+asym +0.38 lopo;
    X4f + C2f −1.50);
  - X4f lopo accuracy by stream position: 0.56 in windows 0–15, then 0.58–0.69
    with no upward trend: the warm-up costs ≤ ~1 point overall.
- **Code fact (checked in `riemann_sealed.py`):** at training, each (person, session)
  is whitened by its *full-session* mean covariance (`align="stream"` branch of `fit`)
  and its features centred / scaled by the *full-session* mean / SD (`fcenter`,
  `fscale`). At test, each stream uses a *causal running* estimate (last 64 windows,
  log-Euclidean, prior weight K = 8 on the global mean: `_stream_whiten`,
  `_stream_fcenter`). The classifier never sees the noisier running-centred inputs it
  gets on the server. Not tested before (LOG search: no hit).

## 2. Goal and priority

1. Measure **train-as-streamed** (`trainstream`, new training-only opt-in: each
   training session replayed through the same causal estimator the test streams use,
   by calling the test-time methods themselves): `f` features, `c` covariances, `cf` both.
2. Cheap side arms: logistic-regression strength C ∈ {0.003, 0.03} on X4f (C = 0.01
   was the only value ever run).
3. If something passes: nested check, then thread-pinned official + all zips.

## 3. Deliverable

`logs/sprint_1104/RESULTS.md`; LOG entry; DIRECTIONS § 1/2/4/6 + change log; HANDOFF § 0;
if a config survives Phase 3, zips in `logs/proteus_w1016/proteus_<c>_cand_{official,all}/`
+ SUBMISSIONS rows (not uploaded: uploads are the user's).

## 4. Rules

- Branch `feat/codabench-track2`, prefix `codabench:`, push to `personal` and `org`.
  No uploads, tags, merges, PRs. `logs/` ignored except `RESULTS*.md` (force-add).
- Defaults bit-identical: zhou regression `EQUAL`; X4f re-run equals the K2.jsonl X4f
  rows to 6 dp on lopo fold 0 + later.
- Evidence: paired people bootstrap over the 10 lopo people (10,000); `later`
  (2 people) reported, never decisive alone. Seed 42. No setting tuned on lopo / later
  beyond the arm list.
- ≤ 3 lanes × 4 threads, `OMP_WAIT_POLICY=PASSIVE`, scoring at 1 thread.
- Estimate, time-log, investigate overruns > 1.5× (CLAUDE.md).

## 5. Phases and decision rules

### Phase 0: implement + smoke (done 17:20 coding; runs ≈ 15 min)
`trainstream` in `riemann_sealed.py` (`_replay_groups`). **Done when:** zhou EQUAL,
X4f re-run equal, each arm > 0.4 on `later`, stateful gates PASS for X4f + `cf`.

### Phase 1: single arms on X4f (ETA ≈ 1.5 h; 3 lanes × 2 arms, ~40 min/arm)
`X4f+tsf`, `X4f+tsc`, `X4f+tscf`, `X4f C0.003`, `X4f C0.03`; secondary (reported):
`C2f+tscf`. lopo 0–9 + later.
**Decision (rules v2, vs X4f):** adopted if Δ lopo ≥ +2.0 with CI lower bound > 0 and
Δ later > 0; lean if Δ lopo ≥ +1.0 with CI lower bound > 0; else no gain.

### Phase 2: follow-up (ETA ≈ 40 min) — only if a ts arm is adopted or lean
Best ts arm (larger lopo; ties → simpler, f < c < cf) with buffer 128, and with the
better C if that C arm is lean. Same rule vs the best ts arm (bar +1.0 lean).

### Phase 3: nested check (ETA ≈ 2–2.5 h on 3 lanes) — only if Phase 1–2 adopted or lean
Inner folds for the passing config(s) (X4f, C2f, fb6, F64 inner rows exist).
`analysis/sprint1028_nested.py` walk extended: F64 → fb6 → C2f → X4f → new config(s);
a config replaces the pick when inner ≥ pick + 1.0 and ≥ F64 + 2.0.
**Flag over-selection** if nested lopo < selected lopo − 3 points; then the new config
is not recommended for the sealed phase.

### Phase 4: bake (≈ 15 min per zip pair) — only if a config survives Phase 3
`scripts/sprint1027_bake.sh` with the config baked, threads = 1, every gate PASS,
ms/window. The official zip goes to the warm-up only if it is still open; otherwise the
format is the same code path as X4f (only training changed), and the all zip is a
sealed candidate on the user's call.

## 6. Keep jobs alive

Phase scripts on `step_lib.sh` (`.done` markers, resumable rows per lane:
`T0`, `T1P/Q/R`, `T2*`, `NT*`), launched in the foreground of a background
`wsl.exe bash -lc`, watched with `monitor_loop.sh` inside WSL. Sleep: AC and DC idle
sleep = never (checked 17:18).

## 7. Time budget

Phase 0 to 17:40; Phase 1 to ~19:30 (night slowdown up to 1.7× → 20:15); Phase 2 to
~21:00; Phase 3 to ~23:30 (→ 01:30 slow); Phase 4 to ~02:00. Launch cutoff: no new job
unless 1.5× its ETA ends before 06:45.

## 8. Contingencies

- Phase 0 fails (not equal): fix before anything else; ≤ 45 min, else stop and report.
- Nothing lean in Phase 1: skip 2–4; use the remaining night for one more pre-registered
  arm only if written here first as an addendum with its rule.
- The user's sealed X4f score arrives: note it in SUBMISSIONS / DIRECTIONS; keep running.

## 9. Addenda (time-stamped, each before the numbers it concerns)

- 2026-11-04 18:20: no deviation. Phase 1 done in 51 min; no arm adopted or lean, so Phases 2-4 are skipped (§ 5 rules). No extra arm added; sprint closes early.
