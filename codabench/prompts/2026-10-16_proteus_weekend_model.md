# Sprint 2026-10-16: the first real PROTEUS model (decision 7) + a second model

Time frame: 2026-10-15 18:20 EDT (the setup session's final commit 4cbc9f5 at 18:46) → **Mon 2026-10-19
08:00 EDT** (the user's "until Monday 8 AM"). The user is away; the user set the
direction as "use the setup session's results, decide a direction, run until
Monday". Chosen by the value check (§ 3). Work autonomously; decide each next step
from the previous result with the rules below; never let a job go stale (§ 6).
Read the whole brief before starting.

## 1. Context (verified 2026-10-15 evening)

- Track 2, branch `feat/codabench-track2`, commit prefix `codabench:`, push to
  `personal` after every commit. Read first: `DIRECTIONS.md`, `PROTEUS_SETUP.md`,
  `logs/proteus_setup/RESULTS.md`, `RELEASE_DAY.md` § 5 (plan + **rules v2**).
- Environment: WSL, `source ~/codabench/env040.sh` (neuralbench 0.4.0,
  `~/neuralbench/.venv040`: torch 2.14.1+cpu, braindecode 1.8.1, pyriemann 0.12;
  upstream worktree `~/neuralbench/competition-040` @ 442edcd: do not move it).
  i7-12700K, 20 threads, 31 GB RAM, no GPU. Sleep and hibernate are off (read
  2026-10-15); C: 61 GB free.
- Data: PROTEUS window cache `~/neuralbench/proteus_cache` (2,655 × 41 × 480,
  120 Hz, µV; = the official loader bit for bit). Protocols in
  `analysis/proteus_splits.py`: `lopo` (10 folds, 28 cells: the unseen-people
  half of the sealed test), `later` (session 1 of everyone → sessions 2–3 of
  P04/P10, 8 cells: the later-sessions half), `official` (8 cells; the warm-up
  test, labels public: information only), `all` (training for the final zip).
  Streamed evaluator: route A `analysis/proteus_eval.py` (`predict_streams`, the
  organisers' metrics) = route B benchopt to 6 decimals.
- What counts: the sealed phase (Oct 28 – Nov 21), test = later sessions of the 10
  people + 10 unseen people, streamed B = 1, fresh model per stream, label-free
  within-session adaptation allowed. Metric: cell-averaged balanced accuracy
  (cell = person × session × interface).
- **Already measured, do not repeat** (information only, untuned vehicle =
  pooled LDA, align none, adapt none, fb + xDAWN, no bpt4): `official` 0.406751,
  `later` 0.492382, `lopo` 0.418506 over 28 cells (fold means 0.370–0.558);
  MeanLogReg `lopo` 0.443, `later` 0.429. Fit 27–49 s, predict 8–10 ms/window.
- **Verified in the code this evening (a claim that was false):** the solver's
  `adapt="online"` keeps one running reference **per routed training person**
  (`_online_whiten`, buffers keyed by the router's index); with `align="none"`
  the training data is not centred, and there is no per-stream reference. So
  rule 6's "online re-centring" for unseen people **does not exist yet**; it is
  built in Phase 0 (`align="stream"`).
- Pre-release stand-in evidence (Scherer etc.; re-validate, do not assume):
  online re-centring +3.5 to +7.5; bpt4 +4.35; icoh +2.2 alone; EEGNet no gain.
- Rule-dependent methods: within-session label-free adaptation is **allowed**
  (FAQ 10-13). Nothing may cross a stream (a fresh copy per stream enforces it).

- **Started before this brief was committed (18:22–18:48, while the setup
  session's review still held the tree):** G-b PASS (sweep vehicle = route B,
  `later` 0.492382, `lopo` f0 0.469926; 18:22:47–18:24:06, est 2 min);
  Phase 1 arms B0/B1/R launched 18:24 (lane A); EEGNet-Stream smoke 18:28
  (10 epochs 29.5 s, ~3 s/epoch; val acc 0.529 after 5 epochs; 94 ms/window
  streamed) and lane B launched 18:29. Incident 18:38: the setup review's
  uncapped job made a third heavy process (load 37/20; one fit 40 s → 383 s);
  lane B paused 18:39:48 and resumed as that job ended.
- **Arm R deviation:** rule 4 asks for w by leave-one-run-out on session 1,
  which the solver does not implement (its `blend_w="auto"` is
  leave-one-session-out and needs ≥ 2 sessions per person). R runs with the
  pre-release fixed w = 0.5; if R comes within 1 point of adoption, build the
  leave-one-run-out choice before deciding.
- **Rule 6 deviation (warm-up):** rule 6 reads "blend with the global reference by
  n/(n+N) until n ≥ N". With N = all windows that never adapts. Implemented
  instead: running reference over the last `buffer` windows (0 = all so far),
  pulled to the global training reference along the geodesic with weight
  t = n/(n+K), K = `stream_k` = 8 fixed (not tuned).

## 2. Goal and priorities

Deliverable by Mon 08:00: `logs/proteus_w1016/RESULTS.md` (every arm, paired
bootstrap CIs), the decided **sealed candidate zip** (trained on `all`, rule-7
gates PASS, "ready, not uploaded"), a second-model verdict (ensemble zip only if
it earns it), the EDA note, updated DIRECTIONS/HANDOFF/LOG/SUBMISSIONS.

Priority order:
1. Phase 0 harness + `align="stream"` (everything else depends on it).
2. Phase 1: decision-7 base arms (bpt4; stream re-centring N; open-set blend).
3. Phase 2: one-at-a-time ablations vs the Phase-1 winner (rules 2, 3, 3c, 8).
4. Phase 4: bake the candidate, rule-7 gates, zip (by Sun 20:00 at the latest,
   so an error leaves a night to fix it).
5. Phase 3 (parallel lane B from Phase 1 on): the second model (EEGNet with the
   same stream alignment, 3 seeds) and the probability ensemble.
6. Phase 5: nested-selection check (how optimistic is choosing on `lopo`?).
7. EDA (RELEASE_DAY step 3), done by me in the waits between checks: low CPU.

## 3. Value check (recorded)

Focus = decision 7, the documented next step (DIRECTIONS § 2 #7, § 6 #1;
RELEASE_DAY § 5 steps 3–6). Not measured yet on PROTEUS; decides the sealed
model; 13 days to the first sealed upload. Compute for the Riemann half is ~1
day, so the frame also takes the second model (step 6) and a selection-bias
check. Uploading the warm-up zips is the user's action (needs you), not a goal.

## 4. Rules

- No uploads, posts, merges, tags, PRs; no downloads; no weights. Frozen:
  `submissions/`, `eegnet_steptype_wu1.py`, upstream `outputs/`, the setup
  session's zips and `riemann_sealed_proteus_pooled.py`.
- `riemann_sealed.py` defaults must stay bit-identical (gate in Phase 0): every
  new setting is opt-in.
- Never tune on test. Choices are made on `lopo` + `later` under rules v2; the
  `official` score is reported, never used to choose (its test people overlap
  `lopo` folds 3 and 8 and its labels are public).
- Evidence: cell mean on `lopo` (28) and `later` (8), paired bootstrap over
  people (10,000), people improved / 10, calibration-run vs online-run cell
  scores beside every row. Noise: a label-blind model's cell mean has sd ≈ 0.025
  on 8 cells. "No gain" said plainly. Ties → simpler. Neural: 3 seeds, mean ± sd.
- Data-shape line first in every run (people, sessions, windows, channels,
  classes, feature count); a run without it is not believed.
- ≤ 2 heavy jobs (lane A Riemann ≤ 8 threads, lane B neural ≤ 10 threads), every
  BLAS/torch pool capped, < 7 GB RSS per job. Per-lane result files only.
- Commit after each phase (code + RESULTS + LOG), never big logs/probs/zips.
  HANDOFF refreshed after each phase.

## 5. Phases and decision rules

### Phase 0: harness and stream alignment (estimate 3 h, mostly coding)

1. Solver opt-in `align="stream"` in `riemann_sealed.py`: training covariances
   centred per (person, session) (session ids from the dataset table, as the
   router path does; record whether the server's training loader exposes
   them; if not, per person, logged); global reference kept as fallback.
   Prediction: one running reference per **stream** (the deep-copied model sees
   one stream; `reset_state()` empties it), buffer N (`buffer`), warm-up blend
   W = ref of mean(n/(n+N)·running ⊕ global) until n ≥ N (rule 6); no router.
   Optional `stream_ref="interface"` is impossible without interface ids at
   test time: not built.
2. `analysis/proteus_sweep.py`: config spec → in-process fit on cache rows →
   route-A streamed evaluation → one JSON row per (config, protocol, fold) in
   `logs/proteus_w1016/<lane>.jsonl` (cell scores with interface and run type),
   test probabilities to `probs/<key>.npz`. Resumable by key.
3. Gates: (G-a) defaults bit-identical: replay of the setup's official zip =
   0.406751 and the zhou regression unchanged; (G-b) the sweep's vehicle row =
   benchopt route B for `later` 0.492382 and `lopo` fold 0 0.469926 to 6 dp;
   (G-c) `align="stream"` streamed B = 1 = a batch predict of the same stream in
   order; deep-copy + `reset_state()` → no state crosses streams (run stream 2
   alone and after stream 1: identical).
   **Done when** G-a/b/c PASS and one lopo fold of `align="stream"` ran.

### Phase 1: base arms (estimate 3 h CPU on lane A: ~11 fits × ~45 s per arm)

Arms on `lopo` (10 folds) + `later`, all pooled LDA, fb + xDAWN:
B0 vehicle (have it); B1 = B0 + bpt4; S_N = B1 + `align="stream"` with
N ∈ {16, 32, 64, 128, ∞ (all windows so far)}; R = B1 + open-set router +
personal blend (w by leave-one-run-out on session 1; for `lopo` the router's
fallback must take the held-out person → expect ≈ B1).
**Decision:** bpt4 kept unless B0 ≥ 1 point better on `lopo` (rule 3). Online:
the best-N `S_N` on `lopo` is adopted if ≥ 2 points over the clean base with
the CI excluding 0 and not losing on `later`; smaller N on a tie (rule 6). R
adopted if `later` +2 **and** `lopo` loses < 1 (rule 4). The adopted set = the
**base** for Phase 2.

### Phase 2: ablations vs the base, one at a time (estimate 8–10 h lane A)

icoh (rule 3: ≥ 2 pts, CI > 0); no xDAWN (rule 3c); `reference="car"` and an
amplitude handling (per-window robust clip at the training 99.9 % |µV|;
925 windows exceed 200 µV) (rule 8: ≥ 2, CI > 0); `subject_context` only if
its path works for unseen people, else recorded "not applicable" (rule 2);
`kind="euclid"` and the time crop **1–3 s** (before feedback onset at +3 s) as
exploratory rows: adopt only on ≥ 2 pts with CI > 0 on `lopo` and no loss on
`later`. Combine adopted changes and re-run the combination once (it must hold
≥ the best single change − 1 point, else keep the best single change).
**Decision:** the base + adopted changes = the **candidate settings**.

### Phase 3: second model, lane B (estimate 20–30 h, parallel from Phase 1)

EEGNet-ST (`xsess_lib.EEGNetSTModel`, early stopping on an inner person-held-out
split, refit) on the same stream alignment (training EA per (person, session);
streamed running EA per stream at test, same N as the base), per-channel
standardisation; 3 seeds × (10 lopo folds + later). Then, only if EEGNet's
`lopo` cell mean ≥ the Riemann candidate − 3 points: an equal-weight
probability ensemble with the Riemann candidate (no weight tuning).
**Amendment 10-15 22:00 (before seeds 34-35 exist):** EEGNet trains a fixed 80
epochs (early stopping failed, LOG); seeds 36-37 go to whichever of E1f / E0f has
the higher 3-seed `lopo` cell mean (seed 33: E0f 0.491, E1f 0.463), and that
variant (5 seeds, probabilities averaged) is the network used in the ensemble.
**Decision:** ensemble adopted if ≥ 2 points on `lopo` with the CI excluding 0
and no loss on `later`; otherwise "no gain", the Riemann candidate stands.
Smoke first: one fold, 5 epochs; time per epoch logged → full ETA.

### Phase 4: bake the candidate (estimate 1.5 h; by Sun 20:00)

Bake the candidate settings into `solvers/bci_decoding/riemann_sealed_proteus_cand.py`
(the setup's bake pattern), `proteus_candidate.sh`-style chain on `all` and
`official`: train → read-only replay → zip → unzipped replay → second training
(identical `y_pred`) → routes A/B identical. Rule 7 gates all PASS. Zip under
`logs/proteus_w1016/`, SUBMISSIONS row "ready, not uploaded". Same for the
ensemble if Phase 3 adopts it.

### Phase 5: nested-selection check (estimate 6–10 h, whichever lane is free)

For each lopo fold k, re-run the Phase 1–2 choice using only the other nine
people (inner lopo) and score the chosen settings on person k. Report the
nested `lopo` cell mean beside the (optimistic) selected one. **Decision:** if
nested < selected − 3 points, flag the candidate as over-selected in
DIRECTIONS and recommend the simpler base for the first sealed upload.

### Phase 6 (added 10-16 06:45, before any number exists): per-stream feature re-centring

Phases 1-3 finished early (the S64 candidate; nothing in Phase 2 or the ensemble
was adopted). One lever with a prior rationale is pre-registered for the spare
compute: on top of the covariance whitening, centre the **feature vectors**
(after the tangent-space and band-power blocks) per training (person, session)
and, at test, per stream with a running mean (last `buffer` windows, pulled to
the global training mean with the same pseudo-count K = 8), opt-in `fcenter`.
Arm F64 = S64 + fcenter on lopo + later. **Decision:** adopted if ≥ 2 points on
`lopo` with the CI excluding 0 and no loss on `later` (rule 6's bar); otherwise
"no gain" and the candidate stays S64. Not tuned: K and the buffer are S64's.
If adopted, the Phase 4 chain is re-run on the new bake and the nested check
is noted as covering the Phase 1 choice only.

### Phase 7 (added 10-16 09:10, before any of its numbers exist)

F64 was adopted (+8.3 lopo). Two follow-ups, nothing else:
(a) **nested check of the full recipe**: arm F64 added to the inner lopo (`nest`);
the nested rule becomes Phase 1's (B1 vs the best stream arm, ≥ 2 points, ties →
smaller N) followed by Phase 6's (fcenter on the chosen stream arm's settings is
taken if the inner F64 beats the inner chosen arm by ≥ 2 points; only F64 exists, so
fcenter is evaluated against S64 only). Report: nested lopo vs selected F64; flag if
nested < selected − 3 points.
(b) **icoh on the new base**: FI64 = F64 + icoh on lopo + later, adopted only on
≥ 2 points with the CI excluding 0 and no loss on later (rule 3). One comparison.
If adopted, the candidate chain is re-run (v4). No other lever is opened this sprint.

## 6. Keeping jobs alive

Lane scripts `scripts/proteus_w1016_<phase>.sh` on `step_lib.sh` (`.done`
markers, STATUS.md ETA rows), launched with `wsl.exe bash -lc '…'` in the
foreground of a background Bash call. Monitor on `monitor_loop.sh <logdir> 300`
re-armed until ALL DONE. Heartbeat every fold / 10 epochs. SLOW (> 1.5× ETA) or
STALLED (10 min no growth) → diagnose (CLAUDE.md), fix the root cause, re-run
(done steps skip). Nights ~1.7× slower (E-cores): revise ETAs, log it. OneDrive
can revert `.git` on power events: push after every commit.

## 7. Time budget

| Window | Lane A (Riemann) | Lane B (neural) |
|---|---|---|
| Thu eve → Fri morning | Phase 0, Phase 1 | Phase 3 smoke, then EEGNet folds |
| Fri | Phase 2 | EEGNet |
| Sat | Phase 4 bake; Phase 5 | EEGNet → ensemble |
| Sun | Phase 5; fixes | ensemble bake; spare |
| Mon 04:00 | **launch cutoff** (1.5 × ETA must end by 06:00) | |
| Mon 06:00–08:00 | **wrap-up reserve**: docs, DIRECTIONS, commit, summary | |

## 8. Contingencies

- A gate fails in Phase 0: fix before any Phase 1 number; never run on an
  unverified harness.
- `align="stream"` cannot get session ids on the server's training loader:
  train-time centring per person (logged), test-time per stream unchanged;
  measure both locally, choose under the deployment condition.
- EEGNet is near chance after the smoke (≤ 0.36 on 3 folds): stop lane B, give
  the CPU to Phase 5, record "no gain".
- Upstream or rules change (check the FAQ/Discord page once a day, read only):
  stop, record, follow the new contract.
- Anything needing the user (uploads, moving upstream, approvals): "needs you".
