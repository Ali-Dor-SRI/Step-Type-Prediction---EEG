# Plan for the next sessions: close the sealed-phase gap with 4 uploads left

Paste this whole file as the first message of the next session. Run it with the
`project-sprint` skill in two sprints: **Sprint 1 = now → today 15:00** (explore,
measure the two cheapest big levers, report, write the weekend brief), then
**Sprint 2 = the weekend → Monday 2026-11-09 12:00** (develop the next submission
model, or explore further, by the rule in § 4). Everything below was verified on
2026-11-06 between 10:10 and 10:45 EST by a planning session; re-verify the items
marked **(verify)** before relying on them.

## 0. Where we stand (verified 2026-11-06)

| Fact | Value | Source |
|---|---|---|
| Sealed upload 1 | **X4f all** (`60a0148c`), 2026-11-04 18:01 EST, Codabench id 1033809: **0.713 / 0.713 / 0.713** (cell mean / pooled / accuracy), 362.63 s | user's leaderboard row |
| Our rank | **51 of 51** on 2026-11-06 10:20 EST. Top 3: 0.828 / 0.825 / 0.825; 10th 0.806; median 0.7745; 50th 0.736 | sealed board, API call in § 0.1 |
| Final ranking rule | **best of each team's last five sealed uploads**; 1 upload per day (phase dates are UTC: Nov 4 23:01 → **Nov 21 22:59 UTC = 17:59 EST**); yaml cap 10 | `$COMPET_ROOT/codabench/pages/timeline.md`, `codabench/competition_bci.yaml` |
| Uploads left | **4** (the user's count; the yaml would allow 9 more) **(verify on the Codabench page)** | user |
| Local numbers of X4f | lopo 0.6275 / later 0.6735; nested 0.6210; per-person outer scores 0.51–0.79 | `logs/sprint_1027/RESULTS.md`, `logs/sprint_1028/RESULTS.md` |
| Sealed test size (estimate) | 362.63 s at the warm-up server's 27–30 ms/window (X4f official: 22.6 s for 755 windows) → **≈ 12–13 k windows ≈ 4.6–5× the training set ≈ 65–70 streams of ~190 windows**. Time headroom: 3,600 s / 13 k ≈ 275 ms/window | SUBMISSIONS.md, user's row |
| What the field says | fact-sheet names on the sealed board: "LDA" 0.808, "iterate_mixed" 0.816, "SDE6-SK-v1" 0.825, "Causal-Covariance-Manifold" 0.775, "plynk_ts_c198_full" 0.762, "FBCNet-ensemble-v1" 0.741; durations 293–4,075 s. **A plain linear model reaches 0.81: the gap is data and adaptation, not model capacity** | API snapshot |
| Trial timeline | official window = `start: 1.0`, `duration: 4.0`; config comment: "cue at 0 s, feedback from 3 s, **trial end at 8 s**"; cue-to-cue spacing min 10.0 s, median 12.5 s; runs hold 30–36 windows, streams 186–199 | `~/neuralbench/.venv040/lib/python3.12/site-packages/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml`; `~/neuralbench/proteus_cache/rows.parquet` |
| Preprocessing (fixed, server side) | 120 Hz, filter [0.1, 75] Hz, notch [50, 60], scale 1e6 (V → µV), no scaler, no clamp | same config.yaml |
| Label sequence | P(label = previous label) ≈ 0.28–0.30 within a run (below 1/3): **no temporal prior exists**; the 11-04 running class-balance probe also found no gain | rows.parquet; `logs/sprint_1104/RESULTS.md` |
| Raw data on disk | 224 `*_eeg.edf` runs (cued + Baseline) under `~/neuralbench/benchopt_data_040/neural_compet/Dreyer2026Proteus/download/nm000290/sub-P*/ses-*/eeg/`; the official cache config at `~/neuralbench/benchopt_data_040/neural_compet/cache/name=Dreyer2026Proteus-85ce7265/config.yaml` | ls |
| Environment | `.venv040` has mne 1.13.2, braindecode 1.8.1, torch 2.14.1+cpu; `source ~/codabench/env040.sh` inside WSL | python -c |
| Compute costs | one Riemann arm (lopo 0–9 + later) ≈ 40 min on a 4-thread lane with `OMP_WAIT_POLICY=PASSIVE` (fits ~70–100 s); EEGNet 80 epochs on 1× data ≈ 58–75 min per seed-config, ~3 s/epoch per 2.4 k windows | LOG 10-16, 10-27, 11-04 |
| Nothing running | WSL idle on 2026-11-06 10:15; branch `feat/codabench-track2` clean at 3535557 | ps, git |

**Already measured, do not repeat** (all no gain unless noted): router + personal LDA
(−4.9 lopo, −8.6 later), no xDAWN, CAR, clipping, 1–3 s crop as a *replacement*
(−3.8), Euclidean means, buffers 16 / 128 / all and K 2 / 32 (on F64; 128 / all were
+2 on later only, CI crosses 0), fscale / logreg alone, dbpt8, bpt8, slow waveform,
fbd, ROI / asymmetry / |asymmetry| / burstiness blocks, train-as-streamed (f / c / cf),
logreg C 0.003 / 0.03, online class-balance correction, log-prob averages of existing
arms (X4f + C2f −1.5), EEGNet E0f / E1f (0.474 / 0.458 lopo at 1× data, fixed 80
epochs; its ensemble with S64 +0.3), the warm-up EEGNet ensemble. Adopted: stream
re-centring, fcenter, bpt4, fb6, icoh, fscale + logreg (= X4f).

**Never tried on PROTEUS:** (a) extra *training* windows at other cue offsets (the
only augmentation in the LOG is the time-delay "augmented covariance" block, unrelated);
(b) label-free test-time adaptation of the *classifier* (class means / bias) inside a
stream; (c) a deep net trained with offset / cropped augmentation and ensembled with X4f.

### 0.1 Reading the sealed board (public API, no login)

```bash
curl -s -m 30 -o /tmp/lb_sealed.json "https://www.codabench.org/api/phases/30483/get_leaderboard/"
```

Then rank `submissions[*].scores[column_key == "balanced_accuracy_cell_mean"]`
descending (`owner`, `duration`, `fact_sheet_answers.model`, `created_when`). Phase
ids: warm-up 30145 and 30961, sealed 30483; competition 17982. Snapshot it to
`logs/sprint_1106/leaderboard_sealed_<date>.json` (git-ignored) and record the top 3,
median and our rank in SUBMISSIONS.md.

## 1. Strategy for the 4 remaining uploads

1. **A worse upload cannot lower the rank** (best of the last five; we have made one,
   so all five count). Every slot is therefore a free draw, but a *wasted* slot is still
   a slot: never spend one on a ≤ 2-point X4f variant. The field is 2.3–11.5 points
   ahead; only model-family changes (more training windows, test-time adaptation, a
   second model family, stacked) can move us, so every upload carries one of those.
2. **Each sealed score is a measurement.** Record Δsealed next to Δlocal (lopo, later)
   for every upload. If upload 2 gains locally but not on the board, local selection
   is over-fitting the 10 people: the next upload must then be a nested-checked,
   structurally different model, not a refinement.
3. **No warm-up check exists any more** (closed Nov 1): a crash or timeout burns a slot.
   Every zip passes the local gates that predicted the server exactly four times
   (`proteus_w1016_gates.py --submission` ALL PASS, unzipped replay = training, threads
   pinned to 1, ms/window × 13 k ≤ 20 min, sha256 recorded) before it is recommended.
4. **Calendar** (1 upload per UTC day; 18:01 EST = 23:01 UTC, so the "day" turns at
   19:00 EST):
   - **Slot 2, Tue Nov 10:** the weekend winner, if it is at least *lean* (Δ lopo ≥ +1.0,
     CI lower > 0) and passes the gates; otherwise keep the slot.
   - **Slot 3, ~Sat Nov 14:** the next sprint's stacked lever (e.g. augmentation +
     adaptation, or + the deep-net ensemble).
   - **Slot 4, ~Wed Nov 18:** the best stack after its nested check.
   - **Slot 5, by Fri Nov 20 18:00 EST**, never in the last hours (the Oct 26 host outage
     lasted days): the final best. Re-uploading an identical model is pointless (the
     pipeline is deterministic); if nothing new is ready, use the slot on the highest-
     variance credible variant, since it cannot hurt.
   - Keep one slot in hand after any failed run, to re-upload the fixed zip next day.
5. **Fact sheet:** the board's "Fact Sheet Answers" column is a free-text model name
   typed on the upload form; suggest a short public-safe name for each zip
   (e.g. `X4f-aug4`) in SUBMISSIONS.md. The user types it.
6. **Audit readiness** (top 3 only; a long shot): every candidate keeps its bake script,
   seed and config committed, as now.

## 2. Sprint 1: 2026-11-06, now → 15:00 (frame ≈ 4 h; wrap-up reserve 14:15–15:00)

Invoke: `/project-sprint until 15:00 today  feasibility of the two cheapest big levers
(§ 2 of prompts/2026-11-06_next_session_plan.md), then write the weekend brief`.
Standing constraints of the skill apply (no uploads, no merges / tags, push to
`personal` and `org` after every commit, defaults bit-identical, estimate / time-log /
investigate overruns). Launch cutoff: nothing starts after 13:30 unless 1.5× its ETA
ends by 14:15.

### P0 Orient (≤ 25 min, done by ~11:30)
- Read DIRECTIONS § 1–2, HANDOFF § 00 / § 0, SUBMISSIONS.md (last three rows), this file.
- Verify: the board (§ 0.1), `git status` clean, nothing running in WSL, the sleep
  settings (`powercfg /query SCHEME_CURRENT SUB_SLEEP`; ask, do not change).
- Record the sealed row now (it must survive a dead session): SUBMISSIONS.md X4f-all row
  gets the score, id, duration and rank; one LOG line.

### P1 Augmented training windows: build the cache and prove it exact (ETA 60–90 min)
Goal: windows at other cue offsets, produced by **the official loader itself** so the
preprocessing is identical, in a **new** cache dir. `~/neuralbench/proteus_cache` and
`~/neuralbench/benchopt_data_040/…/cache` are frozen: never overwrite or invalidate
them (the neuralset cache is keyed by config hash; point the augmented build at its
own `infra.folder`, e.g. `~/neuralbench/proteus_cache_aug/`).
- Route 1 (preferred): `analysis/proteus_cache.py build` calls the loader "exactly as
  `bci_studies.py` does"; find where the stream config's `data.start` enters and
  parameterise a copy of the build by `start`. Offsets `o ∈ {0.0, 0.5, 1.5, 2.0, 2.5,
  3.0, 3.5, 4.0}` keep `[cue + o, cue + o + 4)` inside the 8-s trial. Store `X_<o>.npy` +
  `rows_<o>.parquet` with the same stable keys plus `offset`.
- **Gate A0:** the new path at `start = 1.0` reproduces the official cache **bit for bit**
  (max |diff| = 0.0 on all 2,655 rows, labels and keys equal). This proves every other
  offset is exact too. **Gate A1:** 2,655 rows per offset, labels equal to the official
  per (person, session, run, cue), no NaN.
- Route 2 (only if the loader cannot take an offset): MNE on the EDFs with the
  config's filters (0.1–75 Hz, notch 50 / 60, 120 Hz, µV), gate = at `start = 1.0`
  corr ≥ 0.999 and max |diff| reported; label the lane "approximate" in RESULTS.
- Decision: A0 pass → P2 runs; A0 fail and route 2 corr < 0.99 → the augmentation lane
  is dropped for now (report why), P2 is skipped, P3 gets the lane.

### P2 First augmented arm (launch by 13:15; ETA 45–60 min on 2 lanes)
- Solver opt-in `train_offsets=""` (default, bit-identical: zhou `EQUAL`, X4f re-run
  equal) → `"0.5_1.5_2.0_2.5"`: the training rows get the same session's windows at
  those offsets **for training people / sessions only** (lopo: the 9 training people;
  later: session 1 of everyone; never a test person's or test session's window at any
  offset). Per-session centring at training uses all of that session's rows. Test side
  unchanged (official windows, streamed).
- Arms: `X4f+aug4`, `C2f+aug4` (LDA may profit more from 5× rows). lopo 0–9 + later,
  paired people bootstrap (10,000), vs X4f / C2f.
- **Decision (rules v2):** adopted if Δ lopo ≥ +2.0 with CI lower > 0 and Δ later > 0;
  lean if Δ lopo ≥ +1.0 with CI lower > 0; else no gain. Report fit time at 5× rows
  (sets the weekend ETAs).

### P3 Deep-net smoke, parallel lane, optional (≤ 45 min, 4 threads)
braindecode `ShallowFBCSPNet` and `EEGNetv4` on lopo fold 0 + later, 1× official
windows, per-window z-score (the E0f setting, which beat stream alignment for nets),
60 epochs, seed 33. Log s/epoch and the cell means; compare with X4f fold 0 = 0.6954
(`logs/sprint_1028/RESULTS.md` nested table) and later 0.6735, and with E0f 0.474
(lopo mean). Purpose: the weekend lane's ETA and whether a net is within ~8 points.

### P4 Wrap-up (14:15–15:00)
`logs/sprint_1106/RESULTS.md`; LOG entry (estimate vs actual per phase); SUBMISSIONS
(rank, fact-sheet name suggestion); DIRECTIONS § 1 / § 2 (decision 15: the upload plan)
/ § 6 / change log; HANDOFF § 00; commit + push both remotes. **Write the weekend brief**
`prompts/2026-11-06_weekend_<slug>.md` from the skill template with the branch chosen by
§ 4, pre-registered rules, ETAs re-estimated from P2 / P3 timings. Final summary to the
user: findings with numbers, the branch recommendation ("develop" or "explore"), the
upload calendar, what needs them (§ 5).

## 3. Sprint 2: the weekend, start on the user's go (or 15:00) → Mon 2026-11-09 12:00

Reserve Mon 10:00–12:00. Night runs are up to 1.7× slower (LOG 09-24). ≤ 3 lanes ×
4 threads, `OMP_WAIT_POLICY=PASSIVE`, scoring at 1 thread; resumable step scripts with
`.done` markers, `monitor_loop.sh` inside WSL, STATUS.md ETA rows.

### Branch A, "develop the next submission model" (default when P2 is adopted or lean)
- **W1 Augmentation design** (Fri evening → Sat morning; ≈ 6 arms × 45–90 min / 3 lanes):
  on the better P2 base: `aug4` {0.5, 1.5, 2.0, 2.5}; `aug8` {0.0, 0.5, …, 4.0} \ {1.0};
  `aug-near` {0.75, 1.25}; on both X4f and C2f if P2 was close. Pick by rules v2 vs the
  base; ties → fewer offsets, then the simpler classifier. Also report calib / online
  cells and per-class recall (`sprint1028_perclass.py`).
- **W2 Label-free test-time adaptation** (Sat): add a `--dump` to `proteus_sweep.py`
  that saves each test window's *streamed* feature vector (after fcenter / fscale) and
  logits in stream order, once, for the W1 winner on lopo 0–9 + later. Then post-hoc
  probes (minutes each, no refit; baseline must reproduce the winner exactly):
  (a) pseudo-label class-mean update: after n ≥ 32 windows, move each class mean toward
  the mean of that stream's confident windows (p_max ≥ τ ∈ {0.5, 0.6, 0.7}), shrinkage
  λ ∈ {0.3, 0.5}, causal, LDA-style bias update of the linear scores; (b) bias-only
  entropy minimisation of the logreg per stream (a few gradient steps per window on
  the 3 biases, lr small); (c) a and b together. Rules v2 vs the W1 winner. If adopted
  or lean: implement in `_predict_block` (stateful, per stream, reset in
  `reset_state()`), gate = implementation reproduces the probe bit for bit on lopo
  fold 0 + later, and measure ms/window (must stay ≤ 100 ms).
- **W3 Deep-net lane** (Sat → Sun, one lane, only if P3 landed within 8 points of X4f on
  fold 0): `ShallowFBCSPNet` with `aug8` training rows (= cropped training), per-window
  z-score, early stopping on one inner person, 3 seeds; lopo 0–9 + later. Ensemble with
  the W2 winner: log-prob average, weight w ∈ {0.3, 0.5} chosen inside the 9 training
  people (nest), never on the test person. Rule: the net alone must be ≥ X4f − 5 to be
  ensembled; the ensemble is adopted at ≥ +2.0 over the W2 winner with CI lower > 0 and
  later > 0.
- **W4 Nested check** (Sun): extend `analysis/sprint1028_nested.py`'s walk
  F64 → fb6 → C2f → X4f → +aug → +adapt → +ens; a step replaces the pick when inner
  ≥ pick + 1.0; **flag over-selection if nested lopo < selected − 3** (then the step is
  not recommended for upload).
- **W5 Bake** (Sun night → Mon 10:00): thread-pinned all-trained zip of the surviving
  config, `scripts/sprint1027_bake.sh` pattern, every gate PASS, ms/window × 13 k ≤
  20 min, sha256, SUBMISSIONS row "ready, not uploaded" with a fact-sheet name; LOG,
  DIRECTIONS (decision 15 updated with the recommendation), HANDOFF § 00; final summary
  with the Tue Nov 10 upload recommendation and the expected local → sealed reading.

### Branch B, "explore further" (when P2 is no gain or the cache gate failed)
W1 is dropped. W2 (adaptation probes on X4f itself) and W3 (the deep-net lane, with
`aug8` if the cache exists, else 1× data) share the weekend; W4 / W5 run only for a
config that is adopted. The slot-2 upload moves to the following week.

### Rules for both branches
- Evidence: paired people bootstrap over the 10 lopo people; `later` (2 people) reported,
  never decisive alone; seed 42 for Riemann, seeds 33–35 for nets; nothing tuned on a test
  person or test session; say "no gain" plainly.
- Defaults stay bit-identical (zhou `EQUAL`, X4f re-run equal to `K2.jsonl` to 6 dp on lopo
  fold 0 + later) after every solver edit.
- Every new predict-path element is causal, label-free and session-bounded (FAQ:
  "statistics-based alignment or gradient updates inside predict" are allowed; nothing
  crosses `reset_state()`).
- Commit after every phase (`codabench:` prefix), push to `personal` and `org`;
  `logs/` ignored except `RESULTS*.md` (force-add).

## 4. The 15:00 decision rule (pre-registered)

| P2 result (best of X4f+aug4, C2f+aug4) | Weekend branch | Slot 2 |
|---|---|---|
| adopted (Δ lopo ≥ +2.0, CI > 0, later > 0) | **A, develop**; W1 → W2 → W3 (if P3 ok) → W4 → W5 | Tue Nov 10, the W5 zip |
| lean (+1.0 ≤ Δ < +2.0, CI > 0) | **A, develop**, with `aug8` and C2f-LDA weighted up in W1; W3 mandatory if P3 ran | Tue Nov 10 if W4 holds, else hold |
| no gain, or cache gate failed | **B, explore** (adaptation + deep net) | held for the next week |

The user may override at 15:00; if unreachable, the table decides.

## 5. Needs the user

1. Confirm the remaining-upload counter on the Codabench submission page (4?).
2. Keep the machine on AC and awake Fri → Mon (idle sleep has killed runs here); the
   session reads the power settings and asks before any change.
3. The Tue Nov 10 upload (and its fact-sheet name) is the user's action.
4. Be reachable Mon 12:00 for the weekend report, or say now that the § 4 table decides.
