# Directions: Track 2 at a glance

**Updated 2026-10-27 17:35**: new candidates over F64T (lopo 0.567 / later 0.600):
**C2f 0.608 / 0.642** (rule-selected, 10/10 people up; **warm-up exact match, 22.7 s**)
and **X4f 0.628 / 0.674** (best, pending the nested check). Next: **C2f all = first sealed
upload**. Warm-up open to **Nov 1**; sealed phase from Oct 28. Nothing running. The F64 warm-up upload
**matched exactly** (0.507944 / 0.506931 / 0.507285) but ran **263 ms/window** on
the server (32× local). **Cause found and fixed:** default thread pools; the
thread-pinned F64T gives the same predictions at 9.4 ms/window. F64T zips ready.
MODEL_ARCHITECTURE.md now describes F64. Nothing of ours is running. This is the short version for
making decisions; every number links to where it comes from (§ 7). Kept current
by every session that changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:** registration closed **Oct 24**; the warm-up (5 uploads/day, does
  not rank) was **extended to Nov 1** after the host outage; the **sealed phase,
  Oct 28 – Nov 21** (start date not re-confirmed since the extension), is the only
  phase that ranks: 1 upload/day, 10 in total, top 3 audited.
- **Data: PROTEUS.** 10 people, 14 sessions, 2,655 windows, 41 EEG channels,
  120 Hz, 3 balanced classes. Sealed test = later sessions of these 10 people
  **plus 10 unseen people**, streamed one window at a time per session, label-free
  adaptation inside a session allowed.
- **How we measure (setup 10-15, [PROTEUS_SETUP.md](PROTEUS_SETUP.md)):** `lopo`
  (leave one person out, 28 cells: the unseen-people half) and `later` (session 1 of
  everyone → sessions 2–3 of P04/P10, 8 cells: the later-sessions half), cell-averaged
  balanced accuracy, paired bootstrap over people. Chance = 0.333.
- **Sprint 10-16 result ([LOG](LOG.md) "2026-10-16 sprint"):**

  | model | lopo | later |
  |---|---|---|
  | setup vehicle (pooled LDA, no bpt4) | 0.419 | 0.492 |
  | + bpt4 | 0.427 | 0.528 |
  | + covariance re-centring per stream (S64) | 0.484 | 0.530 |
  | **+ feature re-centring per stream (F64) = the candidate** | **0.567** | **0.600** |
  | EEGNet (5 seeds) / EEGNet + Riemann ensemble | 0.485 / 0.487 | 0.476 / 0.533 |

  Both re-centrings use only the session's own unlabelled windows so far (allowed).
  F64 beats S64 by +8.3 points on lopo (CI +4.5 to +12.1, 10/10 people up) and +7.0 on
  later. No gain: router + personal LDA (−4.9), dropping xDAWN, CAR, amplitude clipping,
  cropping to 1–3 s, Euclidean means, the EEGNet ensemble, other buffers, other K.
  icoh on F64: +1.65 lopo (CI +0.4 to +2.9), below the 2-point bar (variant zip ready).
- **Is the choice over-fitted to these 10 people?** No: re-making every choice inside
  nine people and testing on the tenth gives lopo **0.567** (vs 0.567 selected).
- **Zips ([SUBMISSIONS.md](SUBMISSIONS.md)):** the official-trained F64 zip was
  **uploaded to the warm-up on 10-26: exact match** (0.507944 / 0.506931 /
  0.507285), 198 s for 755 windows. The F64 sealed zip (trained on all 14 sessions) is
  ready, not uploaded. Sealed test runtime ≈ 3 min of the 60 allowed (8.2 ms/window
  on one quiet thread).
- **What the data say ([EDA](reports/proteus_eda/EDA.md)):** every session is its own
  domain (two sessions of one person differ as much as two people), which is why
  re-centring per session carries the model; the class signal is strongest in frontal
  theta (AF4), which may be partly eye movement.

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 13 | **New model for the sealed phase.** C2f = F64T + 6-band filter bank with delta + icoh: lopo **+4.1** [+3.1, +5.2], 10/10 people, later +4.2 (rule-selected). X4f = C2f + feature scaling + logistic regression: lopo **+6.0** [+4.3, +7.8], 10/10, later +7.3 (parts picked after seeing them) | (1) ~~C2f official warm-up~~ **done 17:29: exact match, 22.7 s**; (2) **first sealed upload = C2f all** (`submissions/riemann_sealed_proteus_c2f_all_2026-10-27.zip`, sha256 `7059b103`); (3) run the nested check on X4f (~3 h); if it holds, X4f all is the next sealed upload | ~4–6 points on the ranked board vs F64T | two uploads + a go for the nested run |
| 14 | **Feature-engineering sprint** (brief `prompts/2026-10-28_feature_engineering.md`, pre-registered, not run): 4 compact task-driven blocks on C2f (ROI relative power, hemispheric asymmetry, handedness-free asymmetry magnitude, burstiness; eye channels excluded), then one nested check that also settles X4f | Go when convenient: ~4.5 h for the arms, then ~5 h unattended for the nested check; upload C2f all meanwhile | Expected 0–2 points; the nested check is needed anyway for X4f | a time slot |
| 9 | ~~Warm-up upload of F64~~ done: exact match; 263 ms/window on the server | — | — | — |
| 12 | ~~Runtime~~ **fixed**: F64T warm-up 10-27 = exact match in **19.0 s** (F64 198.3 s). Was: **Runtime:** F64 at default threads = 263 ms/window (≈ 92 min worst case). F64T (1 thread) = identical predictions, 9.4 ms/window locally | Upload `logs/threads_1027/f64t_official/…_2026-10-27.zip` (sha256 `fcfd7c62`) to the warm-up; if it shows the same three numbers in ≲ 1 min, upload the F64T **all** zip (`a774135a`) as the first sealed one | A timeout wastes a sealed upload | one warm-up upload |
| 10 | **First sealed upload = the C2f all-trained zip** (was F64T all) | Accept, after a C2f warm-up check (decision 13) | It is the best measured model; later uploads can test variants | yes / no |
| 11 | **Sealed upload plan:** F64 first; FI64 (+ icoh) as the first variant | Accept; keep ≥ 5 of the 10 uploads unused until the last week | icoh leaned positive twice (+1.1, +1.65) but never cleared the bar | yes / no |
| 8 | Adopt the 0.4.0 setup as the main one | Accept (done in practice) | One environment, one set of docs | yes / no |
| 5 | Optional: REVE/LaBraM weights; merge `docs/sprint0928-consistency` | Low priority | REVE/LaBraM are hours per epoch here | — |
| 6 | **Thesis: what is the One/Two prompt, and what is the `96` trigger?** ([results](../docs/sprints/2026-10-02_window_effect_check_RESULTS.md), branch `feat/window-effect-check`) | Tell me the prompt's form | Protects a public number | a short answer |
| 7 | ~~The first real model on PROTEUS~~ decided 10-16 by the pre-registered rules: F64 | — | — | — |
| 3, 4 | ~~bpt4 by default; strict CV references~~ bpt4 kept by rule 3 (10-16); the personal-LDA path is not used | — | — | — |

Not decisions, still yours: upstream `origin/main` moved past 442edcd (to a66bc23); we
stay on 442edcd until you say to move it. The older warm-up zips (Riemann vehicle,
MeanLogReg) are superseded by the F64 one for the format test.

## 3. The model in one picture (F64; C2f adds a 6-band filter bank with delta + icoh, X4f also feature scaling + logistic regression)

Each test session is a stream. Every window's covariance is whitened by a running
reference of that session's windows so far (last 64, pulled towards the training
mean), then the usual features (filter-bank Riemannian tangent space + xDAWN +
log-variance + bpt4 band-power time course) are centred on the session's running
feature mean, and one pooled shrinkage LDA, trained on all ten people centred the
same way per session, classifies. No person ids, no router. Detail:
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) (rewritten for F64 on 10-26, with
real-data figures: session distances, how fast the reference settles, what each
re-centring does to the features).

## 4. What we know (on PROTEUS, sprint 10-16; 10-27 results: `logs/sprint_1027/RESULTS.md`)

| Lever | Effect (lopo / later, points) | Status |
|---|---|---|
| Feature re-centring per stream | +8.3 / +7.0 over S64 | **in the candidate** |
| Covariance re-centring per stream (buffer 64) | +5.7 / +0.2 over bpt4 base | **in the candidate** |
| bpt4 band-power time course | +0.8 / +3.6 | **in the candidate** |
| icoh | +1.65 / +2.35 on F64 (below the 2-point bar) | variant zip FI64 ready |
| Buffer 16 / all; K 2 / 32 | −1.8 / −0.3; −1.0 / −2.4 on lopo | F64's 64 and 8 kept |
| No gain | router + personal LDA, no xDAWN, CAR, clipping, 1–3 s crop, Euclidean means, EEGNet ensemble | dropped |

The 1–3 s crop loses also in the calibration runs (no feedback): the window's last two
seconds carry task signal, not only feedback.

## 5. Bottlenecks and risks

1. **Server format of an adaptive model** on image `v3`: being tested by the 10-26
   warm-up upload (decision 9).
2. **Small data:** 10 people; `later` rests on 2 people (CIs wide). The nested check
   found no over-selection (0.567 nested vs 0.567 selected).
3. **Timing under load:** 8 ms/window quiet, up to ~190 ms/window when the CPU is
   oversubscribed (≈ 66 min projected). The server's CPU is unknown; the warm-up
   upload reports the real time.
4. **Rules: settled** (FAQ 10-13): within-session, label-free adaptation is allowed.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | **Nested check of X4f** (folded into the feature sprint, decision 14; run alone if the sprint waits) (re-select among F64 / fb6 / C2f / X4f inside 9 people) | decides whether X4f (+1.9 over C2f) is real | ~3 h unattended | your go |
| 2 | ~~Update MODEL_ARCHITECTURE.md for F64~~ done 10-26 | — | — | — |
| 3 | Frontal signal: no EOG in the raw BIDS; delta + finer bins tried 10-27 (dbpt8 / bpt8: no gain; delta in the filter bank: part of C2f). Left: ICA on resting runs for the write-up | honesty, not score | ~2–3 h | — |
| 4 | Thesis: XGB on the 0.5–2 s window | — | ~3–4 h | decision 6 |

Not worth it now: EEGNet variants (no gain alone or in the ensemble), REVE/LaBraM on
this CPU, the personal-LDA path (−4.9).

## 7. Where to read more

| Question | Document |
|---|---|
| This sprint: plan, rules, every arm and decision | [prompts/2026-10-16_proteus_weekend_model.md](prompts/2026-10-16_proteus_weekend_model.md), [LOG.md](LOG.md) "2026-10-16 sprint", `logs/proteus_w1016/RESULTS_p*.md` |
| What exists on PROTEUS, how to run it | [PROTEUS_SETUP.md](PROTEUS_SETUP.md); [HANDOFF.md](HANDOFF.md) top section |
| What was uploaded, what is ready? | [SUBMISSIONS.md](SUBMISSIONS.md) |
| Released data, rules, pre-registered rules v2 | [RELEASE_DAY.md](RELEASE_DAY.md) |
| Neuroscience of the three tasks | [reports/PROTEUS_task_neuroscience_2026-10-15.docx](reports/PROTEUS_task_neuroscience_2026-10-15.docx) |
| The pre-release model and its stand-in evidence | [SEALED_RECIPE.md](SEALED_RECIPE.md) |

## 8. Keeping this file current

- **When:** at the end of every sprint or session that changes a decision, a
  number, the plan or the competition calendar.
- **Size:** under ~150 lines. Detail goes to the documents in § 7, not here.
- **Numbers:** each one must match its source document.
- **Change log** (newest first, one line each):
  - 2026-10-28: feature-engineering sprint brief written and pre-registered (decision 14), not run. No number changed.
  - 2026-10-27 17:35: C2f official warm-up = exact match (0.527602 / 0.526610 / 0.527152), 22.7 s. Upload zips copied to `submissions/` (git-ignored). HANDOFF top rewritten for a new session.
  - 2026-10-27 17:15: sprint complete. fb6 adopted (+2.5); C2f = fb6 + icoh +4.1 (10/10); X4f = C2f + fscale + logreg +6.0 (10/10), pending nested. X1 superseded. Zips for X1 / C2f / X4f baked, all gates pass. Decision 13 rewritten; decision 10 now C2f all.
  - 2026-10-27 13:15: accuracy sprint paused for the user. X1 (icoh + fscale + logreg) lopo 0.5995 (+3.22, CI excludes 0) / later 0.6458; single arms fscale, logreg, delta bins, finer bins, slow waveform no gain; delta in the filter bank lean (+1.09). Warm-up 0.988 of the all-trained zip is memorisation. Decision 13 new.
  - 2026-10-27 (F64T warm-up): exact match 0.507944 / 0.506931 / 0.507285 in 19.0 s on the server (F64: 198.3 s); decision 12 closed. Next: accuracy work.
  - 2026-10-27 (F64T): cause of the slow server run = default thread pools (reproduced locally, 264.7 ms/window); F64T pins them to 1: 9.4 ms/window, 0 of 755 predictions differ; F64T zips built; decisions 10 and 12 updated.
  - 2026-10-27 (later): F64 warm-up exact match, 198.3 s (263 ms/window, 32× local): decision 9 closed, decision 12 (runtime) new.
  - 2026-10-27: no change to decisions or numbers. HANDOFF.md rewritten as a complete state of record for a new agent; LOG back-filled for 10-26.
  - 2026-10-26 (later): host outage; warm-up extended to Nov 1; official-trained F64 zip uploaded to the warm-up (result pending), so the "nothing uploaded" note below is superseded. MODEL_ARCHITECTURE.md rewritten for F64 with PROTEUS figures (step 2 done).
  - 2026-10-26: no change to decisions. Raw BIDS has no EOG (41 EEG only), so step 3 is now indirect: the frontal task signal falls with distance from the eyes, is as strong in delta as in theta and peaks in the second after feedback onset (possibly ocular). Practice phase closed Oct 25 with nothing uploaded.
  - 2026-10-16 (evening): sprint complete. Nested check lopo 0.567 (no over-selection); icoh on F64 +1.65 (FI64 variant zip); buffer and K alternatives no gain; EDA; decision 11 new. Nothing running.
  - older entries (10-02 … 10-16 morning): see git history of this file.
