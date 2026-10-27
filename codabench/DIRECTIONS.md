# Directions: Track 2 at a glance

**Updated 2026-10-27**: the warm-up was extended to **Nov 1** after a host outage.
The 10-26 warm-up upload turned out to be the **old Dreyer zip by mistake** (0.33 =
chance, 18 min); the F64 zip is **still not uploaded**, and the sealed zip is ready.
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
- **Zips ([SUBMISSIONS.md](SUBMISSIONS.md)):** the official-trained F64 zip is
  **not uploaded yet** (10-26 sent the Dreyer zip by mistake; F64 must show
  0.507944 / 0.506931 / 0.507285). The F64 sealed zip (trained on all 14 sessions) is
  ready, not uploaded. Sealed test runtime ≈ 3 min of the 60 allowed (8.2 ms/window
  on one quiet thread).
- **What the data say ([EDA](reports/proteus_eda/EDA.md)):** every session is its own
  domain (two sessions of one person differ as much as two people), which is why
  re-centring per session carries the model; the class signal is strongest in frontal
  theta (AF4), which may be partly eye movement.

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 9 | **Warm-up upload of the official-trained F64 zip**: not done (10-26 sent the Dreyer zip) | Upload `logs/proteus_w1016/proteus_f64_cand_official/…_official_2026-10-16.zip` (sha256 starts `d4b42478`) **today**, then send me the three numbers and the run time | Proves the adaptive, streamed model runs on the `v3` image before the 10 sealed uploads | read three numbers |
| 10 | **First sealed upload (Oct 28) = the F64 all-trained zip** | Accept, if the warm-up numbers match | It is the best measured model; later uploads can test variants | yes / no |
| 11 | **Sealed upload plan:** F64 first; FI64 (+ icoh) as the first variant | Accept; keep ≥ 5 of the 10 uploads unused until the last week | icoh leaned positive twice (+1.1, +1.65) but never cleared the bar | yes / no |
| 8 | Adopt the 0.4.0 setup as the main one | Accept (done in practice) | One environment, one set of docs | yes / no |
| 5 | Optional: REVE/LaBraM weights; merge `docs/sprint0928-consistency` | Low priority | REVE/LaBraM are hours per epoch here | — |
| 6 | **Thesis: what is the One/Two prompt, and what is the `96` trigger?** ([results](../docs/sprints/2026-10-02_window_effect_check_RESULTS.md), branch `feat/window-effect-check`) | Tell me the prompt's form | Protects a public number | a short answer |
| 7 | ~~The first real model on PROTEUS~~ decided 10-16 by the pre-registered rules: F64 | — | — | — |
| 3, 4 | ~~bpt4 by default; strict CV references~~ bpt4 kept by rule 3 (10-16); the personal-LDA path is not used | — | — | — |

Not decisions, still yours: upstream `origin/main` moved past 442edcd (to a66bc23); we
stay on 442edcd until you say to move it. The older warm-up zips (Riemann vehicle,
MeanLogReg) are superseded by the F64 one for the format test.

## 3. The model in one picture (F64)

Each test session is a stream. Every window's covariance is whitened by a running
reference of that session's windows so far (last 64, pulled towards the training
mean), then the usual features (filter-bank Riemannian tangent space + xDAWN +
log-variance + bpt4 band-power time course) are centred on the session's running
feature mean, and one pooled shrinkage LDA, trained on all ten people centred the
same way per session, classifies. No person ids, no router. Detail:
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) (rewritten for F64 on 10-26, with
real-data figures: session distances, how fast the reference settles, what each
re-centring does to the features).

## 4. What we know (on PROTEUS, sprint 10-16)

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
| 1 | Upload the F64 official zip; compare its three numbers and run time with local (the Dreyer zip took 1.43 s/window there, so the time matters) | proves the sealed format | minutes | your numbers |
| 2 | ~~Update MODEL_ARCHITECTURE.md for F64~~ done 10-26 | — | — | — |
| 3 | Frontal signal: the raw BIDS has **no EOG** either (checked 10-26), so test indirectly: (a) delta + finer bins around feedback onset on lopo (a possible lever), (b) ICA on the resting runs to remove eye components (for the write-up) | (a) unknown; (b) honesty, not score | (a) ~1–2 h; (b) ~2–3 h | — |
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
  - 2026-10-27 (later): the 10-26 warm-up upload was the Oct-1 Dreyer zip by mistake (hash checked): 0.328, 1,081.6 s. F64 not uploaded; decision 9 reopened.
  - 2026-10-27: no change to decisions or numbers. HANDOFF.md rewritten as a complete state of record for a new agent; LOG back-filled for 10-26.
  - 2026-10-26 (later): host outage; warm-up extended to Nov 1; official-trained F64 zip uploaded to the warm-up (result pending), so the "nothing uploaded" note below is superseded. MODEL_ARCHITECTURE.md rewritten for F64 with PROTEUS figures (step 2 done).
  - 2026-10-26: no change to decisions. Raw BIDS has no EOG (41 EEG only), so step 3 is now indirect: the frontal task signal falls with distance from the eyes, is as strong in delta as in theta and peaks in the second after feedback onset (possibly ocular). Practice phase closed Oct 25 with nothing uploaded.
  - 2026-10-16 (evening): sprint complete. Nested check lopo 0.567 (no over-selection); icoh on F64 +1.65 (FI64 variant zip); buffer and K alternatives no gain; EDA; decision 11 new. Nothing running.
  - 2026-10-16 (morning, sprint running): first real PROTEUS model. Candidate F64 (stream covariance + feature re-centring, pooled LDA, bpt4): lopo 0.567, later 0.600; zips ready, not uploaded; decision 7 closed, decisions 9–10 new; nested check and icoh retest running.
  - 2026-10-15 (evening): setup brief run: 0.4.0 entry point, window cache, four protocols, streamed evaluation (two routes = benchopt), Riemann vehicle end to end, self-test 349/349; two PROTEUS zips ready (not uploaded); decision 7's evaluation half done, decision 8 largely done; § 3–4 trimmed.
  - 2026-10-15: RELEASE_DAY.md rewritten for PROTEUS (old runbook archived; rules v2; window straddles feedback onset at +3 s); neuroscience framework for the three tasks (.docx). Setup brief not yet run.
  - 2026-10-13 (evening): setup brief for the PROTEUS training/test configuration written and reviewed; not run yet.
  - 2026-10-13: PROTEUS released and set up (0.4.0 env, loader and baseline smoke pass). New contract: streamed, adaptation allowed, no calibration, 10 unseen people. Decision 1 answered; decisions 7–8 new.
  - 2026-10-02: sprint 10-02 (evening). Thesis window check done (decision 6); warm-up EEGNet ensemble 0.8095, no gain, stopped.
  - 2026-10-02: sprint 10-01. Dual LDA; strict CV references (decision 4); deployment zip ready; shared-covariance LDA no gain.
