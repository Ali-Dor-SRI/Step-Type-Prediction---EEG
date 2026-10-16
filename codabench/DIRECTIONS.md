# Directions: Track 2 at a glance

**Updated 2026-10-16 (morning), sprint running until Mon 10-19 08:00**: the first
real PROTEUS model exists. A sealed-phase candidate (F64) is built, every gate
passed, two zips are ready and **not uploaded**. This is the short version for
making decisions; every number links to where it comes from (§ 7). Kept current
by every session that changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:** registration closes **Oct 24**; the warm-up (5 uploads/day, does
  not rank) ends **Oct 25**; the **sealed phase, Oct 28 – Nov 21**, is the only
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
  cropping to 1–3 s, Euclidean means, the EEGNet ensemble. icoh +1.1 (below the bar).
- **Ready, not uploaded ([SUBMISSIONS.md](SUBMISSIONS.md)):** the F64 sealed zip
  (trained on all 14 sessions) and an official-trained F64 zip for the warm-up
  (server must show 0.507944 / 0.506931 / 0.507285). Sealed test runtime ≈ 3 min of
  the 60 allowed (8.2 ms/window on one quiet thread).
- **Still running this sprint:** a nested check of the selection (how optimistic
  is choosing on `lopo`?) and one pre-registered icoh test on F64. Results by Monday.

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 9 | **Upload the official-trained F64 zip to the warm-up** (before Oct 25) | Yes, now: it is the only way to prove the adaptive, streamed model runs on the `v3` scoring image before the sealed phase | The sealed upload budget is 10; a format failure there costs a day | one upload; compare the three numbers |
| 10 | **First sealed upload (Oct 28) = the F64 all-trained zip** | Accept, if the warm-up numbers match | It is the best measured model; later uploads can test variants | yes / no |
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
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) (pre-F64; to update).

## 4. What we know (on PROTEUS, sprint 10-16)

| Lever | Effect (lopo / later, points) | Status |
|---|---|---|
| Feature re-centring per stream | +8.3 / +7.0 over S64 | **in the candidate** |
| Covariance re-centring per stream (buffer 64) | +5.7 / +0.2 over bpt4 base | **in the candidate** |
| bpt4 band-power time course | +0.8 / +3.6 | **in the candidate** |
| icoh | +1.1 / +1.0 (below the bar) | retest running on F64 |
| No gain | router + personal LDA, no xDAWN, CAR, clipping, 1–3 s crop, Euclidean means, EEGNet ensemble | dropped |

The 1–3 s crop loses also in the calibration runs (no feedback): the window's last two
seconds carry task signal, not only feedback.

## 5. Bottlenecks and risks

1. **Server format of an adaptive model** is untested on image `v3`: decision 9.
2. **Small data:** 10 people; `later` rests on 2 people (CIs wide). The nested check
   (running) will say how optimistic the choice on `lopo` is.
3. **Timing under load:** 8 ms/window quiet, up to ~190 ms/window when the CPU is
   oversubscribed (≈ 66 min projected). The server's CPU is unknown; the warm-up
   upload reports the real time.
4. **Rules: settled** (FAQ 10-13): within-session, label-free adaptation is allowed.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | Warm-up upload of the F64 official zip; compare numbers and run time | proves the sealed format | one upload | you |
| 2 | Finish this sprint's nested check and icoh test (running) | an honest estimate; maybe +1 | none | — |
| 3 | EDA on the real data (per-person ERD maps, P04/P10 drift) | guides the next lever | ~half a day | — |
| 4 | Re-centring variants with a pre-registered rule (stream_k, buffer on F64) | 0–2 points, risk of over-selection | ~half a day | the nested result |
| 5 | Thesis: XGB on the 0.5–2 s window | — | ~3–4 h | decision 6 |

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
  - 2026-10-16 (morning, sprint running): first real PROTEUS model. Candidate F64 (stream covariance + feature re-centring, pooled LDA, bpt4): lopo 0.567, later 0.600; zips ready, not uploaded; decision 7 closed, decisions 9–10 new; nested check and icoh retest running.
  - 2026-10-15 (evening): setup brief run: 0.4.0 entry point, window cache, four protocols, streamed evaluation (two routes = benchopt), Riemann vehicle end to end, self-test 349/349; two PROTEUS zips ready (not uploaded); decision 7's evaluation half done, decision 8 largely done; § 3–4 trimmed.
  - 2026-10-15: RELEASE_DAY.md rewritten for PROTEUS (old runbook archived; rules v2; window straddles feedback onset at +3 s); neuroscience framework for the three tasks (.docx). Setup brief not yet run.
  - 2026-10-13 (evening): setup brief for the PROTEUS training/test configuration written and reviewed; not run yet.
  - 2026-10-13: PROTEUS released and set up (0.4.0 env, loader and baseline smoke pass). New contract: streamed, adaptation allowed, no calibration, 10 unseen people. Decision 1 answered; decisions 7–8 new.
  - 2026-10-02: sprint 10-02 (evening). Thesis window check done (decision 6); warm-up EEGNet ensemble 0.8095, no gain, stopped.
  - 2026-10-02: sprint 10-01. Dual LDA; strict CV references (decision 4); deployment zip ready; shared-covariance LDA no gain.
