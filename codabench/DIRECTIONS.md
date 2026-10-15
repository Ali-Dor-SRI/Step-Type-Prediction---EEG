# Directions: Track 2 at a glance

**Updated 2026-10-15**: the official data (PROTEUS) is out and the contract
changed. [RELEASE_DAY.md](RELEASE_DAY.md) was rewritten for it on 10-15 (data
structure, what carries over, what needs tuning, rules v2); the old runbook is
archived. § 1, 2, 5, 6 are current; **§ 3–4 are pre-release proxy evidence and
must be re-validated on PROTEUS**. This is the short version
for making decisions. Every number links to where it comes from. Read the long
documents only if you want the detail (§ 7). Kept current by every session that
changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:**
  - registration closes **Oct 24**;
  - the warm-up (5 uploads/day, does not rank) ends **Oct 25**;
  - the **sealed phase, Oct 28 – Nov 21**, is the only phase that ranks:
    1 upload/day, top 3 audited.
- **Data: released and set up (2026-10-13).** PROTEUS, NEMAR nm000290, 1.5 GB,
  CC-BY. It is much smaller than planned for: **10 people, 14 sessions, 2,655
  windows** (~190 per session; 8 people have 1 session, P04 and P10 have 3),
  41 EEG channels, 3 balanced classes, Graz and BrainHero runs in every session.
  - Lives in WSL at `~/neuralbench/benchopt_data_040`, with a separate
    0.4.0 environment (`~/neuralbench/.venv040`) and upstream checkout
    (`~/neuralbench/competition-040`). The frozen 0.3.1 setup is untouched.
  - Checked: official prepare (2.5 min), loader, and the organisers' tiny
    baseline end to end (MeanLogReg 0.396 on 8 cells; chance 0.333). No NaNs,
    no flat channels. **No model of ours has run on it yet.**
- **The contract changed with the release** (upstream FAQ, neuralbench 0.4.0):
  - **Sealed test = later sessions of these 10 people + 10 unseen people.**
    No calibration data for anyone: the old "sessions 1–3 labelled" plan is gone.
  - **Streamed:** `predict` gets one window at a time, in recording order,
    per person-session; the model is copied fresh for each session.
  - **Adapting inside a session without labels is allowed** (old decision 1).
  - **Input:** 41 EEG channels, 120 Hz, 4 s starting 1 s after the cue, in
    microvolts, 0.1–75 Hz, **no scaling or clamping**. No EMG/EOG, ever.
  - **Warm-up now scores on PROTEUS** (test = P04's 3 sessions + P09). Its
    labels are public, so it is only a format check. Old uploads no longer fit.
  - **Sealed phase: 1 upload/day and 10 in total.**
- Every number in § 3–4 comes from stand-in datasets, before the release.
- **Warm-up (clean Dreyer split):** a 5-seed EEGNet ensemble scored **0.8095** vs the single model's 0.820: no gain, so no more tries. The 0.9+ leaderboard entries pretrain on all of Dreyer (test subjects included) or use per-batch statistics: not built.
- **Uploaded so far** (warm-up, which doesn't rank;
  [SUBMISSIONS.md](SUBMISSIONS.md)):
  - EEGNet, 0.82;
  - the **sealed model's deployment test** on 2026-10-02: Finished, 0.62 in
    59 s. The Riemann-Sealed model runs on the scoring server.
- **Sealed-phase candidate:** **Riemann-Sealed + bpt4**. It retrains end to
  end on release day with the runbook: **~4–4.5 h** now, down from 6–7 h,
  rehearsed at full size on 10-01 ([archived runbook § 8](archive/RELEASE_DAY_2026-10-01_pre-release.md)).
  **Superseded on 10-15:** the retrain path is now [RELEASE_DAY.md § 5](RELEASE_DAY.md)
  (fits take minutes at 2,655 windows; the work is in the evaluation and the
  path for unseen people, not in compute).
- **Headline on the stand-in closest to the sealed tasks** (Scherer: word
  association / subtraction / hand imagery, 3 classes, chance 0.333):
  - **0.529** balanced accuracy, up from 0.486 before this week;
  - **0.592** if test-time re-centring is allowed (see decision 1).
  - Source: [SEALED_RECIPE § 2](SEALED_RECIPE.md).

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 1 | ~~Ask the organisers about test-time statistics~~ **answered 2026-10-13 by the FAQ: allowed**, causal, label-free, within one session | — | Online re-centring can go in; it must work one window at a time | — |
| 7 | **Architecture of the first PROTEUS run** (to discuss) | Local evaluation first: leave-one-person-out for the "unseen people" half, and train on session 1 → test on sessions 2–3 of P04/P10 for the "later sessions" half; then the Riemann recipe with streaming re-centring, without the calibration blend | The sealed test is half new people, and only 2 people show any session-to-session change | a conversation |
| 8 | **Adopt the 0.4.0 setup as the main one** (the 20 × 6 runbook was retired on 10-15: archived, RELEASE_DAY rewritten with rules v2) | Accept: the old scripts stay for the regression gate only | Keeps one environment and one set of docs | yes / no |
| 2 | ~~Deployment-test upload~~ **done 2026-10-02**: Finished, 0.62, 59 s | — | The sealed model's format runs on the scoring server | — |
| 3 | **Keep bpt4 by default** unless `lopo` shows ≥ 1 point better without it | Accept | It is the one solid gain of 09-29 (+4.4), and its 1-s bins split the window at feedback onset (+3 s). Rule 3 in [RELEASE_DAY § 5](RELEASE_DAY.md) | read one rule |
| 4 | **Review: strict CV references stay** (`wcv_ref="strict"`), but the LOSO fold rule must change: only P04/P10 have ≥ 2 sessions | Accept strict; choose w by leave-one-run-out on session 1 | Rule 4 in [RELEASE_DAY § 5](RELEASE_DAY.md) | read one rule |
| 5 | Optional: approve REVE/LaBraM weights; merge the thesis docs branch `docs/sprint0928-consistency` (`main`'s perf-loop SUMMARY and MODELS still show old numbers) | Low priority | REVE/LaBraM are hours per epoch on this CPU | — |
| 6 | **Thesis: what is the One/Two prompt, and what is the `96` trigger?** The window check ([results](../docs/sprints/2026-10-02_window_effect_check_RESULTS.md), branch `feat/window-effect-check`, not merged) found no CV leakage, but the full-vs-late gap lives entirely in the first 0.5 s after t = 0 (cross-person AUC 0.675 there, chance after 0.5 s). t = 0 is the `96` event ~0.55 s after the prompt | Tell me the prompt's form (spoken word, symbol, image). If it differs by condition, reword the README Results (the 0.714 mostly reads the instruction response) | Protects a public number | a short answer |

## 3. The model in one picture (Riemann-Sealed + bpt4)

```
4-s EEG window, 43 channels (EMG/EOG dropped unless the release-day rule admits them)
 │
 ├─ 1 Router       log power 1–45 Hz → "which participant is this?"
 │                 (87–100 % correct on stand-ins; when unsure → shared path)
 ├─ 2 Re-centre    whiten with that participant's mean covariance
 │                 (removes person- and session-specific offsets)
 ├─ 3 Features     ~5,800 numbers per window (at 43 ch):
 │                   filter-bank covariances 4–8/8–13/13–30/30–45 Hz → tangent space  3,784   most of the signal
 │                   broadband covariance → tangent space                              946
 │                   xDAWN covariances (class templates) → tangent space               300
 │                   log-variance per channel                                           43
 │                   bpt4: band power per channel in 4 one-second bins                 688   new, +4.4
 └─ 4 Classifier   shrinkage LDA on everyone (pooled), blended with the participant's own LDA;
                   blend weight chosen on their calibration sessions
                   (since 10-01: per-person LDAs solved in a 50× faster, equivalent form)
```

- **What is not in the sealed model:** the neural network (EEGNet), which
  scored 6–12 points below this pipeline on every stand-in.
- **Full explanation:** [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md). This
  draft from another session (2026-09-29, not yet committed) predates bpt4.

## 4. What we know

| Lever | Effect on the stand-ins | Status |
|---|---|---|
| Personal + pooled LDA blend | +1 to +12 points vs pooled only | in the model |
| Router + per-person re-centring (no test data) | up to +5 pooled, ~0 per-person | in the model |
| **Online re-centring** (uses unlabelled test windows) | +3.5 to +7.5; +3.0 to +4.4 if test windows arrive shuffled | **built, waiting on decision 1** |
| Filter bank (4 bands) | carries most of the 3-class signal | in the model |
| **bpt4**: when in the trial each band's power changes | +4.35 (CI +2.05, +6.67), positive on 4 of 4 stand-ins | **in the model since 09-29** |
| **icoh**: lagged connectivity between channels | +2.2 alone; +6.5 with bpt4 (post-hoc) | release-day test; **fits at 500 Hz since 10-01** (19.3 min, 1.23×) |
| xDAWN (class templates) | +4–5 on motor imagery, ~0 on the sealed-like tasks | kept; first to drop if the real data disagrees |
| Strict CV references for the blend weight | correctness, not accuracy: on Zhou w moved 0.5 → 0.75; the others unchanged | **release-day setting since 10-01** |
| No gain | EEGNet; pre-training on other datasets; more participants; re-referencing / Laplacian; CSP; regional covariances; band-power maps; time-segment covariances (under the deployed model); 8–30 Hz band-pass; **personal LDA with a shared (pooled) covariance** (10-01: −2.7 / −0.9 on Scherer 3-class) | dropped |

- **Where the brain-activation differences are** (training sessions only,
  [activation EDA](reports/features_0929/activation_eda.md)):
  - alpha, posterior-right, after ~1.5 s;
  - a word-association beta decrease at F3;
  - early frontal slow activity that looks like eye movement (kept out of
    the model: its bands start at 4 Hz).
- Differences are strong *within* each person but differ between people, which
  is why personalisation matters.

## 5. Bottlenecks and risks

0. **(2026-10-13) The sealed model does not fit the new contract as built.**
   It needs: batch-of-1 streaming with per-session state (`reset_state`,
   deep-copyable), microvolt input, 41 channels, and a path for people it has
   never seen (the router and the personal LDAs cover at most half the test).
   Items 3–6 below describe the old plan.
1. **Very little data.** 2,655 windows from 10 people, and only P04 and P10
   show session-to-session change. Local scores will be noisy (the warm-up
   test is 8 cells); the organisers' deep baselines sit at chance
   (EEGNet 0.326, REVE 0.360 across people).
2. **Rules: settled.** Within-session adaptation is allowed; nothing may
   cross a session boundary.
3. **Deployment: tested 2026-10-02**: the sealed model runs on the scoring
   server. Untested there: per-context references and the 47-ch input (Dreyer
   has neither). Upload the release-day candidate to the warm-up once it
   switches to Graz + BrainHero, if that is before Oct 25.
4. **Compute (CPU only): largely solved on 10-01.** The per-person LDAs now
   cost ~0.15 s instead of ~9 s each. At 500 Hz the recipe fits in 15.7 min,
   bpt4 in 17.7 and bpt4 + icoh in 19.3 ([archived runbook § 8](archive/RELEASE_DAY_2026-10-01_pre-release.md)).
   What remains is the feature computation itself.
5. **Few windows per person.** About 360–720 calibration windows (mock
   structure; real counts unknown) against ~6,000 features. Heavy shrinkage
   keeps it stable, but new blocks must be cheap and strong.
6. **Release-day time.** The runbook now takes ~4–4.5 h at 120 Hz (rehearsed
   at full size: ablations 69 min, replica check 28 min). The release date is
   still unknown, with the sealed phase on Oct 28.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | Run the setup brief [prompts/2026-10-13_proteus_train_test_setup.md](prompts/2026-10-13_proteus_train_test_setup.md) in a fresh session: four training/test protocols, streamed evaluation, one solver end to end, no tuning | a verified local yardstick and two warm-up zips | ~7 h, unattended | paste the brief |
| 2 | Choose and score the first real model on those protocols (the solver's inference already runs one window at a time; what is missing is the path for unseen people) | first real number; fits take minutes at this size | ~half a day | step 1, decision 7 |
| 2b | Upload it to the PROTEUS warm-up before **Oct 25** | proves the streamed format on the server | one upload | step 2 |
| 3 | Riemann + EEGNet probability ensemble | +1 to +2 (EEGNet is now 6–12 points behind, so likely less) | half a day | nothing |
| 4 | Thesis: **done 2026-10-02**, see decision 6. Next: the nested XGB pipeline on the 0.5-2 s window for a headline-comparable number | Wait for decision 6 | ~3-4 h | decision 6 |

Done on 10-01 (no longer open): faster per-person LDA; icoh's 500 Hz cost;
strict per-fold blend-weight search. Not worth it now: REVE/LaBraM on this CPU,
more training participants, more spatial-filter variants, a restricted icoh,
a shared-covariance personal LDA.

## 7. Where to read more

| Question | Document |
|---|---|
| What exactly is the model, and what is the evidence? | [SEALED_RECIPE.md](SEALED_RECIPE.md) § 1–2 (Phase 7 = this week's features) |
| What is the released data, what carries over, what needs tuning, what are the rules now? | [RELEASE_DAY.md](RELEASE_DAY.md) (rewritten 10-15) |
| Why these three tasks, and what does the brain do in each (feeds the model design)? | [reports/PROTEUS_task_neuroscience_2026-10-15.docx](reports/PROTEUS_task_neuroscience_2026-10-15.docx) |
| What happened in the last session, and what is running? | [HANDOFF.md](HANDOFF.md), top section |
| How do the models work, for EEG readers? | [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) |
| Every run, timing and decision | [LOG.md](LOG.md) |
| What was uploaded? | [SUBMISSIONS.md](SUBMISSIONS.md) |
| Dates and rules | [COMPETITION.md](COMPETITION.md) |

## 8. Keeping this file current

- **When:** update it at the end of every sprint or session that changes a
  decision, a number, the plan or the competition calendar.
- **Size:** keep it under ~150 lines. Detail goes to the documents in § 7, not
  here.
- **Numbers:** each one must match its source document.
- **Change log** (newest first, one line each):
  - 2026-10-15: RELEASE_DAY.md rewritten for PROTEUS (old runbook archived; rules v2;
    window straddles feedback onset at +3 s; calibration vs online runs; unseen-people path);
    neuroscience framework for the three tasks written (.docx). Setup brief still not run.
  - 2026-10-13 (evening): setup brief for the PROTEUS training/test
    configuration written and reviewed (§ 6 step 1); not run yet.
  - 2026-10-13: PROTEUS released and set up (0.4.0 env, loader and baseline
    smoke pass). New contract: streamed, adaptation allowed, no calibration,
    10 unseen people in the sealed test. Decision 1 answered; decisions 7–8 new.
  - 2026-10-02: sprint 10-02 (evening). Thesis window check done (decision 6); warm-up EEGNet ensemble 0.8095, no gain, stopped.
  - 2026-10-02: deployment test passed (0.62, 59 s); decision 2 done. Organiser
    email bounced: decision 1 is now one Discord question (order/persistence from code).
  - 2026-10-02: sprint 10-01. Dual LDA (icoh fits at 500 Hz: old decision 4
    resolved); strict CV references adopted (new decision 4: review);
    deployment zip ready; release day ~4–4.5 h; shared-covariance LDA no gain.
  - 2026-10-01: created. bpt4 is in the model; icoh is a release-day test; 4
    decisions pending.
