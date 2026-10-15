# Directions: Track 2 at a glance

**Updated 2026-10-15 (evening)**: the official data (PROTEUS) is out, the contract
changed, and the setup brief has now been **run**: the 0.4.0 environment, a window
cache, four training/test protocols, a streamed local evaluation and one solver
end to end are verified ([PROTEUS_SETUP.md](PROTEUS_SETUP.md)). § 1, 2, 5, 6 are
current; **§ 3–4 are pre-release proxy evidence and must be re-validated on
PROTEUS**. This is the short version for making decisions. Every number links to
where it comes from. Read the long documents only if you want the detail (§ 7).
Kept current by every session that changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:** registration closes **Oct 24**; the warm-up (5 uploads/day, does
  not rank) ends **Oct 25**; the **sealed phase, Oct 28 – Nov 21**, is the only
  phase that ranks: 1 upload/day, 10 in total, top 3 audited.
- **Data: PROTEUS (released 2026-10-13).** 10 people, 14 sessions, 2,655 windows
  (~190 per session; 8 people have 1 session, P04 and P10 have 3), 41 EEG
  channels, 120 Hz, microvolts, 3 balanced classes, Graz and BrainHero runs in
  every session. In WSL: `~/neuralbench/benchopt_data_040`, environment
  `source ~/codabench/env040.sh` (neuralbench 0.4.0). The frozen 0.3.1 setup is
  untouched and still passes its regression.
- **The contract** (upstream FAQ, 0.4.0): streamed, one window at a time per
  person-session on a fresh copy of the model; adapting inside a session without
  labels is allowed; sealed test = later sessions of these 10 people **plus 10
  unseen people**, no calibration data; the warm-up scores on PROTEUS (test =
  P04 × 3 + P09, labels public, so only a format check).
- **Setup session 2026-10-15 ([brief](prompts/2026-10-13_proteus_train_test_setup.md), started 17:38): every gate passed.**
  - One self-test (`analysis/proteus_selftest.py`, 349 checks, 55 s) proves the
    environment, the cache (= the official loader bit for bit; labels = the raw
    BIDS events), the four protocols (no leaks, right order and cells) and the
    streamed evaluation (two routes agree with benchopt to 6 decimals; test
    doubles behave).
  - The Riemann-Sealed solver went through fit → export → read-only replay → zip
    → unzipped replay on PROTEUS with identical scores everywhere, in its simplest
    valid form for unseen people (pooled LDA, no router, no personal LDA).
  - **Plumbing numbers, for information only (untuned; differences below 0.05
    mean nothing on 8 cells):** `official` (8 cells): Riemann 0.407, MeanLogReg
    0.396, chance 0.333; `lopo` (28 cells): Riemann 0.419, MeanLogReg 0.443;
    `later` (2 people): Riemann 0.492, MeanLogReg 0.429
    ([RESULTS.md](logs/proteus_setup/RESULTS.md)). Inference 8–10 ms per window:
    the whole sealed test would take about 3 minutes of the 60 allowed.
- **Uploaded so far** (warm-up, [SUBMISSIONS.md](SUBMISSIONS.md)): EEGNet 0.82 and
  the Riemann deployment test 0.62 (2026-10-02), both on the old Dreyer warm-up.
  **Two PROTEUS zips are ready and not uploaded**: the Riemann vehicle (expected
  server numbers 0.406751 / 0.405717 / 0.405298) and MeanLogReg (0.395863 /
  0.395672 / 0.396026). The scoring image changed (`v2` → `v3`) since our last
  server test.
- **Sealed-phase candidate:** none yet on PROTEUS. The old candidate
  (Riemann-Sealed + bpt4, blend of pooled and personal LDAs) routes unseen people
  to a training person silently; its evaluation half is replaced by the protocols
  above, its model half is decision 7.
- **Stand-in evidence (pre-release, Scherer 3-class, chance 0.333):** 0.529
  balanced accuracy, 0.592 with test-time re-centring
  ([SEALED_RECIPE § 2](SEALED_RECIPE.md)). To be re-validated on PROTEUS.

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 7 | **The first real model on PROTEUS.** The evaluation half is done: protocols `official`, `lopo` (10 folds, 28 cells), `later`, `all`; self-test; candidate script | Run one comparison next, on `lopo` (the unseen-people half) and `later` (the later-sessions half), scored by the cell mean: (a) the pooled vehicle as is; (b) + bpt4; (c) + streamed within-session re-centring (`adapt="online"`, allowed); (d) a calibration-free per-cell variant. Decide from the two protocols together, never from one. The plumbing numbers in § 1 are not evidence for or against any arm | The sealed test is half unseen people; only P04/P10 measure session change | a conversation, then ~half a day of runs |
| 8 | **Adopt the 0.4.0 setup as the main one** (largely done: `env040.sh`, `PROTEUS_SETUP.md`, the protocols and the self-test exist side by side with the old setup; the stale docs carry a banner) | Accept: keep the old scripts only for the regression gate (`train_sealed.sh` on zhou2016, still passing); retire the old sections of the long docs at their next rewrite | One environment, one set of docs | yes / no |
| 3 | **Keep bpt4 by default** unless `lopo` shows ≥ 1 point better without it | Accept; it is one arm of the decision-7 comparison | the one solid gain of 09-29 (+4.4) on stand-ins; its 1-s bins split the window at feedback onset (+3 s). Rule 3 in [RELEASE_DAY § 5](RELEASE_DAY.md) | read one rule |
| 4 | **Strict CV references stay** (`wcv_ref="strict"`), but the LOSO fold rule must change: only P04/P10 have ≥ 2 sessions | Accept strict; choose any blend weight by leave-one-run-out on session 1 | Rule 4 in [RELEASE_DAY § 5](RELEASE_DAY.md) | read one rule |
| 5 | Optional: approve REVE/LaBraM weights; merge the thesis docs branch `docs/sprint0928-consistency` | Low priority | REVE/LaBraM are hours per epoch on this CPU | — |
| 6 | **Thesis: what is the One/Two prompt, and what is the `96` trigger?** The window check found no CV leakage, but the full-vs-late gap lives in the first 0.5 s after t = 0 ([results](../docs/sprints/2026-10-02_window_effect_check_RESULTS.md), branch `feat/window-effect-check`, not merged) | Tell me the prompt's form (spoken word, symbol, image). If it differs by condition, reword the README Results | Protects a public number | a short answer |
| 1 | ~~Ask the organisers about test-time statistics~~ answered 2026-10-13 by the FAQ: allowed, causal, label-free, within one session | — | online re-centring can go in | — |
| 2 | ~~Deployment-test upload~~ done 2026-10-02 (0.62, 59 s) | — | the joblib format ran on the server (image `v2`; it is `v3` now) | — |

Two things for you that are not decisions: **upstream `origin/main` moved past
442edcd (to a66bc23)**: we stay on 442edcd until you say to move the worktree
(the loader and the scoring could change under us). **Upload** the Riemann zip
(and, if you like, the MeanLogReg zip) to the warm-up before Oct 25 to prove the
streamed format on the `v3` image; the server's three numbers must match § 1.

## 3. The model in one picture (Riemann-Sealed + bpt4, pre-release)

Filter-bank Riemannian tangent-space features (+ xDAWN, log-variance, bpt4 band-power
time course) → shrinkage LDA on everyone, blended with a per-person LDA chosen by a
log-power router. Pre-release design: the router and per-person parts have no path for
unseen people. Full explanation and diagram: [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

## 4. What we know (stand-ins, pre-release; re-validate on PROTEUS)

| Lever | Effect on the stand-ins | Status |
|---|---|---|
| **Online re-centring** (unlabelled test windows, within session) | +3.5 to +7.5; +3.0 to +4.4 if windows arrive shuffled | **allowed by the rules (10-13); built; untested on PROTEUS** |
| **bpt4**: when in the trial each band's power changes | +4.35 (CI +2.05, +6.67), positive on 4 of 4 stand-ins | in the model; decision 3 |
| **icoh**: lagged connectivity between channels | +2.2 alone; +6.5 with bpt4 (post-hoc) | release-day test; fits in time |
| Personal + pooled LDA blend; router + per-person re-centring | +1 to +12; up to +5 pooled | in the model; **no path for unseen people** |
| No gain (dropped) | EEGNet, pre-training, re-referencing, CSP, regional covariances, time-segment covariances, 8–30 Hz band-pass, shared-covariance personal LDA | — |

Full lever table, evidence and the activation EDA: [SEALED_RECIPE § 2](SEALED_RECIPE.md).

## 5. Bottlenecks and risks

0. **The old sealed model does not fit the new contract as built**: it needs a
   path for unseen people. Its streaming side is now proven (`reset_state()`,
   deep-copyable, B = 1 identical to a batch).
1. **Very little data.** 2,655 windows from 10 people; only P04 and P10 show
   session-to-session change; local scores are noisy (8 cells; the cell mean of
   a label-blind model has sd ≈ 0.025). The organisers' deep baselines sit at chance.
2. **Rules: settled.** Within-session adaptation allowed; nothing crosses a session.
3. **Deployment: the streamed format is untested on the server** (`v3` image, a
   pyriemann joblib under the new contract). The ready zips exist for that.
4. **Compute: not a concern.** Fits take 27–49 s; inference 8–10 ms per window.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | **Decision 7's comparison** on `lopo` + `later` through `scripts/proteus_candidate.sh` and `datasets/proteus_protocols.py` (vehicle, + bpt4, + online re-centring, calibration-free per-cell variant) | the first real number and the model choice | ~half a day | decision 7 |
| 2 | Upload the Riemann zip to the PROTEUS warm-up before **Oct 25** | proves the streamed format and the `v3` image | one upload | you |
| 3 | EDA on the real data (per-person ERD maps, laterality, calibration vs online runs, P04/P10 drift) feeding the model design | guides step 1 | ~half a day | nothing |
| 4 | A summariser for the pre-registered rules v2 of [RELEASE_DAY § 5](RELEASE_DAY.md) on the protocol outputs | release-day discipline | ~2 h | step 1 |
| 5 | Thesis: the nested XGB pipeline on the 0.5–2 s window for a headline-comparable number | — | ~3–4 h | decision 6 |

Not worth it now: REVE/LaBraM on this CPU, more training participants, more
spatial-filter variants, a restricted icoh, a shared-covariance personal LDA.

## 7. Where to read more

| Question | Document |
|---|---|
| What exists on PROTEUS, how to run it, which gates passed? | [PROTEUS_SETUP.md](PROTEUS_SETUP.md); values in [logs/proteus_setup/RESULTS.md](logs/proteus_setup/RESULTS.md) |
| What exactly was the pre-release model, and what is the evidence? | [SEALED_RECIPE.md](SEALED_RECIPE.md) § 1–2 |
| What is the released data, what carries over, what are the rules now? | [RELEASE_DAY.md](RELEASE_DAY.md) (rewritten 10-15) |
| Why these three tasks, and what does the brain do in each? | [reports/PROTEUS_task_neuroscience_2026-10-15.docx](reports/PROTEUS_task_neuroscience_2026-10-15.docx) |
| What happened in the last session, and what is running? | [HANDOFF.md](HANDOFF.md), top section |
| How do the models work, for EEG readers? | [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) |
| Every run, timing and decision | [LOG.md](LOG.md) |
| What was uploaded, what is ready? | [SUBMISSIONS.md](SUBMISSIONS.md) |
| Dates and rules | [COMPETITION.md](COMPETITION.md) (pre-release text; banner) |

## 8. Keeping this file current

- **When:** at the end of every sprint or session that changes a decision, a
  number, the plan or the competition calendar.
- **Size:** under ~150 lines. Detail goes to the documents in § 7, not here.
- **Numbers:** each one must match its source document.
- **Change log** (newest first, one line each):
  - 2026-10-15 (evening): setup brief run: 0.4.0 entry point, window cache, four protocols, streamed evaluation (two routes = benchopt), Riemann vehicle end to end, self-test 349/349; two PROTEUS zips ready (not uploaded); decision 7's evaluation half done, decision 8 largely done; § 3–4 trimmed.
  - 2026-10-15: RELEASE_DAY.md rewritten for PROTEUS (old runbook archived; rules v2; window straddles feedback onset at +3 s); neuroscience framework for the three tasks (.docx). Setup brief not yet run.
  - 2026-10-13 (evening): setup brief for the PROTEUS training/test configuration written and reviewed; not run yet.
  - 2026-10-13: PROTEUS released and set up (0.4.0 env, loader and baseline smoke pass). New contract: streamed, adaptation allowed, no calibration, 10 unseen people. Decision 1 answered; decisions 7–8 new.
  - 2026-10-02: sprint 10-02 (evening). Thesis window check done (decision 6); warm-up EEGNet ensemble 0.8095, no gain, stopped.
  - 2026-10-02: deployment test passed (0.62, 59 s); decision 2 done. Organiser email bounced: decision 1 became one Discord question.
  - 2026-10-02: sprint 10-01. Dual LDA; strict CV references (decision 4); deployment zip ready; shared-covariance LDA no gain.
