# Directions: Track 2 at a glance

**Updated 2026-10-29**: **X4f official reproduced exactly on the warm-up server**
(0.546004 / 0.545484 / 0.545695, 22.6 s). **Next: upload X4f all when the sealed phase opens (Nov 4, 18:01 EST).**
10-28: X4f passed its nested check (nested lopo 0.621 vs
selected 0.628, no over-selection) and is now the recommended sealed model:
**lopo 0.628 / later 0.674** (C2f 0.608 / 0.642 stays the fallback). The 10-28 feature
sprint (ROI power, asymmetry, burstiness) found **no gain**. Warm-up open to
**Nov 1**; sealed phase **Nov 4, 18:01 EST – Nov 21, 17:59 EST** (projected; user, 10-29). Nothing running. This is the short version
for making decisions; every number links to where it comes from (§ 7). Kept current
by every session that changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:** registration closed **Oct 24**; the warm-up (5 uploads/day, does
  not rank) was **extended to Nov 1** after the host outage; the **sealed phase,
  Nov 4, 18:01 EST – Nov 21, 17:59 EST** (projected start, per the user 10-29; it has not opened yet), is the only
  phase that ranks: 1 upload/day, 10 in total, top 3 audited.
- **Data: PROTEUS.** 10 people, 14 sessions, 2,655 windows, 41 EEG channels,
  120 Hz, 3 balanced classes. Sealed test = later sessions of these 10 people
  **plus 10 unseen people**, streamed one window at a time per session, label-free
  adaptation inside a session allowed.
- **How we measure (setup 10-15, [PROTEUS_SETUP.md](PROTEUS_SETUP.md)):** `lopo`
  (leave one person out, 28 cells: the unseen-people half) and `later` (session 1 of
  everyone → sessions 2–3 of P04/P10, 8 cells: the later-sessions half), cell-averaged
  balanced accuracy, paired bootstrap over people. Chance = 0.333.
- **Models (lopo / later; [sprint 10-27](logs/sprint_1027/RESULTS.md),
  [sprint 10-28](logs/sprint_1028/RESULTS.md)):**

  | model | lopo | later | status |
  |---|---|---|---|
  | F64T: per-session covariance + feature re-centring, pooled LDA | 0.567 | 0.600 | old baseline; warm-up exact, 19 s |
  | C2f: + 6-band filter bank with delta + icoh | 0.608 | 0.642 | rule-selected; warm-up exact, 22.7 s |
  | **X4f: C2f + feature scaling + logistic regression** | **0.628** | **0.674** | **nested check passed 10-28** |
  | X4f, nested (every choice re-made inside 9 people) | 0.621 | — | optimism +0.6 points |

  Both re-centrings use only the session's own unlabelled windows so far (allowed).
- **Zips ([SUBMISSIONS.md](SUBMISSIONS.md)):** X4f official `9fb7ffe3` (warm-up: exact match 10-29)
  and X4f all `60a0148c` (sealed) in `submissions/`, thread-pinned, every gate PASS,
  12–14 ms/window (sealed test ≈ 5 min of the 60 allowed). C2f all `7059b103` ready as
  the fallback. **Nothing uploaded to the sealed phase yet.**
- **What the data say ([EDA](reports/proteus_eda/EDA.md)):** every session is its own
  domain (two sessions of one person differ as much as two people), which is why
  re-centring per session carries the model; the class signal is strongest in frontal
  theta (AF4), which may be partly eye movement.

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 13 | **Sealed model = X4f** (nested check passed 10-28: 0.621 nested vs 0.628 selected; inner choice split X4f 5 / C2f 5, so its edge over C2f is small but survives) | (1) ~~upload **X4f official**~~ **done 10-29: exact match, 22.6 s** (`submissions/riemann_sealed_proteus_x4f_official_2026-10-27.zip`, sha256 `9fb7ffe3`) to the warm-up; expected 0.546004 / 0.545484 / 0.545695; (2) if exact, upload **X4f all** (`…_x4f_all_2026-10-27.zip`, `60a0148c`) to the sealed phase | ~2 points over C2f on lopo, 6 over F64T | two uploads |
| 14 | ~~Feature-engineering sprint~~ **run 10-28: no gain.** ROI relative power, asymmetry, \|asymmetry\|, burstiness: all within 1 point of C2f and X4f, CIs cross 0 (burst on X4f −0.9) | — | — | — |
| 9 | ~~Warm-up upload of F64~~ done: exact match; 263 ms/window on the server | — | — | — |
| 12 | ~~Runtime~~ **fixed**: F64T warm-up 10-27 = exact match in **19.0 s** (F64 198.3 s). Was: **Runtime:** F64 at default threads = 263 ms/window (≈ 92 min worst case). F64T (1 thread) = identical predictions, 9.4 ms/window locally | Upload `logs/threads_1027/f64t_official/…_2026-10-27.zip` (sha256 `fcfd7c62`) to the warm-up; if it shows the same three numbers in ≲ 1 min, upload the F64T **all** zip (`a774135a`) as the first sealed one | A timeout wastes a sealed upload | one warm-up upload |
| 10 | **First sealed upload = X4f all** (was C2f all); if C2f all is already up, X4f all is the next one | Accept, after the X4f warm-up check (decision 13) | Best measured model that survived its nested check | yes / no |
| 11 | **Sealed upload plan:** X4f first, C2f as the fallback variant | Accept; keep ≥ 5 of the 10 uploads unused until the last week | The ranked board is the only check of the 10 unseen people | yes / no |
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
| Task-driven blocks (10-28): ROI relative power, asymmetry, \|asymmetry\|, burstiness | −0.9 to +0.2 on C2f / X4f | no gain; asymmetry lifts motor-imagery recall ~1 point only |

The 1–3 s crop loses also in the calibration runs (no feedback): the window's last two
seconds carry task signal, not only feedback.

## 5. Bottlenecks and risks

1. **Server format:** proven for F64T, C2f and X4f (exact matches; X4f
   10-29, 22.6 s).
2. **Small data:** 10 people; `later` rests on 2 people (CIs wide). The nested check
   found no over-selection for F64 (10-16) or X4f (10-28: 0.621 vs 0.628).
3. **Timing under load:** 8 ms/window quiet, up to ~190 ms/window when the CPU is
   oversubscribed (≈ 66 min projected). The server's CPU is unknown; the warm-up
   upload reports the real time.
4. **Rules: settled** (FAQ 10-13): within-session, label-free adaptation is allowed.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | ~~Nested check of X4f~~ **done 10-28: holds** | — | — | — |
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
  - 2026-10-29: sealed phase has not started; projected Nov 4, 18:01 EST – Nov 21, 17:59 EST (user). X4f all waits for it.
  - 2026-10-29: X4f official warm-up = exact match (0.546004 / 0.545484 / 0.545695, 22.6 s); next = X4f all to sealed.
  - 2026-10-28 21:15: feature sprint run: roi / asym / asymabs / burst no gain; X4f nested check passed (0.621 vs 0.628); decision 13 = X4f, decision 14 closed, decisions 10–11 updated, § 1 rewritten.
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
