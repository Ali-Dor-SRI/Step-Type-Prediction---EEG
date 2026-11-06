# Sprint 2026-11-06 (10:50 → 15:00): the two cheapest big levers — results

Brief: `prompts/2026-11-06_next_session_plan.md` § 2 (Sprint 1). Base everywhere:
**X4f** (lopo 0.6275 / later 0.6735; rows `logs/proteus_w1016/K2.jsonl`), **C2f**
(0.6083 / 0.6424, `J2.jsonl`). Cell-averaged balanced accuracy; Δ in points ×100;
CI = paired bootstrap over people (10,000). Rows: `logs/sprint_1106/rows/*.jsonl`
(git-ignored). **Headline: every lever screened today is no gain.**

## P1 — augmented training windows from the official loader: exact (10:52–10:53)

`analysis/proteus_cache_aug.py`: the organisers' `load_task` with the one data
override `{"start": o}`, its exca cache pointed at a **copy** (`~/neuralbench/proteus_cache_aug/nbcache`);
the frozen caches were not touched. Offsets 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5,
3.0, 3.5, 4.0 s after the cue (`[cue + o, cue + o + 4)` inside the 8-s trial), 2–13 s each.

| gate | result |
|---|---|
| A0: offset 1.0 = official cache bit for bit | **PASS** (max \|diff\| 0.0, all 2,655 sha1 equal, keys + labels equal) |
| A1: every offset 2,655 rows, keys + labels = official row by row, start = cue + o, finite | **PASS** (all 10 other offsets; no window identical to an official one) |
| harness parity: `proteus_sweep.py --train-offsets none` (the new loader, official rows only) | **PASS**: X4f lopo fold 0 0.695397, later 0.673546 = K2 exactly |

Estimate 60–90 min, actual ≈ 25 min including the parity check (the loader takes an
override; no MNE route needed).

## P2 — X4f / C2f trained with extra windows (lopo 0–9 + later)

`proteus_sweep.py --train-offsets` (harness side: the solver file is untouched, so its
defaults are bit-identical by construction): the training rows' windows at the extra
offsets join the training set (training people / sessions only), sorted in recording
order, each session keeping its stream id. Test side unchanged (official windows,
streamed). `:sep` = each offset's copy of a session is its own session for the
per-session centring (as a test stream is centred on its own windows).

| arm | rows ×  | lopo folds done | Δ lopo per fold vs base | lopo | Δ lopo [95 % CI] (up/10) | later | Δ later |
|---|---|---|---|---|---|---|---|
| X4f + aug4 {0.5, 1.5, 2.0, 2.5} | 5 | 10 | −5.9 −3.1 +1.6 −0.8 −1.1 +2.2 +2.2 −5.8 −8.5 +1.4 | 0.6157 | **−1.18 [−3.56, +0.50] (4/10)** | 0.6632 | −1.04 [−3.18, +1.11] |
| C2f + aug4 | 5 | 1 (stopped) | **−13.7** (0.5208 vs 0.6579) | — | — | — | — |
| X4f + near2s {0.75, 1.25}, `:sep` (added, see deviations) | 3 | 10 | −3.1 +0.6 +1.6 −1.6 +1.0 +1.1 −1.5 −1.0 −2.7 +0.8 | 0.6228 | **−0.47 [−1.44, +0.48] (5/10)** | 0.6629 | −1.07 [−1.58, −0.55] |

Calibration / online runs: X4f 0.654 / 0.613; aug4 0.621 / 0.613; near2s 0.628 / 0.620 —
the extra windows cost the calibration runs and buy nothing on the online runs.
Table: `logs/sprint_1106/p2_summary.md`.

Fit time: 5× rows ≈ 580–870 s per fold under 3–4 concurrent jobs (X4f at 1× alone:
74 s) → **≈ 10–14 min per fold, ≈ 2–2.5 h per arm on one lane**; 3× rows ≈ 6–9 min/fold.

**Why it hurts (mechanism).** X4f's xDAWN templates and bpt4 band-power time course are
locked to the official window (cue + 1 s); shifted windows blur them, and the 10-16
crop ablation (1–3 s, −3.8) already showed the window's timing carries the signal. The
LDA (C2f) suffers most (−13.7 on fold 0). Small shifts with per-offset centring
(near2s) are about neutral. The shift-tolerant net loses too (P3): the information is
in *this* window, not in the trial.

**Decision (rules v2).** **No gain** for both arms (Δ lopo < +1.0, CIs cross 0, later
negative). Training-window augmentation at other cue offsets is dropped for the Riemann
pipeline. By the 15:00 table (plan § 4) the weekend branch is **B (explore)**.

## P3 — deep-net smoke (lopo fold 0 + later, 60 epochs, seed 33, per-window z-score)

`analysis/sprint1106_netsmoke.py` (braindecode 1.8.1, AdamW 1e-3, cosine, label
smoothing 0.1, batch 64, 4 threads; nets are stateless, so batch = streamed).

| model | training windows | fold 0 (X4f 0.6954) | later (X4f 0.6735) | s / epoch |
|---|---|---|---|---|
| ShallowFBCSPNet | official | 0.5853 (−11.0) | 0.5618 (−11.2) | 2.6–4.4 |
| EEGNetv4 | official | 0.5007 (−19.5) | 0.4511 (−22.2) | 8.7–10.3 |
| ShallowFBCSPNet | official + aug8 (9×) | **0.4286 (−26.7; −15.7 vs its 1× run)** | stopped | 33 |

**Decision.** No net is within the brief's 8-point bar of X4f on fold 0: by the letter of
the brief the W3 deep-net lane is not triggered. Augmentation hurts the net more than
the Riemann model. (Earlier: EEGNet E0f lopo 0.474, ensemble with S64 +0.3.)

### P3b — X4f + ShallowFBCSPNet ensemble (exploratory, run in idle lane time)

Shallow 1× (as above) on all lopo folds + later, seeds 33–35 (25–30 min per seed on one
4-thread lane). Ensemble = log-probability average of X4f's streamed probabilities
(the dumps) and the net's (seed-averaged), net weight w. **Nested** row: for test person
k, w is the grid value that maximises the mean cell score over the other nine lopo folds
(never the test person); `later` uses the w picked on all ten lopo folds.
`analysis/sprint1106_ensemble.py`; text `logs/sprint_1106/ens_x4f_shallow*.txt`.

| | lopo | Δ lopo [95 % CI] (up/10) | later | Δ later |
|---|---|---|---|---|
| X4f | 0.6275 | — | 0.6735 | — |
| Shallow alone, seeds 33 / 34 / 35 / 3-seed avg | 0.530 / 0.524 / 0.525 / 0.533 | −9.8 … −10.4 | 0.562 / 0.585 / 0.584 / 0.588 | −8.5 … −11.2 |
| ensemble, one seed, nested (33 / 34 / 35) | 0.6381 / 0.6368 / 0.6366 | +1.06 [+0.12, +1.84] / +0.93 [−0.04, +1.80] / +0.91 [−0.76, +2.07] | 0.705 / 0.702 / 0.685 | +3.17 / +2.84 / +1.16 |
| **ensemble, 3-seed net, nested** (w = 0.3 in 9 folds, 0.4 in 1) | **0.6406** | **+1.32 [−0.12, +2.33] (8/10)** | **0.7117** | **+3.81** |
| ensemble, 3-seed net, fixed w = 0.3 | 0.6440 | +1.65 [+0.83, +2.40] (9/10) | 0.7117 | +3.81 |
| ensemble, 3-seed net, fixed w = 0.2 / 0.4 / 0.5 | 0.6396 / 0.6407 / 0.6346 | +1.21 / +1.32 / +0.71 | 0.688 / 0.702 / 0.691 | +1.4 / +2.8 / +1.8 |

**Reading (rules v2).** The only lever of the day that moves: +0.9 to +1.7 on lopo for
every seed and weight in 0.2–0.4, positive on `later` every time (+1.2 to +4.1). The
fully nested 3-seed row is just short of **lean** (CI lower −0.12); the fixed w = 0.3 row,
the weight the nested selection takes 9 times in 10, is lean. Below the +2.0 adopt bar.
A Shallow net 10 points weaker than X4f still adds independent information: the two
families err on different windows.

**5 seeds (33–37, net probabilities averaged; seeds 36–37 run 12:58–13:22):** Shallow alone
lopo 0.5312 / later 0.5803; ensemble **nested: w = 0.3 picked in all 10 folds → lopo
0.6444, +1.69 [+0.94, +2.38] (9/10); later 0.7013 (+2.78)** = **lean** by rules v2
(below the +2.0 adopt bar). w = 0.2 / 0.4: +1.03 / +1.21. This is the weekend's starting
point (`prompts/2026-11-06_weekend_ensemble.md`).

## W2 screen pulled forward — label-free test-time adaptation of X4f's classifier

Not in Sprint 1's plan; run because the lanes left time. `analysis/sprint1106_dump.py`
replays every test stream (fresh copy, `reset_state()`, one window per call) with the
classifier wrapped in a recorder: the features *as the classifier sees them* (after
stream whitening, fcenter, fscale) + the training features. All 11 splits reproduce X4f
exactly (lopo 0.6275, later 0.6735). `analysis/sprint1106_tta_probe.py` (causal,
label-free, reset per stream; post hoc, minutes):

| probe | lopo | Δ lopo [95 % CI] (up/10) | later | Δ later |
|---|---|---|---|---|
| X4f (base, reproduced) | 0.6275 | — | 0.6735 | — |
| EM prior adjustment (Saerens), n0 16 | 0.5943 | −3.32 [−4.77, −1.23] (3) | 0.6192 | −5.44 |
| bias-only information maximisation, lr 0.5 | 0.6127 | −1.48 [−2.56, −0.39] (2) | 0.6520 | −2.16 |
| same, lr 0.1 | 0.6196 | −0.79 [−1.88, +0.11] (3) | 0.6601 | −1.34 |
| T3A, M = 5 / 20 / 50 | 0.5495 / 0.6193 / **0.6283** | −7.79 / −0.82 / **+0.08 [−1.24, +1.33] (6)** | 0.5562 / 0.6560 / 0.6747 | −11.73 / −1.76 / +0.11 |
| pseudo-label class-mean shift, λ 0.3 / 0.5 × τ 0.5 / 0.6 / 0.7 | 0.6204–0.6246 | −0.29 … −0.71 | 0.6642–0.6747 | −0.93 … +0.12 |

**Why nothing moves.** In a stream, every class cluster sits uniformly closer to the
centre than its training mean (W_c·(μ_stream,c − μ_train,c) ≈ −1.6 to −1.8 for all three
classes on fold 0): the shift is a contraction that fcenter / fscale already undo, not
a per-class translation, so the LDA-style bias update moves all logits alike.
**Decision: no gain** (best +0.1, CI crosses 0). The 11-04 class-balance probe agrees.

## Diagnostics (X4f dumps, `sprint1106_diag.py`)

| | lopo | later |
|---|---|---|
| accuracy, stream windows 0–15 | 0.563 | 0.578 |
| windows 16–63 (≈ calibration runs) | 0.67–0.69 | 0.68–0.83 |
| windows 64+ (online runs) | 0.60–0.62 | 0.64–0.69 |
| calibration runs / online runs | 0.653 / 0.612 | 0.696 / 0.661 |
| recall mi / sub / word | 0.62 / 0.64 / 0.62 | 0.57 / 0.72 / 0.74 |

The stream start-up (first 16 windows, before the reference settles) costs ≈ 9 points on
8 % of the windows (≤ 0.8 overall); errors are spread over all class pairs.

## Deviations from the brief

1. **C2f+aug4 stopped after fold 0** (11:13; −13.7): finishing it would have taken
   ≈ 2.5 h of a lane for no decision value. The lane ran **X4f + near2s:sep** instead,
   the mechanism test (small shifts keep the time-locked features; per-offset centring).
2. **W2 adaptation screen pulled forward** from the weekend into the idle lane time.
3. **Shallow + aug8** added to P3 (the nets' natural augmentation) and its `later`
   split stopped after fold 0 at −15.7.

## Timing

| step | estimate | actual |
|---|---|---|
| P1 caches + gates | 60–90 min | 13 s + 35 s (+ 3 min parity), under |
| P2 X4f+aug4 | 70 min (lane ETA) | 1 h 50 min (10:58–12:48; 5× rows ⇒ 7–12× fit under contention), 1.6× over: diagnosed 11:10, real compute, not a stall |
| P2b near2s | 75–90 min | 71 min (11:13–12:24), under |
| Shallow seed 33 lopo 1–9 | 40–45 min | 25 min, under |
| P3 nets 1× | 25–30 min | 26 min, on |
| P3 Shallow aug8 fold 0 | ≈ 25 min | 33 min (1.3×) |
| dump X4f (11 splits) | 40–45 min | 30 min, under |
| TTA probes | 8–10 min | 2 min, under |
