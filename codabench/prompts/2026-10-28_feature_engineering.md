# Sprint brief (pre-registered, not yet run): task-specific features on top of C2f

Written 2026-10-28, before any number for these features exists. **Not launched.**
Time frame is set when the user starts it (needs ~4.5 h attended-or-not for Phases
0–2, then ~5 h unattended for the nested check, e.g. overnight). Run it with the
`project-sprint` skill, pointing it at this file.

## 1. Why this sprint

The three classes recruit different networks (neuroscience report, 10-15): motor
imagery = sustained hand-area mu/beta decrease, contralateral to the non-dominant hand
(side unknown: handedness withheld); subtraction = frontal-midline theta increase plus
parietal alpha decrease, ramping over the window; word generation = left
fronto-temporal alpha/beta decrease, bursty (one retrieval per word). The current
features (6 band covariances in tangent space, broadband, log-variance, xDAWN, bpt4,
icoh: ~8,600 numbers) capture these only implicitly. This sprint adds a few
**compact, hypothesis-driven blocks** and asks whether a pooled classifier uses them.

Honest prior: the gain is likely small (0–2 points). Linear combinations of features
already in the vector add little to a linear model, so each block below is chosen for
something the vector cannot form linearly: per-window normalisation, channel pooling,
a handedness-invariant magnitude, or within-window variability. A null result is a
valid outcome; the sprint then still delivers the X4f nested verdict (Phase 3).

## 2. Context (verify in Phase 0)

- Branch `feat/codabench-track2`; state of record `HANDOFF.md` § 0, `DIRECTIONS.md`.
- Base **C2f** = F64T + `fbands=1to4_4to8_8to13_13to20_20to30_30to45` +
  `xblocks=bpt4_icoh`: **lopo 0.6083, later 0.6424** (rows `logs/proteus_w1016/J2.jsonl`).
  **X4f** = C2f + `fscale=True clf=logreg0.01`: lopo 0.6275, later 0.6735 (rows
  `K2.jsonl`), pending its nested check.
- Harness: `analysis/proteus_sweep.py` (fits, B = 1 streamed scoring, rows carry
  y_true / y_pred, resumable; `--protocols nest` for the nested check);
  `analysis/proteus_w1016_summary.py` (paired people bootstrap, 10,000).
- Runtime: a C2f-sized arm ≈ 35–40 min per lane (lopo 10 folds + later), lanes at
  4 threads with `OMP_WAIT_POLICY=PASSIVE` (10-27 incident: 10-thread lanes spin-wait).
- **Already measured, do not repeat:** finer bins (bpt8), delta bins (dbpt8), slow
  waveform, fscale / logreg alone, CAR, clipping, crop, xDAWN removal, router.

## 3. The new blocks (all opt-in `xblocks` names; defaults bit-identical)

Channels by name (the solver's `_match_channels` order), asserted present. **AF3/AF4
are excluded from every new block**, so a gain cannot be more eye signal. Bands = the
six fb6 bands. Band power = mean of the squared band-passed window.

ROIs on this montage (no Fp, F7/8, T7/8, O channels):

| ROI | Channels | Hypothesis |
|---|---|---|
| FM frontal midline | Fz, FCz, F1, F2, FC1, FC2 | subtraction theta increase |
| LIF left inferior-frontal | F5, F3, FC5, FC3, C5 | word alpha/beta decrease |
| RIF right counterpart | F6, F4, FC6, FC4, C6 | reference for LIF |
| LSM left hand area | C3, C1, CP3, CP1 | motor imagery (left-handers) |
| RSM right hand area | C4, C2, CP4, CP2 | motor imagery (right-handers) |
| PAR parietal | P1, P2, P3, P4, Pz, CPz | subtraction alpha decrease |
| LPT left posterior | CP5, TP7, P5, PO7 | word (posterior temporal) |
| RPT right posterior | CP6, TP8, P6, PO8 | reference for LPT |

| Block | Definition | Size | What is new for a linear model |
|---|---|---|---|
| `roi` | per ROI and band: log(ROI-mean band power) − log(ROI-mean 1–45 Hz power) | 48 | per-window relative power (a window's broadband gain, e.g. a 4,000 µV excursion, cancels); channel pooling |
| `asym` | per homologous pair (17 pairs, AF excluded) and band: log power left − log power right | 102 | compact laterality prior (redundant in principle with bpt4; tests whether a compact form helps under shrinkage) |
| `asymabs` | \|asym − the stream's running mean of asym\| (same buffer 64, prior K = 8 as `fcenter`), appended **after** feature re-centring | 102 | handedness-invariant MI laterality magnitude; needs the stream state (most invasive change) |
| `burst` | per ROI and band: log of the coefficient of variation of the band-power envelope (250 ms moving mean) over the window | 48 | within-window variability: bursty word vs ramping subtraction vs sustained MI |

Not added, with reason: pre-named ROI contrasts (e.g. FM theta − PAR alpha) are linear
in `roi`; power slopes are linear in bpt4; per-channel `burst` (246) only if the ROI
version is lean.

Mechanistic check (reported, never decisive): per-class recall change vs base,
computed from the stored predictions. Expected: `asym`/`asymabs` → `mi` (and `word`
via LIF), `roi` → `sub`, `burst` → `word`. A gain on the "wrong" class is a flag to
look at, not a reason to drop the block.

## 4. Rules

- Commit on `feat/codabench-track2`, prefix `codabench:`, push to `personal` and `org`.
  No uploads, tags, merges, PRs. `logs/` ignored except `RESULTS*.md` (force-add).
- Defaults bit-identical: zhou regression `train 0.770000 replay 0.770000 EQUAL`;
  C2f re-run equals `J2.jsonl` to 6 dp on lopo fold 0 + later.
- Evidence: paired people bootstrap over the 10 lopo people; `later` (2 people) is
  reported, never decisive alone. One fixed seed. No setting is tuned on lopo/later
  beyond the arm list below.
- ≤ 3 lanes × 4 threads, `OMP_WAIT_POLICY=PASSIVE`, scoring at 1 thread. Lanes
  `P1`, `Q1`, `R1` (Phase 1), `P2`, `Q2` (Phase 2), `NF*` (Phase 3).
- Estimate, time-log and investigate overruns (> 1.5× ETA) as in `../CLAUDE.md`.

## 5. Phases and decision rules

### Phase 0: implement + smoke (estimate 1.5–2 h coding, 10 min runs)
Add the four blocks to `solvers/bci_decoding/riemann_sealed.py` (parse in
`_parse_xblocks`, compute in `_xblock_raw` on `parts["fb_bands"]`; `asymabs` in the
stream path after `fcenter`/`fscale`, with its own running state emptied by
`reset_state()`). Unit checks: block sizes 48 / 102 / 102 / 48; no NaN on the
±4,000 µV windows; B = 1 equals batch per stream for `asymabs`.
**Done when:** zhou EQUAL, C2f re-run equal, each arm scores > 0.4 on `later`,
stateful gates PASS for `asymabs`. Otherwise fix first; drop an arm only if the fix
takes > 30 min, and say so.

### Phase 1: single blocks (estimate 1.7–2 h, 3 lanes)
Primary, decides adoption — on **C2f**: `C2f+roi`, `C2f+asym`, `C2f+asymabs`
(asymabs implies asym), `C2f+burst`. Secondary, reported — the same four on **X4f**
(feature scaling evens out block sizes, which shrinkage LDA does not). lopo 0–9 + later.
**Decision (rules v2, vs its own base):** adopted if Δ lopo ≥ +2.0 with CI lower
bound > 0 and Δ later > 0; lean if Δ lopo ≥ +1.0 with CI lower bound > 0; else no gain.

### Phase 2: combination (estimate 40 min, 2 lanes) — only if ≥ 1 block is adopted or lean
`C2f+F` = C2f + every adopted/lean block (from the C2f arms); `X4f+F` = X4f + every
adopted/lean block (from the X4f arms). Candidate rule as on 10-27: the simplest
config that passes vs C2f; a larger one only if ≥ 1.0 lopo better; rule-selected
(C2f path) before exploratory (X4f path) on ties.

### Phase 3: one nested check (estimate 4.5–5 h unattended; 3.5–4 h if Phase 2 skipped)
Re-select inside 9 people per outer fold, by the same rules, among
{F64, fb6, C2f, X4f} ∪ {Phase 1–2 passing configs}. F64 nest rows exist
(`N*.jsonl`). **Flag over-selection** if the nested lopo < selected lopo − 3 points.
This also settles the pending X4f question (DIRECTIONS decision 13) whatever the new
blocks do.

### Phase 4: bake (estimate 15 min per candidate) — only if a new config survives Phase 3
Thread-pinned official + all zips via `scripts/sprint1027_bake.sh`, every gate PASS,
ms/window reported. Official zip → warm-up check (user's action) before the all zip
goes to the sealed phase.

## 6. Deliverables
`logs/sprint_feat/RESULTS.md` (all arms, CIs, Graz/BrainHero and calibration/online
splits, per-class recall), LOG entry, DIRECTIONS § 1/2/4/6 + change log, HANDOFF § 0.

## 7. Contingencies
- Sealed phase news or a C2f sealed result arrives: note it, keep running.
- Phase 1 > 1.5× late: decide Phase 2 on the finished arms, report the rest.
- Nothing lean in Phase 1: skip Phase 2, run Phase 3 on {F64, fb6, C2f, X4f} only.

## 8. Addenda (time-stamped, each before the numbers it concerns)

- **18:40, Phase 0 running, before any Phase 1 number.** (a) The new blocks are
  computed on the *unwhitened* windows: the existing blocks see covariance-whitened
  windows, whose "channels" are mixtures, so ROI and pair identities only hold before
  whitening. They are appended last, so `fcenter` still removes session offsets
  (e.g. the left-mastoid reference's asymmetry). `align="stream"` only.
  (b) **Phase 3 rule, made exact** (`analysis/sprint1028_nested.py`): walk the configs
  in the order F64 → fb6 → C2f → (C2f+F) → X4f → (X4f+F), starting from pick = F64; a
  config replaces the pick when its inner score ≥ pick + bar **and** ≥ F64 + 2.0, with
  bar = 2.0 if it adds a task-driven block the pick lacks, else 1.0. Same walk on the
  full lopo = "selected". Flag over-selection if nested < selected − 3 points.
  (c) Lane R1 starts C2f's inner folds after its two arms (Phase 3 needs them in every
  outcome); still ≤ 3 lanes.
