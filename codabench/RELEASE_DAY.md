# Release day, revised: from the PROTEUS release to a sealed-phase candidate

**Updated 2026-10-15.** The Track 2 training data was released on 2026-10-13
as **PROTEUS** (NEMAR `nm000290`), and the model-facing contract changed with
it. This document replaces the pre-release runbook, which was written against
a different plan (20 people × 6 sessions, 47 channels, labelled calibration
sessions, batched evaluation). That runbook is archived at
[archive/RELEASE_DAY_2026-10-01_pre-release.md](archive/RELEASE_DAY_2026-10-01_pre-release.md);
its rehearsal timings (§ 8), gate mechanics (§ 6 rule 7, § 7) and literal mock
log (§ 9) are still valid for the parts of the code that carry over.

What this document holds:

1. what the released data is (structure, sizes, lengths);
2. the contract we now train and score against;
3. what in the repository works as it is, what must be re-tuned, and what
   must be rebuilt;
4. topics the data raises that the old plan never had to consider;
5. the revised order of work to the sealed phase, with pre-registered rules;
6. where everything lives.

The executable first step is the setup brief
[prompts/2026-10-13_proteus_train_test_setup.md](prompts/2026-10-13_proteus_train_test_setup.md)
(**not run as of 2026-10-15**: no `env040.sh`, no `PROTEUS_SETUP.md`, no
`logs/proteus_setup/` exist yet). Every number below comes from that brief's
verified § 1, from the organisers' FAQ and code at upstream `442edcd`, or from
the BIDS files read on 2026-10-15; where a number is a projection it says so.
The scientific background on the three tasks is in
[reports/PROTEUS_task_neuroscience_2026-10-15.docx](reports/PROTEUS_task_neuroscience_2026-10-15.docx).

## 0. What changed, in one table

| | Planned for (pre-release docs) | Released (PROTEUS, 2026-10-13) |
|---|---|---|
| People × sessions | 20 × 6 (10 fully labelled, 10 with hidden sessions 4–6) | **10 people, 14 sessions**: 8 people × 1 session, P04 and P10 × 3 |
| Windows | ~14,400 (mock) | **2,655** (≈ 190 per session) |
| Channels | 47 = 43 EEG + 2 EMG + 2 EOG | **41 EEG**, no EMG/EOG ever |
| Rate and units | 500 Hz, scaled | 500 Hz source → **120 Hz**, 0.1–75 Hz, notch 50/60, **microvolts, no scaling or clamping** |
| Sealed test | hidden sessions 4–6 of 10 known people, with their sessions 1–3 as labelled calibration | **later sessions of these 10 people + 10 people absent from the release**; no calibration data for anyone |
| Evaluation | batches of 64, order unknown | **streamed, B = 1**, recording order, one stream per person-session, fresh `copy.deepcopy` + `reset_state()` per stream |
| Test-time adaptation | rule unknown (decision 1) | **allowed** within a session, label-free, causal |
| Metric | cell-averaged balanced accuracy | same: person × session × interface cell, averaged |
| Warm-up | Dreyer 2023, 2 classes, 27 ch | **PROTEUS, organisers' split** (test = P04 × 3 + P09; labels public) |
| Uploads | 1/day sealed | 1/day, **10 in total**; ranking = best of a team's **last five** sealed uploads; 3,600 s per run |
| Environment | `~/neuralbench/.venv` (0.3.1), `~/codabench/2026-competition` | `~/neuralbench/.venv040` (0.4.0), `~/neuralbench/competition-040`, data home `~/neuralbench/benchopt_data_040` |

## 1. The released data

### 1.1 Identity and provenance

- **PROTEUS BCI Bordeaux**, NEMAR `nm000290` v1.0.0, CC-BY-4.0, DOI
  `10.82901/nemar.nm000290`. Authors Dreyer, Bourdil, Bechon, Kojima, Velut,
  Rimbert, Roy, Lotte (Inria Bordeaux / Potioc). Funding: ANR PROTEUS.
- Recorded with a 64-channel actiCAP slim / actiCHamp (Brain Products);
  **41 EEG channels released**, 500 Hz, **reference M1 (left mastoid)**,
  ground FpZ, EDF+ in µV with per-file digital scaling.
- The release is a subset of a larger multi-session user-training study
  ("subset of subject-sessions; relabelled subjects"), which is where the
  sealed test's later sessions and ten unseen people come from.
- `participants.tsv` gives only a binned age (18–29: P01, P02, P03, P05, P10;
  30–44: P04, P06, P07, P08, P09). **Handedness, sex and the questionnaires are
  withheld.** "Non-dominant hand" therefore cannot be resolved to left or right
  per person from the metadata; it must be read off the EEG (§ 4.6).
- Size on disk in WSL: 1.5 GB BIDS + 730 MB extraction cache (2.2 GB total).
  Not mirrored to `Z:`.

### 1.2 Sessions and runs

Every session has 8 runs in this order: two 187-s resting baselines (run 0:
eyes closed, eyes open, fixation cross), then **6 cued runs numbered 1–6 in
recording order**:

| Run type | BIDS task | Feedback | Trials | Window count |
|---|---|---|---|---|
| Calibration, Graz interface | `AcquisitionGraz` | sham | 36 (12 per class) | 36 |
| Calibration, BrainHero interface | `AcquisitionBH` | sham | 33–43 (varies) | as trials |
| Online, Graz | `OnlineRawGraz` (2 runs) | real | 30 (10 per class) | 30 each |
| Online, BrainHero | `OnlineRawBH` (2 runs) | real | 30 (10 per class) | 30 each |

- Runs 1–2 are always the two calibration runs, runs 3–6 the four online runs;
  **which interface comes first differs between sessions**, and the online
  runs alternate or pair the interfaces in a session-specific order (e.g.
  P04/1: BH, Graz, BH, Graz, BH, Graz; P08/1: BH, Graz, Graz, Graz, BH, BH).
- Run durations (P04/1): calibration BH 423 s, calibration Graz 466 s, online
  BH 358–360 s, online Graz 391 s. A session is ≈ 46 min of recording.
- **84 cued runs** in total (14 × 6); the 28 baseline runs hold no cue and
  produce no window. The organisers' FAQ says the resting runs **may** be used
  for training or analysis; Codabench never sends them to `predict`.
- 13 cued runs have fewer trials than usual for their type; 1 run has an EDF
  resolution coarser than 1 µV per bit.

### 1.3 Trial timeline (what the 4-s window contains)

From the BIDS `events.tsv` (OpenViBE codes; cues 33024 `mi`, 33025 `sub`,
33026 `word`):

| Time (s, cue = 0) | Graz interface | BrainHero interface |
|---|---|---|
| −1.5 / ≈ −0.9 | `start_trial` + fixation cross | `start_trial` + fixation cross |
| −0.5 | auditory **beep** | *(no beep)* |
| **0** | **cue** (`mi` / `sub` / `word`, annotated duration 8 s) | cue |
| **+1 … +5** | **the delivered window** (480 samples at 120 Hz) | same |
| **+3** | `start_feedback` (bar feedback; sham in calibration runs) | `start_feedback` (game feedback; sham in calibration runs) |
| +8 | `end_trial` | `end_trial` |
| next cue | +12.5 fixed | +10.5 … +12.5 (jittered) |

Three consequences, new relative to every proxy we used:

1. **The window straddles feedback onset.** Seconds 1–3 of the window are
   task-only; seconds 3–5 carry feedback-driven activity (visual, evaluative).
   `bpt4`'s four 1-s bins land exactly on this boundary (bins 1–2 before
   feedback, 3–4 after), which is a reason to keep it.
2. **In online runs the feedback is label-correlated** (it follows the online
   classifier's output, which is right more often than not), **in calibration
   runs it is sham**. A model can therefore learn feedback-evoked activity that
   is informative in online runs and uninformative in calibration runs. Both
   run types sit in every cell, so the cell metric averages over the two.
   Score calibration and online runs separately in every local evaluation
   (§ 4.1).
3. **Graz trials have a beep, BrainHero trials do not**, and the interfaces
   differ in visual load. That is what the per-interface cell and the
   `subject_context` alignment are for.

### 1.4 Windows: counts, cells, lengths

- **2,655 windows**, classes balanced: `mi` 885 / `sub` 886 / `word` 884
  (labels 0 / 1 / 2, in that order).
- Per session, in the order P01, P02, P03, P04/1, P04/2, P04/3, P05, P06, P07,
  P08, P09, P10/1, P10/2, P10/3: 190, 189, 189, 189, 189, 188, 190, 187, 189,
  189, 189, 192, 186, 199. So **a stream is 186–199 windows**, about 38 min of
  trials.
- **28 cells** (14 sessions × 2 interfaces). Graz cells have exactly 96 windows
  (36 + 2 × 30); BrainHero cells 90–103. Every cell and every run holds all
  three classes.
- Within a stream the windows arrive by run number, then cue time: **the two
  calibration runs come first**, i.e. the first ≈ 70 windows of every sealed
  stream are sham-feedback trials in one interface, then the other. A
  within-session adaptation scheme sees one interface for the first ~35
  windows.
- Amplitude: 925 windows (34.8 %) exceed 200 µV in absolute value somewhere;
  extremes −4,417.8 and +3,511.5 µV. No NaN, no flat channel.

### 1.5 Channels and montage

Order as delivered (41): C2, CP2, P2, Pz, CPz, Cz, C1, CP1, P1, P3, PO7, P5,
TP7, CP5, CP3, C3, PO8, P6, P4, TP8, CP6, CP4, C6, C4, FC6, FC4, FC2, F6, F4,
F2, AF4, AF3, F5, F3, F1, Fz, FC5, FC3, FC1, FCz, C5.

- A **motor-centred montage**: dense FC / C / CP / P rows (1–6 on each side),
  midline Fz–FCz–Cz–CPz–Pz, frontal cover to AF3/AF4, lateral to F5/F6,
  FC5/FC6, C5/C6, CP5/CP6, TP7/TP8, posterior only to PO7/PO8.
- **Absent:** Fp1/Fp2, F7/F8, T7/T8, P7/P8, O1/O2/Oz, POz. So no occipital
  channel, no anterior temporal channel, and the eye-nearest channels are
  AF3/AF4.
- **Single-mastoid reference (M1).** Every channel is measured against the
  left mastoid, which adds a left-weighted common component. The pre-release
  finding "re-referencing / CAR / Laplacian: no gain" was made on proxies with
  other references and must be re-tested here (§ 3.2).
- The solver matches channels **by name** (`_match_channels`), so the 41
  names are what the joblib will carry.

### 1.6 Delivery to the model

`predict` receives one float32 tensor `(1, 41, 480)`: 120 Hz, 4 s from 1 s
after the cue, 0.1–75 Hz band-pass and 50/60 Hz notch applied at 500 Hz
before resampling, **microvolts, no scaling, no clamping**. Nothing else: no
person, session, run, interface or label. The NeuralBench stream config that
fixes this is `tasks/eeg/_motor_imagery_stream/config.yaml` in the 0.4.0
package (`start: 1.0`, `duration: 4.0`, `scale_factor: 1e6`, `scaler: null`,
`clamp: null`). Our own preprocessing of the training data is allowed, but
the test tensor is fixed; a submission cannot request 500 Hz data.

### 1.7 Sizes of the training and test sets we can build

Protocols as defined in the setup brief § 3 (stable keys: `subject`,
`session`, `run`, `task`, `start`):

| Protocol | Train | Test | Streams / cells | Stands for |
|---|---|---|---|---|
| `official` | P01 P02 P03 P06 P07 P10 (1,521); validation P05 P08 (379) kept apart | P04 × 3 + P09 (755) | 4 / 8 | the Codabench warm-up |
| `lopo` | all sessions of 9 people | the held-out person (190, 189, 189, 566, 190, 187, 189, 189, 189, 577 for P01–P10) | 14 / 28 over 10 folds | the 10 **unseen** people of the sealed test |
| `later` | session 1 of all ten (1,893) | sessions 2–3 of P04 and P10 (762) | 4 / 8 | **later sessions** of known people |
| `all` | all 14 sessions (2,655) | none that means anything | — | the training set of a real submission |

Noise floor: on an 8-cell protocol the cell mean of a label-blind model has an
SD of about 0.025, so differences under ~0.05 mean nothing there; `later`
tests two people. `lopo` over 28 cells is the least noisy local number.

## 2. The contract

- **Streamed inference.** `predict(X)` is called once per window, in
  recording order, one stream per person × session. Each stream runs on a
  fresh `copy.deepcopy` of the loaded model, whose optional `reset_state()` is
  called first. The evaluator (`benchmark_utils/streaming.py`) checks
  contiguity and time order and raises otherwise.
- **Adaptation** using the current and earlier windows of the current session,
  without labels, is explicitly allowed, including statistics-based alignment
  and gradient updates inside `predict`. Nothing crosses a session boundary.
  State may persist across the Graz and BrainHero runs of one session.
- **Output:** one integer per window, `(B,)`: 0 `mi`, 1 `sub`, 2 `word`.
- **Metric:** balanced accuracy within each person × session × interface cell,
  averaged over cells (`objective_balanced_accuracy_cell_mean`). Pooled
  balanced accuracy, session-mean balanced accuracy and accuracy are
  diagnostics.
- **Sealed test (Oct 28 – Nov 21):** later sessions of P01–P10 plus 10 unseen
  people; no calibration data through `fit`; identities, session allocation
  and stream order private. 1 upload per day, **10 in total**; the ranking
  takes the **best of a team's last five** sealed uploads; **3,600 s** per
  run; top 3 audited by re-running training.
- **Warm-up (ends Oct 25, 19:59 EDT; 5 uploads/day; does not rank):** PROTEUS,
  organisers' split (train P01 P02 P03 P06 P07 P10; test P04 × 3 + P09; 8
  cells). The test labels are public, so the warm-up is a **format and
  runtime check**, not a model comparison. The server runs
  `BCI[study=dreyer2026proteus,subset=full]`, seed 42, image
  `lionelyneuro/neural-compet:v3` (our 2026-10-02 deployment test ran on
  `v2`).
- **Organisers' reference numbers** (public cross-subject split, NeuralBench
  `bal_acc_stream_mean`, seeds 33–35): constant 33.3 %, EEGNet 32.6 %, CTNet
  35.0 %, REVE fine-tuned 36.0 %. `MeanLogReg` on the warm-up split: cell mean
  0.395863 (8 cells). These are chance-level baselines; the task is hard
  cross-subject.

## 3. What carries over, what needs tuning, what must be rebuilt

### 3.1 Works as it is (verified on 2026-10-13 or by reading the code)

| Component | Where | Status on PROTEUS |
|---|---|---|
| Riemann-Sealed inference at B = 1 | `solvers/bci_decoding/riemann_sealed.py` | `predict` on `(1, C, T)` returns `(1,)`; deep-copyable; B = 1 vs batch agree to 2.4e-13; ≈ 8.5 ms per window on microvolt input |
| Feature blocks at 120 Hz | same file | filter bank 4–8 / 8–13 / 13–30 / 30–45 Hz, broadband TS, log-variance, `bpt4`, `icoh` all fit under a 60 Hz Nyquist; 4,646 features without `bpt4`, ≈ 5,300 with it at 41 ch |
| Dual (n < p) shrinkage LDA | same file | applies above 4,000 features; fit on 1,521 windows took 5.9 s with `personal="pooled"`, `align="none"` |
| Channel matching by name | `_match_channels` | 41 names, all typed EEG from the names |
| joblib loads on the scoring server | deployment test 2026-10-02 | proved on image `v2`; re-prove on `v3` with the first warm-up upload |
| Organisers' evaluator and metric | `benchmark_utils.streaming.predict_streams`, `metrics.balanced_accuracy`, `group_scores` | call directly; `predict_streams` on the official test loader reproduces benchopt's scores in 0.6 s |
| A benchopt `Dataset` of our own by path | `datasets/mock_sealed.py` as template | `-d "<file>.py[protocol=lopo,fold=3]"` verified on benchopt 1.10.0 |
| EEGNet-StepType solver file | `solvers/bci_decoding/eegnet_steptype.py` | contract-compatible after retraining; see § 3.2 for input scaling |
| Zip format | `python -m zipfile`, files at the root | unchanged |

### 3.2 Needs tuning or re-testing on the real data

| Item | Why | What to do |
|---|---|---|
| **Path for unseen people** | The router + personal LDAs assume every test person was seen in training. With the file's defaults an unseen person is routed to *some* training person and gets that person's whitening and personal LDA, silently (`:1204–1212`, `:1625–1634`). Half the sealed test is unseen people | Add an open-set fallback: route only when the router posterior is high **and** the window's covariance is close to that person's reference; otherwise global whitening + pooled LDA. The existing 1st-percentile / 0.5 cap was tuned for closed-set routing. Score on `lopo` (must not hurt) and `later` (should help) |
| **Online re-centring at B = 1** | Now allowed. The buffer of 64 was chosen on batched proxies. A stream has 186–199 windows and starts with ~35 windows of one interface then ~35 of the other | Choose N ∈ {16, 32, 64} and a warm-up rule (how many windows before the running reference replaces the global one) on `lopo` and `later` through the streamed evaluator. Reference per session, not per (session, interface), unless the context ablation says otherwise |
| **Blend weight `auto`** | LOSO over calibration *sessions* needs ≥ 2 sessions per person; only P04 and P10 have them. On 8 people the fold is undefined | Replace the fold rule by leave-one-run-out within session 1 (6 runs), or fix w from `later`. The strict-reference logic (`wcv_ref="strict"`) stays |
| **xDAWN** | Its premise is a cue-locked evoked response. The window starts 1 s after the cue, but feedback onset sits at 2 s into the window and is class-correlated in online runs only | Keep as an ablation (`xd0`), decide by the rule in § 5. Check the GFP peak ratio per run type with the EDA |
| **Amplitude handling** | 35 % of windows exceed 200 µV; no EOG/EMG to regress on; OAS covariances are not robust to a 4,000 µV excursion | Test (a) nothing, (b) per-window clipping at ±k·MAD, (c) a robust covariance (e.g. Huber-weighted), as opt-in solver settings. Pre-register: adopt only if ≥ 2 points on `lopo` with the CI excluding 0 |
| **Reference** | M1 mastoid, left-lateral; our "no gain from CAR / Laplacian" came from differently referenced proxies | Re-run the `reference` ablation (`none` / `car` / Laplacian) on `lopo` |
| **Filter-bank bands and `bpt4` bins** | Bands are sensible for the three tasks (theta for subtraction, mu/beta for MI, alpha/beta for word generation); the 1-s bins straddle feedback onset | Keep; optionally add a 2-s split (`tseg2`) as an ablation. Do not add bands below 4 Hz without EOG |
| **EEGNet-StepType** | Input is now unscaled microvolts; its `standardize` option and reference were tuned on Dreyer | Retrain with per-window standardisation on; expect chance-level cross-subject unless aligned (the organisers' EEGNet: 32.6 %) |
| **Class priors** | Balanced in training and in every cell | uniform priors are right; leave |
| **Routing by context** (`subject_context`) | 10 training people × 2 interfaces = 20 pairs, ~95 windows each in session 1 | feasible; rule 2 of § 5 decides |

### 3.3 Must be rebuilt or replaced

| Item | Why it is unusable | Replacement |
|---|---|---|
| `scripts/train_sealed.sh`, `release_ablations.sh`, `sealed_lib.sh`, `install_xsess_overlays.sh`, `prepare_track2_data.sh` | all source `env.sh` (0.3.1 environment, old clone, old data home); the NeuralBench overlay mechanism they drive (`<study>_xsess`) does not match the stream task | `scripts/proteus_candidate.sh` (bake → train → replay → equal → zip) from the setup brief Phase 5; ablations by protocol through `datasets/proteus_protocols.py` |
| `analysis/xsess_cache.py`, `xsess_lib.py`, the `sealed_*` harness (`sealed_personal.py`, `sealed_run.py`, `sealed_stream.py`) and `release_eda.py`, `release_summarize.py` | wired to the old clone on `sys.path`; the harness mis-scores PROTEUS without failing (wrong default split, 14 cells instead of 28, batches only) | `analysis/proteus_cache.py`, `proteus_splits.py`, `proteus_eval.py` (route A: `make_loader` + `predict_streams`), `proteus_selftest.py`; a new summariser for the § 5 rules |
| Every existing joblib and zip | trained by the old loader (other channel sets, scaled input) | retrain everything |
| `config/nb_overlays/*`, `datasets/mock_sealed.py` as data | mock structure (20 × 6, 47 ch) | keep only as code templates |
| Pre-release rule 1 (channels) | no non-EEG channel will ever arrive | dropped |
| `SEALED_RECIPE.md` § 1 "calibration blend" | there is no calibration data in the sealed phase | the blend applies only to the 10 represented people, with their public sessions as "calibration" |

## 4. Notable topics the old plan never had to consider

### 4.1 Two kinds of run inside every cell

Calibration runs (sham feedback) and online runs (real feedback) are mixed in
every cell and in every stream. Any local evaluation must report cell scores
split by run type as well; a model whose online-run score is far above its
calibration-run score is reading the feedback, and the sealed streams also
start with ~70 calibration windows where that signal is absent.

### 4.2 Interface as context

The Graz bar and the BrainHero game differ in beep, visual load and reward.
The cell metric scores them apart. The alignment choice is between one
reference per session (simpler, 190 windows) and one per (session, interface)
(halves the data per reference). With adaptation allowed, the per-interface
reference can be built online: the buffer naturally holds mostly one interface
for ~35 windows at a time.

### 4.3 Two populations in one test

Half the sealed cells are later sessions of P01–P10, half are unseen people.
A model should be evaluated on both halves separately (`later` and `lopo`)
and never on one number that mixes them. The components that help the first
half (router, personal LDAs) can hurt the second half if they misfire; the
open-set fallback of § 3.2 is what reconciles them.

### 4.4 Very little data, strong between-person variability

2,655 windows from 10 people; the organisers' deep baselines sit at chance.
Every gain must come from structure that generalises across people: canonical
physiology (contralateral mu/beta ERD for MI, frontal-midline theta and
parietal alpha for subtraction, left fronto-temporal alpha/beta for word
generation; see the neuroscience document), covariance alignment that removes
head-geometry and reference effects, and shrinkage. There is no room for a
feature search with 17 candidates as on 2026-09-29: with 28 `lopo` cells the
noise floor is ~0.02–0.03.

### 4.5 Session-to-session change is observable for two people only

P04 and P10 have three sessions each. That is the only local evidence about
drift and about user learning (an MI-trained user's ERD typically becomes
more focal over sessions). It is also 2 of the 10 people the sealed test
revisits. Treat `later` as a sanity check with wide error bars, not as a
ranking.

### 4.6 "Non-dominant hand" with handedness withheld

For right-handers the imagined hand is the left one and the ERD is
right-lateralised (C4 / CP4 / FC4); for a left-hander it is the mirror image.
The EDA should fit, per person, the sign of the C3-minus-C4 mu/beta ERD during
`mi` trials, and the model should not assume a fixed side. Laterality-aware
features (left–right differences) only help if the side is resolved per
person, which an unseen person's first windows cannot give; symmetric features
(both hemispheres' covariances) are safer for the `lopo` half.

### 4.7 Warm-up labels are public

A warm-up number is a plumbing check. The only reasons to upload before Oct 25
are to prove the streamed format, the `v3` image and the runtime; upload one
Riemann zip and one `MeanLogReg` zip (expected cell mean 0.395863 for the
latter) and compare the server's three numbers with the local replay.

### 4.8 Runtime is not a concern, but prove it

Sealed-test size is unknown. The brief's upper bound is 106 sessions × 199
windows ≈ 21,000 windows: at 8.5 ms per window ≈ 3 min, far under 3,600 s. A
per-stream `deepcopy` of a 5,300-feature LDA model is negligible. Online
re-centring adds one 41 × 41 inverse square root per window, also negligible.
Report ms per window in every candidate's RESULTS.

### 4.9 Things that are easy to get wrong in the new code

- The integer ids in `info` (`context_id`, `record_id`, `stream_id`) are
  numbered by first appearance per loader and are **not stable keys**
  (0 = Graz in train, 0 = BrainHero in the official test loader). Use the
  trigger-table columns.
- The train loader is shuffled; read items by index through `ds[i]` with
  `ds.seg_ds.triggers.iloc[int(ds.index[i])]`.
- `MeanLogReg` depends on training-row order (lbfgs tolerance): 0.393–0.397
  across orders. Exact gates compare one exported model across routes, never
  two trainings.
- benchopt prints no scores: read the parquet; without `--output` it lands in
  `$COMPET_ROOT/tracks/bci_decoding/outputs/`. An inference-only run without
  weights predicts class 0 silently (cell mean 0.333333).
- Any script that sources `env.sh` silently switches to the old environment.
- The old regression (`train_sealed.sh` on `zhou2016_xsess`, 0.770000) is
  still the gate for edits to `riemann_sealed.py`.

## 5. The revised plan to the sealed phase

Calendar: today 2026-10-15; registration closes **Oct 24**; warm-up ends
**Oct 25**; sealed phase **Oct 28 – Nov 21**. Thirteen days to the first
sealed upload.

| # | Step | Est. | Output |
|---|---|---|---|
| 1 | **Run the setup brief** (environment, window cache, four protocols, streamed evaluation, Riemann vehicle end to end, self-test) | ~7 h unattended | `PROTEUS_SETUP.md`, `logs/proteus_setup/RESULTS.md`, two "ready, not uploaded" zips |
| 2 | **Warm-up upload** of the `official` Riemann zip and the `MeanLogReg` zip (user's action) | 2 uploads | server numbers = local replay; `v3` image proven |
| 3 | **EDA on the real data**: per-person ERD maps per task from the raw BIDS (1–5 s vs the pre-cue second), laterality sign per person, GFP peak ratio per run type, calibration-vs-online split, drift of P04/P10 across sessions, amplitude distribution | ~half a day | `reports/proteus_eda/` |
| 4 | **Candidate ablations on `lopo` and `later`** through the streamed evaluator, each as an opt-in solver setting, one change at a time against the base (pooled LDA, global whitening, fb + bpt4, no xDAWN decision yet): online re-centring N; open-set router + personal blend; xDAWN; reference; amplitude handling; `subject_context`; `icoh` | ~1 day of CPU, fits are minutes | a RESULTS table with paired bootstrap CIs over people (10) and cells (28) |
| 5 | **Decide by the rules below**, bake the defaults into a candidate copy, train on `all`, replay from the read-only copy and from the unzipped folder, check train = replay | ~1 h | the sealed candidate zip, "ready, not uploaded" |
| 6 | **Second model** only if time remains: EEGNet-StepType retrained with standardisation and the same alignment; probability ensemble with the Riemann candidate | ~1 day | an ensemble zip, same gates |
| 7 | **Sealed-phase upload plan** (user's action): upload the candidate on Oct 28; keep ≤ 5 of the 10 uploads for variants; the **last five** uploads must contain the best model, so finish experiments by the fifth upload | — | SUBMISSIONS.md rows |

### Pre-registered decision rules, v2 (written 2026-10-15, before any PROTEUS model number exists)

Metric for every comparison: the cell-averaged balanced accuracy on `lopo`
(28 cells) **and** on `later` (8 cells), reported with a paired bootstrap over
people (10,000 resamples). A change is adopted only if it does not lose on
either protocol and gains on at least one as stated. Calibration-run and
online-run cell scores are reported alongside every row.

1. ~~Channels~~ (dropped: EEG only).
2. **Context alignment** (`subject_context`, or per-interface online
   reference): adopt if ≥ 2 points on `lopo` with the CI excluding 0.
3. **Feature blocks:** keep `bpt4` unless the base without it is ≥ 1 point
   better on `lopo`; add `icoh` if ≥ 2 points with the CI excluding 0 (fit
   time is no longer a constraint at this size).
   3c. **xDAWN:** drop it if the row without it is ≥ 1 point better on `lopo`
   or if the calibration-run cells lose ≥ 2 points with it (feedback-reading).
4. **Blend weight and personalisation** (represented people only): the
   open-set router + blend is adopted if `later` gains ≥ 2 points **and**
   `lopo` loses < 1 point. w is chosen by leave-one-run-out on session 1.
5. **Pooling:** everyone, always (10 people; no subset makes sense).
6. **Online re-centring:** adopt the N with the best `lopo` score if it beats
   the clean base by ≥ 2 points with the CI excluding 0; prefer the smaller N
   on a tie. Warm-up rule: the running reference is blended with the global
   one by n/(n+N) until n ≥ N windows.
7. **Gates before a zip is kept** (unchanged in substance): train = replay
   exactly from the read-only copy and from the unzipped folder; the defaults
   in the zipped `submission.py` equal the decided settings; `predict_streams`
   through the organisers' evaluator and through our route give identical
   `y_pred`; a second training with seed 42 gives identical `y_pred` (the
   audit re-runs training).
8. **Amplitude handling and reference:** adopt only on ≥ 2 points on `lopo`
   with the CI excluding 0; otherwise keep the raw microvolt input and
   `reference="none"`.

## 6. Where everything lives

| What | Where |
|---|---|
| Upstream code (0.4.0, read-only) | `~/neuralbench/competition-040` (detached worktree of `~/codabench/2026-competition`, at `442edcd`; never `checkout/pull/reset/stash/fetch` there) |
| venv | `~/neuralbench/.venv040` (neuralbench 0.4.0, benchopt 1.10.0, scikit-learn 1.9.1, pyriemann 0.12, torch 2.14.1+cpu) |
| Data home | `~/neuralbench/benchopt_data_040` (BIDS under `neural_compet/Dreyer2026Proteus/download/nm000290`; extraction cache under `neural_compet/cache`) |
| Organisers' FAQ and code | `$COMPET_ROOT/codabench/pages/scientific_faq_bci.md`, `tracks/bci_decoding/{README.md,objective.py,datasets/bci_studies.py}`, `benchmark_utils/{streaming,data,nb_task,metrics}.py` |
| Our solver | `codabench/solvers/bci_decoding/riemann_sealed.py` (defaults: `personal="blend"`, `align="subject"`, `adapt="none"`, `buffer=64`, `xblocks=""`, `wcv_ref="all"`) |
| Setup brief (step 1) | `codabench/prompts/2026-10-13_proteus_train_test_setup.md` |
| Window cache, runs, logs (to be created by step 1) | `~/neuralbench/proteus_cache/`, `~/neuralbench/proteus_runs/<run>/`, `~/codabench/logs/proteus_setup/` |
| Old environment (frozen, do not use) | `~/neuralbench/.venv`, `~/neuralbench/benchopt_data`, `~/neuralbench/xsess_cache`, `codabench/env.sh` |
| Entry point for every WSL call | the activation block in the setup brief § 1 (`source ~/neuralbench/.venv040/bin/activate`, `BENCHOPT_DATA_HOME=~/neuralbench/benchopt_data_040`, `COMPET_ROOT=~/neuralbench/competition-040`), to become `codabench/env040.sh` |
| Scientific background on the tasks | `codabench/reports/PROTEUS_task_neuroscience_2026-10-15.docx` |
| Archived pre-release runbook | `codabench/archive/RELEASE_DAY_2026-10-01_pre-release.md` |

Git for our repository runs on the Windows side only (branch
`feat/codabench-track2`, push to `personal`); `codabench/logs/` is ignored
except for `RESULTS*.md` / `STATUS.md` added with `git add -f`.

## 7. Change log

- 2026-10-15: rewritten for PROTEUS. Old runbook archived. Data structure
  (§ 1), contract (§ 2), carry-over / tuning / rebuild tables (§ 3), new
  topics (§ 4), revised plan and rules v2 (§ 5). Neuroscience document added.
- 2026-10-01 and before: see the archived file's history.
