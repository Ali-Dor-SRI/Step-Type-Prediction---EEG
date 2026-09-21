# Changelog

## 2026-09-21 — Documentation corrections

Docs only: no code, config, model or result changed. Each value was re-read from
the run artifact or source line it cites.

- **CV design of the 0.655 run:** `bin_full_cnv_rich_mean_0125_xgb` and the five late-window binning runs are express tier, 5 splits × 2 repeats with 2 inner folds (10 folds per participant), not 5 × 20 — fixed in XGB_MODEL_SUMMARY §3.1/§3.4/§3.5, the 2026-06-12 entry below, `RICH_POOLING_SUMMARY.md` and the perf-loop `LEDGER.md`.
- **Inner-CV metric:** no config sets `modeling.scoring`, so the binary models' inner search scores accuracy (`models/train.py:791`); every binary gap (+0.169; +0.173 → −0.014; +0.198 → −0.039; +0.177 → −0.036) is inner accuracy − held-out AUC, and the XGB_MODEL_SUMMARY §3.1 column that read "Inner-CV AUC" is relabelled, as are the pooling tables in MODELS §7 and `docs/OVERFITTING_GAP_SOLUTIONS.md`.
- **Confidence intervals:** the `ci95` written by `evaluate.py` is fold-level (1.96·SD/√n over CV folds) and is now labelled so in the screening reports, MODELS §7 and XGB_MODEL_SUMMARY §3.3; headline results also carry a participant-level interval (0.655: ±0.060 participant-level vs ±0.025 fold-level), method in README → Confidence intervals.
- **Primary window:** MODELS §1/§2, XGB_MODEL_SUMMARY §3, SCRIPT_GUIDES §3.9 and a README example comment called late CNV (1–2 s) primary; `configs/default.yaml` makes full CNV (0–2 s) primary.
- **Late-window comparator:** the matched late-window value for 0.655 is 0.568 (same `rich_mean_0125` recipe, n = 20); the D1 rows had paired 0.65 with 0.58 (the n = 8 screen) or 0.56.
- **Pooling lift:** +0.031 (t = 1.27, n = 20) is from the reduced ~2.3k-feature fast set (`r1_pool_confirm20`) and +0.0386 (t = 1.17) from the rich set; each mention now names its set, and the 2026-06-10 entry below no longer calls the 4-fold × 1-repeat confirm "express CV".
- **BiLSTM:** MODELS §8 implied an invalidated LSTM result; there is none — it was left out of screening because the driver feeds one timestep per feature.
- **Feature-selection scope:** README presented the corr → k-best → RFECV → gain → SHAP funnel as the selection stage for every decoder; it runs for the tabular models only (stability selection is the default selector, RFECV legacy) and the tensor models skip it.
- **Status and wording:** MODELS §8/§9 now record the finished 20-subject pooling confirms, the README scope note no longer uses "real-time", and the 2026-05-29 screening summary carries a dated note that the default window has since changed.

## 2026-09-14 — PyTorch EEGNet port, Docker image, GitHub Actions CI (v2.6.0)

### Added

- **`models/eegnet_torch.py`** — a layer-for-layer PyTorch port of the Keras
  `models/eegnet.py`, registered as the `eegnet_torch` model / `--speed-tier
  eegnet_torch` (`configs/eegnet_torch.yaml`, a clone of `configs/eegnet.yaml`).
  The Keras model is unchanged. Two pieces: `EEGNetTorch`, a plain
  `torch.nn.Module` whose `forward` takes `(batch, n_channels, n_times)`
  (braindecode's input convention, so a braindecode/eegdash training loop could
  take it as is — not exercised here) plus an optional tabular tensor for the
  hybrid fusion branch;
  and `EEGNetTorchClassifier`, a hand-rolled scikit-learn estimator (no skorch)
  that runs unchanged under the nested-CV driver's GridSearchCV and
  `scripts/08_tensor_model_diagnostics.py`. CPU only, and neither skorch nor
  braindecode is imported by the port (the fold-local standardizer it reuses
  from `cnn.py` calls braindecode when installed, as the Keras path does).
  Mirrored from the Keras file: the layer sequence; the **max-norm
  constraints**, which PyTorch has no API for and which are therefore
  re-applied with Keras's own formula, over the same axes, after every
  optimizer step (depthwise spatial filters to 1.0, fusion and classifier
  dense weights to `norm_rate`); TensorFlow "same" padding; the channels-last
  flatten order; glorot-uniform init with Keras's fan convention; BatchNorm
  momentum/epsilon; Adam's epsilon; the L2 penalty on the tabular dense
  kernel; and the training loop (50 epochs, batch 16, `validation_split=0.2`,
  early stopping on `val_loss` with `restore_best_weights`). Unlike the Keras
  path, every fit seeds torch and numpy from `modeling.random_state`.
- **`tests/test_eegnet_torch.py`** (26 tests) — parameter counts against
  constants recorded from the Keras model (six shapes, including the real
  64 × 2049 tensor) *and* against a live Keras build; a forward pass with the
  Keras weights copied in, agreeing to `atol=1e-5`; the max-norm helper against
  Keras's `MaxNorm` and the bound holding after real training steps, with a
  control that fails if the constraint call is removed; Keras validation-split
  and early-stopping semantics; the sklearn wrapper; the overlay being a
  faithful clone; and an end-to-end `--model eegnet_torch` CLI run on
  `configs/smoke.yaml` plus the overlay. The live Keras comparisons skip where
  TensorFlow is absent (CI, the Docker image); the recorded constants cover
  those environments.
- **`Dockerfile` + `.dockerignore`** — CPU-only `python:3.12-slim` image
  installing `-e ".[dev,torch]"` from the PyTorch CPU index, `CMD` runs
  `pytest -q`. No TensorFlow. The ignore list keeps `data/`, `outputs/`,
  `legacy/`, the venvs, `docs/models_figs/`, `fsaverage/` and all
  `.fif/.parquet/.npz/.bdf/.h5` files out of the build context.
- **`.github/workflows/ci.yml`** — on push and pull request: `ruff check .` plus
  `pytest -q` on Python 3.12 with the CPU torch wheels (pip cached), printing
  the collected-test count; and a second job that builds the image and runs the
  suite inside it.
- **`[torch]` extra** in `pyproject.toml`, and a `[tool.ruff]` section that
  lints the package, tests and CLI drivers while excluding `outputs/` (recorded
  -result harnesses, never edited) and `scripts/stim_module/` (pre-existing
  style debt).
- **`tests/conftest.py`** — a `synthetic_epoch_tensor` fixture (shuffled
  `(n_epochs, n_channels, n_times)` data with a learnable class signal).

### Changed

- **`models/train.py`** — `eegnet_torch` added to `MODEL_FACTORIES` and
  `NEURAL_HYBRID_MODELS`. It is registered through a small `_lazy()` factory
  wrapper because its module imports torch: a classical XGB run, and each of
  its joblib workers, must not pay that import.
- **`models/normalization.py`** — routes `eegnet_torch` to the same fold-local
  exponential-moving standardizer the Keras EEGNet uses.
- **`run.py`** — imports `NEURAL_HYBRID_MODELS` from the training driver rather
  than keeping a second copy, and adds the `eegnet_torch` tier.
- **`scripts/04_train.py`, `07_feature_informativeness.py`,
  `08_tensor_model_diagnostics.py`** — `eegnet_torch` added to the `SPEED_TIERS`
  maps and to 08's tensor / full-CNV model sets.
- **`scripts/06_compare_runs.py`** — the five inline "single tier" model sets
  collapsed into one `SINGLE_TIER_MODELS` tuple, so registering a tensor model
  is one edit; `eegnet_torch` added there and to the name-inference fallbacks
  (before `eegnet`, so the longer token wins).
- **`tests/test_imports.py`** — `eegnet_torch` in the import list, plus
  `test_every_hybrid_neural_model_is_wired_into_every_registry`, which loops
  over `NEURAL_HYBRID_MODELS` and checks each one across the registry,
  normalizer, the four `SPEED_TIERS` maps, 08's sets, 06's tuple and inference,
  its `configs/<model>.yaml` overlay, and the SCRIPT_GUIDES value lists. The
  fsaverage preflight test now skips under
  `EEG_STEPTYPE_SKIP_FSAVERAGE_TESTS=1` (its BEM is a network download and is
  not in the repo), which CI and the image set.
- **Version 2.5.0 → 2.6.0**, matched across `pyproject.toml` (which had been
  left at 0.1.0), `CITATION.cff` and the README BibTeX block.
- Lint fixes for the CI gate: unused imports in `features/tensor.py`,
  `models/lstm.py`, `docs/make_models_figs.py`, `scripts/_xgb_perf_snapshot.py`;
  an ambiguous `l` in `preprocessing/filter.py`; an f-string without
  placeholders in `06_compare_runs.py`; `noqa` markers in `tests/conftest.py`.

### Results — a one-subject sanity check, not a parity claim

P13, `--speed-tier` runs, 2 outer folds × 1 repeat, full-CNV window, same
inputs (80 epochs × 64 channels × 2049 samples + 25,857 tabular features),
`.venv312`, 2026-09-14:

| model | fold AUCs | mean AUC | wall time |
|---|---|---|---|
| `eegnet` (Keras) | 0.690 / 0.540 | 0.615 | 59 s |
| `eegnet_torch` | 0.638 / 0.548 | 0.593 | 34 s |

One subject and two folds cannot establish equivalence. The two recorded Keras
EEGNet cohort runs put P13 at 0.44 (2026-05-29) and 0.615 (2026-05-30) under the
same config, because scikeras is unseeded — a run-to-run spread wider than the
gap between the two models here. The port has not been run on any other
participant.

### Notes

- **Inherited Keras behaviour, reproduced deliberately:** `validation_split`
  takes the *trailing* fraction of the training fold before shuffling. The
  epoch tensor is stacked One-then-Two and scikit-learn returns sorted fold
  indices, so that tail is almost entirely `Two` and early stopping watches a
  single-class validation set. Changing it would change `cnn`/`eegnet`/`eegnext`
  too, so it is left as a separate decision.
- **The 0.94 for P13 quoted in `MODELS.md` §5.7/§7** is the `baseline_auc` of
  `scripts/08_tensor_model_diagnostics.py`, which refits on all of a
  participant's epochs and scores those same epochs — an in-sample fit, as that
  script states, not a held-out result. Recorded in `MODELS.md` §5.7b.
- Test suite: **125 → 153 collected**, all passing in `.venv312`. Without
  TensorFlow (`.venv`, CI, the container) the 10 live-Keras parity tests skip.

## 2026-06-12 — Rich-pooling sub-loop: cheap ANOVA pre-filter (`modeling.pre_kbest`)

Sub-loop on `perf/agentic-improvements` testing cross-subject **partial pooling on the
RICH ~12k-feature set** (the prior loop ran only on the 2.3k fast set). One code change
unblocks it; everything else is config + docs.

### Added

- **`modeling.pre_kbest`** (default `null` = legacy/off) — a cheap univariate ANOVA top-K
  pre-filter in `models/train._fit_score_split`, fit on the **train fold only** and applied
  **before** the correlation drop. The correlation drop builds a dense `p x p` matrix per
  fold (~1.2 GB at 12.3k cols, ~8 GB at 31.7k) — the binding wall-clock cost of the rich
  path, made ~20x heavier by pooling. Cutting `p` with a linear-time top-K first collapses
  it (XGB_MODEL_SUMMARY §4). Leakage-safe (reuses `feature_selection.select_kbest`); the
  null default leaves the per-participant path byte-identical. Legacy/opt-out: leave unset
  or `null`. Enable with an int, e.g. `modeling.pre_kbest: 2000`.
- **`configs/pooling_compare_rich.yaml`** — rich pooled comparison overlay: blocks
  `amplitude(0.125, mean) + slopes + psd` (drops `src`+`cnv_benchmark` so all 20 subjects
  run without the eLORETA b0.125 caches that P35/P39 lack), fresh `cache_tag: rich_nosrc_0125`,
  the light pooled funnel (k_best 150, stability n_subsamples 8 / n_lambda 5), and
  `pre_kbest: 2000`.
- **`configs/pooling_rich.yaml`** — committed train-time overlay (the **recommended HONEST
  rich-pooled config**): the rich no-src frame + light funnel + `pre_kbest: 2000` +
  `modeling.pooling.mode: partial`. Use `scripts/04_train.py --model xgb --config
  configs/pooling_rich.yaml`. Legacy/opt-out: `modeling.pooling.mode: per_participant`.

### Result — CONFIRMED rich partial pooling (20-subject cohort, run `r_rich_conf20`)

| arm | cohort AUC | gap |
|---|---|---|
| recorded rich per-participant (heavy funnel + src, 5×2 express CV) | 0.655 | +0.169 |
| per_participant (matched: no-src, light funnel) | 0.5990 | +0.1978 |
| **partial (rich pooled)** | **0.6376** | **−0.0385** |

- **Paired partial − per_participant: +0.0386 AUC** (SE 0.033, t=1.17, 11/20 up) — clears the
  +0.03 bar; t=1.17 ⇒ real but **not** statistically significant (like the fast set's +0.031).
- **Gap collapses +0.1978 → −0.0385** — the robust headline; reproduces at 8 and 20 subjects,
  fast and rich. vs the recorded rich 0.655, the pooled 0.6376 is ~flat (−0.017, within noise)
  but **honest**. Pooling makes the project's best-AUC region trustworthy.
- The 8-subject screen was *pessimistic* (+0.0156) — its subset is enriched for strong subjects;
  the full cohort showed the real lift (shrinkage helps the harder subjects, drags the stars).
- `pre_kbest` global default stays `null` (tractability lever, AUC-neutral). `pooling.mode`
  global default stays `per_participant` (paradigm preservation). `src` not re-added (lower-EV).
- Full write-up: `outputs/perf_loop/RICH_POOLING_SUMMARY.md`; numbers in `LEDGER.md`. 120 tests green.

## 2026-06-11 — Perf loop concluded: plateau after Round 4 (v2.4.1)

The agentic perf-improvement loop reached its stopping condition (3 consecutive no-win
rounds). No code/behavior change — documentation and results only.

### Results

- **Rounds 2–4 produced no further XGB win.** Looser feature funnel (−0.039),
  richer/deeper search (+0.020 screen → +0.001 at cohort scale), and Legendre shape
  features (+0.008) are all null at the 20-subject scale. The pooled XGB is at its ceiling
  on the 2.3k fast feature set. Every promising 8-subject screen lift shrank toward zero on
  the 20-subject confirm — the confirm step killed 2 false positives.
- **CNN confirmation baseline** established: cohort AUC 0.5675 (gap +0.013, 18 subjects).
  No CNN improvement candidate was screened (cost-prohibitive in-session).
- Final: **XGB 0.5674 → 0.5957** (gap +0.166 → −0.014) via Round-1 partial pooling.

### Added / updated

- **`outputs/perf_loop/SUMMARY.md`** — baseline→final, ranked table of all 8 changes tried,
  the winner's key + legacy override, and recommended next steps.
- **`XGB_MODEL_SUMMARY.md` §3.5, `MODELS.md` §7** — pooling section updated with the
  confirmed 20-subject numbers (was "re-run to confirm magnitudes").
- `LEDGER.md` — Rounds 2–4 results and the loop-complete summary.

## 2026-06-10 — Cross-subject partial pooling wired into the train entrypoint (perf loop, v2.4.0)

Agentic perf-improvement loop, Round 1. Cross-subject **partial pooling** confirmed
on the full 20-subject cohort as the strongest lever on the inner-vs-outer overfit gap.

### Added

- **`models/train.py` — `run()` now routes on `modeling.pooling.mode`.** `partial`/`full`
  dispatch the pooled workflow (`models.pooling`) from the normal `04_train.py` path
  (previously reachable only via `scripts/09_pooling_comparison.py`). Tabular models only;
  **tensor models (cnn/eegnet/eegnext) and <2-subject cohorts auto-fall back to
  per_participant**, so the toggle is safe for neural runs and smoke configs.
- **`configs/pooling.yaml`** — committed overlay whose default *is* the improved behavior
  (`modeling.pooling.mode: partial`). Layer on any tier:
  `python scripts/04_train.py --model xgb --config configs/pooling.yaml`.
- **`outputs/perf_loop/`** — the loop's ledger (`LEDGER.md`, source of truth) and
  screen→confirm harness (`aggregate.py`, `screen.sh`, …). `*.log` are gitignored (90 MB+).

### Changed / new config key

- **`modeling.pooling.mode`** — now an explicit, documented key in `configs/default.yaml`.
  - **Default (legacy):** `per_participant` — one model per subject on ~80 epochs (the
    per-subject paradigm; global default unchanged, preserves chronological check + tests).
  - **Improved (confirmed):** `partial` — each subject's train split + all other subjects'
    epochs, same test folds (paired), subject-grouped inner CV. Enable via `configs/pooling.yaml`
    or set the key directly. `full` = leave-one-subject-out transfer.

### Results (20-subject cohort, 2.3k fast feature set, 4-fold × 1-repeat CV with 2 inner folds; `r1_pool_confirm20`)

| mode | cohort AUC | overfit gap |
|---|---|---|
| per_participant (baseline) | 0.5646 | +0.173 |
| **partial** | **0.5957** | **−0.014** |
| full | 0.5882 | −0.012 |

Partial vs per_participant (paired, same folds): **+0.031 AUC** (t=1.27, not significant —
the AUC lift is modest/noisy) and a **robust gap collapse (+0.173 → −0.014)**. Pooling
strictly dominates the objective (AUC not worse, guardrail far better). Default kept at
`per_participant` as a deliberate judgment call (a paradigm flip is disproportionate to a
t=1.27 lift); `partial` is the one-line, confirmed-better cross-subject option. 118 tests pass.

## 2026-06-08 — EEGNeXt sophisticated hybrid CNN

### Added

- **`models/eegnext.py`** — a more sophisticated CNN built on the EEGNet-lite
  block, registered as the `eegnext` model / `--speed-tier eegnext`
  (`configs/eegnext.yaml`). Three upgrades over `cnn`/`eegnet`: a **multi-scale
  temporal stem** (parallel temporal convs at several kernel lengths), **squeeze-
  and-excitation channel attention**, and **residual separable blocks**. Keeps
  the hybrid tensor + tabular fusion (`require_source: true`) and the full-CNV
  window. TensorFlow imports stay deferred so the package still imports without
  TF.

### Changed

- **`models/train.py`** — `eegnext` added to `MODEL_FACTORIES` and
  `NEURAL_HYBRID_MODELS`; the two neural-model branch checks now key off
  `NEURAL_HYBRID_MODELS` instead of a hardcoded `{"cnn", "eegnet"}` set so future
  hybrid models slot in automatically.
- **`models/normalization.py`** — routes `eegnext` to its fold-local
  exponential-moving standardizer.
- **`run.py`, `scripts/04_train.py`, `scripts/07_feature_informativeness.py`,
  `scripts/08_tensor_model_diagnostics.py`** — `eegnext` added to the
  `SPEED_TIERS` maps and the tensor / full-CNV model sets.
- **`scripts/06_compare_runs.py`** — `eegnext` added to the screening
  diagnostics: the `--default-tier` choices, the single-tier/tensor-model
  classification sets, and the run-name model/tier inference fallbacks
  (`eegnext` ordered before `eegnet` so the more specific token wins). `eegnext`
  now has full parity with `cnn`/`eegnet` across the performance recorders and
  diagnostic tools (per-run metrics, screening, occlusion).

### Tests

- **`tests/test_imports.py`** — added `eegnet`/`eegnext`/`lstm` to the import
  smoke list (previously only `cnn` was covered) and a
  `test_eegnext_has_full_recorder_and_diagnostic_parity` guard that locks the
  registry, forced-full-channel, normalizer, and diagnostic-script wiring so the
  parity cannot silently regress.

## 2026-05-29 — Shape-decomposition features + stability selection

Two changes targeting (a) information lost when amplitude time courses are
collapsed to per-bin means, and (b) the data-starved 5× 2-fold RFECV at the
small per-participant trial counts.

### Added

- **`features/basis.py`** — shape-decomposition (basis-expansion) features that
  describe each channel's time course by a few coefficients instead of per-bin
  means:
  - `polynomial_basis_features` — orthogonal Legendre/Chebyshev coefficients
    (per-epoch, leakage-free). c0 = level, c1 = CNV-ramp slope, c2 = curvature.
  - `bspline_basis_features` — clamped least-squares B-spline coefficients
    (per-epoch, leakage-free) for localized deflections.
  - `FunctionalPCABasis` — data-driven functional PCA over per-channel amplitude
    bins, implemented as a scikit-learn transformer so it is fit on the training
    fold only (leakage-safe), wired into `models.train` via
    `modeling.feature_selection.fpca`.
  - Opt-in via `features.blocks: [..., basis]`; configured under `features.basis`.
- **`feature_selection.stability_select`** — complementary-pairs stability
  selection (Shah & Samworth 2013) with an elastic-net logistic base. Robust at
  small trial counts, model-agnostic (logistic/svm/xgb), and carries a
  false-discovery bound. Now the **default** in-fold selector
  (`modeling.feature_selection.method: stability`).
- **`tests/test_basis_features.py`, `tests/test_stability_select.py`** — unit
  tests for the basis math and the selector (synthetic data, no MNE needed).

### Changed

- `models/train.py` step 3 now dispatches on `modeling.feature_selection.method`
  (`stability` | `rfecv` | `none`) and applies the optional in-fold fPCA before
  selection. Stability selection replaces iterated RFECV as the default; RFECV
  is retained for comparison runs. ROI channel parsing recognises the new
  `poly_/bspl_/fpca_` columns.
- `configs/default.yaml`, `configs/smoke.yaml` — added `features.basis` and
  `modeling.feature_selection` stanzas; RFECV marked legacy.

### Notes

- Shape decomposition and stability selection compose: orthogonal/fPCA features
  are uncorrelated, which is exactly what makes the elastic-net selector's
  selection frequencies stable.

## 2026-05-01 — Pipeline reorganization

Moved from a folder of stand-alone scripts to a config-driven, installable
package. Old code stays on disk for reference but is gitignored.

### Added

- **`src/eeg_steptype/`** — installable package (`pip install -e .`):
  - `preprocessing/` — automated raw → epoch pipeline using PyPREP
    (bad-channel detection), `mne-icalabel` (conservative ICA component
    classification at p > 0.9), and `autoreject` (per-channel rejection
    thresholds). Replaces the per-participant scripts at
    `bad_interpolated/Pxx/Pxx_CNV.py`.
  - `source_localization/` — eLORETA pipeline. Hoists `noise_cov`,
    `forward`, and `inverse_operator` out of the per-epoch loop (they
    were rebuilt for every epoch in the old `SRC_writer.py`); caches
    `forward` per participant.
  - `features/` — amplitude, slopes, PSD (Morlet) extraction. Caches the
    wide feature matrix to parquet so model runs no longer re-read `.fif`.
  - `models/` — feature selection (correlation drop / SelectKBest /
    iterated RFECV / gain prune / SHAP prune) and classifier factories
    for XGBoost, SVM, LSTM, and logistic regression. The shared
    per-participant fit/eval driver in `train.py` replaces the duplicated
    inline loops in `CNV_XGB_4.3.py`, `CNV_LSTM_3.py`, `CNV_ML_SVM_1.py`.
- **`configs/`** — single source of truth for all paths and hyper-parameters:
  - `default.yaml` — committed defaults.
  - `local.yaml.example` — template for per-machine path overrides.
  - `smoke.yaml` — tiny end-to-end check (1 participant, logistic
    regression, shrunk grids).
  - `overrides/Pxx.yaml` × 34 — per-participant tweaks. Manual cuts and
    appends from each original `Pxx_CNV.py` are preserved declaratively
    (e.g. P02 multi-file concat, P08 two-window crop, P14/P19/P23 single
    crop, P37 cut+concat with B17/B22 electrode swap, P03 extended ICA
    training window). Lab-flagged bad channels and the legacy hand-tuned
    ICA-exclude lists / rejection thresholds are also captured (legacy
    values commented for fallback).
- **`scripts/01_preprocess.py`...`05_visualize.py`** — thin per-stage CLIs.
- **`run.py`** — single-process driver: `python run.py --stages …`.
- **`Makefile`** — `make install / smoke / test / preprocess / src /
  features / train MODEL=xgb`.
- **`tests/`** — `test_imports.py` (every module imports + override
  spot-checks) and `test_smoke_pipeline.py` (synthetic-data end-to-end
  run in <60 s).
- **`pyproject.toml`** — installable package metadata.
- **`REORG_PROPOSAL.md`** — design doc this layout was built from.

### Changed

- `requirements.txt` — added `pyprep`, `mne-icalabel`, `autoreject`,
  `pyyaml`, `pyarrow`, `scikeras`.
- `README.md` — rewritten around the new layout, quick-start, and
  reproducibility model.
- `.gitignore` — gitignores legacy folders (`01_preprocessing/`,
  `02_models/{archive,lstm,svm,xgboost}/`, `03_visualization/python/`,
  `sandbox/`, `_repo_export/`) and new pipeline data
  (`data/interim/`, `data/features/`, `data/src/`, `outputs/runs/`).
  Per-machine `configs/local.yaml` is gitignored; `local.yaml.example`
  is committed.

### Preserved

- All R-side code at `02_models/R/` and `03_visualization/R/` is
  untouched and still tracked.
- Original per-participant preprocessing scripts under
  `bad_interpolated/Pxx/Pxx_CNV.py` are unchanged in their lab folder
  (outside this repo).

### Behavioral notes

- ICA component selection is now automated (ICLabel @ p > 0.9, conservative).
  Each override YAML keeps the original hand-picked exclude list as a
  commented fallback in case the auto-classifier under-flags a participant.
- Epoch rejection is now `autoreject` by default. The original per-condition
  voltage thresholds (e.g. `One: 48e-6`, `Two: 51.5e-6` for P25) are kept
  as commented fallbacks per participant.
- Final filter bandpass default changed to `[0.1, 40]` Hz (the modal value
  across the cohort). Participants whose original script used a different
  bandpass have it set explicitly in their override (P05/P08/P10/P11/P12/
  P16/P17/P21/P24/P25/P28/P29/P30/P31/P37).
- Every training run writes a stamped folder under `outputs/runs/<id>/`
  containing the full config snapshot, git SHA, and metrics — any past
  result can be reproduced from those three files.
