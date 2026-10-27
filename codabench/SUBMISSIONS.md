# Submissions ledger

One row per Codabench upload (and per notable local run worth comparing).
Warm-up scores are indicative only: public test data, leakage possible.

## Codabench uploads

| Date | Track / phase | Solver + key params | ZIP | Local bal. acc. | Codabench score | Notes |
|---|---|---|---|---|---|---|
| *ready* | Track 2 / warm-up | **EEGNet-StepType-WU1** (`solvers/bci_decoding/eegnet_steptype_wu1.py`): 100 epochs, patience 20, ref none, seed 33 | `submissions/eegnet_steptype_wu1_2026-09-24.zip` | 0.820 (config mean 0.806 ± 0.016, 3 seeds) | 0.82 (public leaderboard, submitted 2026-09-25 14:51 UTC as adoroodchi; presumably WU1, confirm) | replay-verified (read-only, inference-only, identical score) |
| 2026-10-02 10:32 (as listed on Codabench) | Track 2 / warm-up | **Riemann-Sealed deployment test** (`logs/sealed_s1001/deploytest/riemann_sealed_cand.py`: the solver as of 71607ea with `xblocks="bpt4"` baked, other parameters default) | `logs/sealed_s1001/deploytest/riemann_sealed_deploytest_dreyer2023_2026-10-01.zip` (4.1 MiB, sha256 `0cf3c91b…2e1c17`) | 0.615873 (train = read-only replay) | **0.62** (submission #45, id 956971, adoroodchi, queue "Neural Interface 2026 - AWS", 58.68 s) | **Finished: the sealed model (sklearn/pyriemann joblib) runs on the scoring image.** A deployment test, not a leaderboard attempt. Only log message: benchopt's harmless `wcv_ref=all` literal-value warning (also in every local run) |
| 2026-10-26 (scored; downloaded 10-27) | Track 2 / warm-up (PROTEUS) | **Wrong zip: the Oct-1 Dreyer deployment test re-uploaded** (meant to be the F64 official zip) | `riemann_sealed_deploytest_dreyer2023_2026-10-01.zip` (sha256 `0cf3c91b…2e1c17`, checked on the user's download) | — (trained on Dreyer: 2 classes, 27 ch) | **0.328352** cell mean / 0.328313 bal. acc. / 0.328477 accuracy (= 248/755), **1,081.6 s** | Chance on 3 classes, as expected for a Dreyer model; says nothing about F64. Runtime 1.43 s/window over 755 streamed windows (this old solver runs a Welch-PSD router per window); F64's own warm-up time is still unmeasured on the server |
| *ready, not uploaded* (2026-10-15) | Track 2 / warm-up (PROTEUS) | **Riemann-Sealed-Proteus-Pooled** (`solvers/bci_decoding/riemann_sealed_proteus_pooled.py`: `riemann_sealed.py` as of the 10-15 `reset_state()` edit with `personal="pooled"`, `align="none"` baked; `adapt="none"` and everything else default). Trained on the organisers' dataset `BCI[study=dreyer2026proteus,subset=full]`, `official` split, seed 42, through benchopt exactly as the server runs it | `logs/proteus_setup/cand_official/riemann_sealed_proteus_pooled_official_2026-10-15.zip` (562,449 B = 0.54 MiB, sha256 `cd9f6a193a3286e73368c811284d00bdfe1f6987e6b09b5616e24f650458c742`; `submission.py` + `riemann_sealed.joblib` at the zip root) | cell mean **0.406751**, pooled bal. acc. 0.405717, accuracy 0.405298 (8 cells; train = read-only replay = unzipped replay = route B replay = 2nd training run; for information only: an untuned plumbing vehicle, no path for unseen people) | — | If uploaded, the server's three numbers (cell mean, pooled balanced accuracy, accuracy) must read 0.406751 / 0.405717 / 0.405298; locally 8.9 ms/window, 755 windows in ~7 s. **The scoring image changed since our last server test (`v2` → `v3`)**: a finished run also re-proves that a pyriemann 0.12 / scikit-learn 1.9.1 joblib loads there |
| *ready, not uploaded* (2026-10-15) | Track 2 / warm-up (PROTEUS) | **MeanLogReg** (the organisers' baseline, `$COMPET_ROOT/tracks/bci_decoding/solvers/mean_logreg.py`), the Phase 1 reference export: `benchopt run --config tracks/bci_decoding/training.yml -s MeanLogReg` (seed 42, official loader) | `logs/proteus_setup/ref_meanlogreg/meanlogreg_official_2026-10-15.zip` (3,022 B, sha256 `5b31484b87541c826589a6803ec3615b0930c5ebf6996db805181b50086d6583`; `submission.py` + `weights.joblib` at the zip root) | cell mean **0.395863**, pooled 0.395672, accuracy 0.396026 (8 cells; train = unzipped replay) | — | Expected on the server: 0.395863 / 0.395672 / 0.396026. A format check of the new streamed contract on image `v3`, nothing more |
| *superseded by proteus_f64 (2026-10-16 09:40)* | Track 2 / warm-up (PROTEUS) | **Riemann-Sealed-proteus_s64v2, official-trained** (`solvers/bci_decoding/riemann_sealed_proteus_s64v2.py`: `riemann_sealed.py` as of e17d403 with `personal="pooled"`, `align="stream"`, `adapt="online"`, `buffer=64`, `xblocks="bpt4"` baked; sprint 2026-10-16 candidate S64). Trained on `BCI[study=dreyer2026proteus,subset=full]`, `official` split, seed 42 | `logs/proteus_w1016/proteus_s64v2_cand_official/riemann_sealed_proteus_s64v2_official_2026-10-16.zip` (668,526 B, sha256 `e7cb48ea904883afa1afa317083885a84a66cca25df8cf22197f2e19e765996d`) | cell mean **0.462192**, pooled 0.461790, accuracy 0.460927 (8 cells; train = replay = unzipped = route B = 2nd training; information only: P04 / P09 labels are public) | — | **The warm-up test of the sealed candidate's format** (online re-centring inside a stream, `reset_state()` per stream). Expected on the server: 0.462192 / 0.461790 / 0.460927. Locally 8.2 ms/window on one quiet thread |
| *superseded by proteus_f64 (2026-10-16 09:40)* | Track 2 / sealed candidate (S64) | **Riemann-Sealed-proteus_s64v2, trained on `all`** (2,655 windows, 10 people, 14 sessions; same file and settings as the row above) | `logs/proteus_w1016/proteus_s64v2_cand_all/riemann_sealed_proteus_s64v2_all_2026-10-16.zip` (819,017 B, sha256 `69ff61f01c567d3471453f2e49f79cbeee40be62dadf51a9985b4def1a13550f`) | lopo 0.4839 / later 0.5302 cell means for this recipe (sprint 2026-10-16 Phase 1; the zip's own official score 0.990 is meaningless: test inside training) | — | Train = read-only replay = unzipped replay = route B (EQUAL); stateful streaming contract PASS on the export and the unzipped folder; sealed test ≈ 3 min (8.2 ms/window). **For the first sealed upload (Oct 28)**, after the warm-up row above has proven the format on the server |
| *ready, NOT uploaded* (the 10-26 upload was the Dreyer zip by mistake, see the row above; warm-up open until Nov 1) | Track 2 / warm-up (PROTEUS) | **Riemann-Sealed-proteus_f64, official-trained** (`solvers/bci_decoding/riemann_sealed_proteus_f64.py`: `riemann_sealed.py` as of b989a06 with `personal="pooled"`, `align="stream"`, `adapt="online"`, `buffer=64`, `xblocks="bpt4"`, `fcenter=True` baked; sprint 2026-10-16 candidate F64). Trained on `BCI[study=dreyer2026proteus,subset=full]`, `official` split, seed 42 | `logs/proteus_w1016/proteus_f64_cand_official/riemann_sealed_proteus_f64_official_2026-10-16.zip` (704,929 B, sha256 `d4b424782baac55dc21c2a4b50e090a5133b46a01080f8a61a32f11771cd0cd3`) | cell mean **0.507944**, pooled 0.506931, accuracy 0.507285 (8 cells; train = replay = unzipped = route B = 2nd training; information only: the warm-up test labels are public) | — | **Upload this one to the warm-up** to prove the sealed candidate's format (streamed, adaptive, `reset_state()` per stream) on image `v3`. Expected on the server: 0.507944 / 0.506931 / 0.507285 |
| *ready, not uploaded* (2026-10-16) | Track 2 / **sealed phase candidate** | **Riemann-Sealed-proteus_f64, trained on `all`** (2,655 windows, 10 people, 14 sessions; same file and settings) | `logs/proteus_w1016/proteus_f64_cand_all/riemann_sealed_proteus_f64_all_2026-10-16.zip` (856,757 B, sha256 `59121c39da5c4ed5bc4d199814645a6b4b68f9c1e70c4fe4644fbea6f78e1822`) | this recipe: lopo **0.5672**, later **0.6005** cell means (sprint 2026-10-16 Phase 6); the zip's own official score 0.988 is meaningless (test inside training) | — | Train = read-only replay = unzipped replay = official loader (EQUAL); stateful streaming contract PASS; 8.2 ms/window on one quiet thread (sealed test ≈ 3 min). **For the first sealed upload (Oct 28)**, after the warm-up row above matches on the server |
| *ready, not uploaded* (2026-10-16) | Track 2 / sealed **variant** (not the candidate) | **Riemann-Sealed-proteus_fi64, trained on `all`** (F64 + `xblocks="bpt4_icoh"`; Phase 7b: +1.65 lopo [+0.43, +2.85], +2.35 later over F64, below rule 3's 2-point bar) | `logs/proteus_w1016/proteus_fi64_cand_all/riemann_sealed_proteus_fi64_all_2026-10-16.zip` (1,034,095 B, sha256 `e58d72f08d95318eb2d2fe22b9bd3d3e58b664135877d691f8673e6c039060e1`); official-trained twin `…/proteus_fi64_cand_official/riemann_sealed_proteus_fi64_official_2026-10-16.zip` (882,199 B, sha256 `ebef0a8ff55b03c358970e915b46779c2bf0ea2ef01f629d669166cabba65937`; expected official 0.503897 / 0.502830 / 0.503311) | this recipe: lopo 0.5838, later 0.6240 | — | All gates as F64 (EQUAL chains, stateful contract, zhou regression); 8.9 ms/window quiet (≈ 3.1 min). For a sealed upload after F64, if the plan keeps a variant slot |

### Deployment test (done 2026-10-02: passed)

Recommended by the final sprint review, 2026-09-29. No sklearn/pyriemann joblib
has ever run on the Codabench scoring image (`tommoral/neural-compet:v2`). The only
upload so far is the torch EEGNet above. The image's scikit-learn is unpinned, and
pyriemann arrives only transitively. Riemann-Sealed ships 22 sklearn LDAs
(pickled with sklearn 1.9.1) plus pyriemann 0.12 objects. The sealed phase allows 1
submission per day and hides the logs, so find out **during warm-up** (open until
2026-10-25, 5/day):
- train Riemann-Sealed with its **default** parameters (what Codabench runs) on
  the warm-up study. Dreyer is cross-subject, so the score does not matter and the
  cross-session harness does not apply; train with benchopt directly:
  `cd ~/codabench/2026-competition && COMPET_SUBMISSION_DIR=$HOME/codabench/logs/wu_riemann_deploytest/submission benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" -s ../solvers/bci_decoding/riemann_sealed.py -o "BCI-decoding[training=True]" --no-plot --no-html --no-cache --output wu_riemann_deploytest`,
  then zip the two files in that submission folder at the zip root (as
  `train_sealed.sh` does);
- upload the zip (the user's action);
- check that it reaches *Finished*, and read the ingestion log and its duration.

**Prepared 2026-10-01 (sprint 1001); uploaded 2026-10-02 as submission 956971:
Finished, 0.62, 58.68 s (row above).**

| | |
|---|---|
| zip | `codabench/logs/sealed_s1001/deploytest/riemann_sealed_deploytest_dreyer2023_2026-10-01.zip` (4.1 MiB, at the zip root: `submission.py` + `riemann_sealed.joblib` 4.6 MB) |
| sha256 | `0cf3c91b57e9cfc0035316a05b83e9bb6c3a9509d0cafce694186873272e1c17` |
| model | Riemann-Sealed (the solver as of commit 71607ea: dual LDA, `wcv_ref`), `xblocks="bpt4"` baked as the sealed candidate will be; other parameters at their defaults (blend_w 0.5, align subject); trained on `BCI[study=dreyer2023]` by `scripts/sprint1001_deploytest.sh` |
| local check | train 0.615873 = read-only replay 0.615873 (inference only, from a read-only copy). Dreyer is cross-subject, so the score means nothing. The public warm-up leaderboard shows 0.82 for WU1: if Codabench displays the latest submission, keep WU1 as the one shown (the warm-up does not rank) |
| joblib holds | numpy arrays, sklearn `LinearDiscriminantAnalysis` (pooled, 52 per-subject, router), pyriemann `XdawnCovariances` / `TangentSpace`. No `covariance_` matrices: the first build was a 2.6 GB joblib (2.4 GB zip) before the solver dropped them at every size |

Checked locally (2026-10-01, `analysis/joblib_compat_check.py`): the joblib
loads and gives identical outputs (≤ 2e-14) under sklearn 1.6.1–1.8.0, numpy
1.26–2.5 and pyriemann 0.7–0.11, with only sklearn's InconsistentVersionWarning.
So the upload mainly tests the ingestion path (benchopt, the solver's imports,
time), not the pickles.

What to check after the upload:
1. It reaches *Finished*, not *Failed*.
2. The ingestion log shows no unpickling warning (sklearn
   `InconsistentVersionWarning`) and no pyriemann import error. A warning
   alone is information. A failure means the sealed candidate needs
   plain-array parameters instead of pickled objects: do this before
   Oct 28.
3. The duration. Locally the replay took 32–37 s for Dreyer's test set.
4. Record the row in the upload table above.

## Local reference runs

| Date | Dataset | Solver | Setting | Bal. acc. | Log |
|---|---|---|---|---|---|
| 2026-09-23 | tangermann2012 (4-class, 22 ch) | Riemann-StepType | full train, defaults | 0.536 | `logs/test_tangermann_1805.log` |
| 2026-09-23 | tangermann2012 | EEGNet-StepType | full train, early stop @ epoch 15 | 0.454 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | EEGNet (upstream braindecode baseline) | 20 epochs | 0.580 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | MeanLogReg (upstream floor) | full train | 0.266 | `logs/test_tangermann_1805.log` |
| 2026-09-23 | dreyer2023 (2-class, 27 ch) | Riemann-StepType | test batch: 40 batches (2,560 windows) | 0.716 | `logs/test_track2_dreyer2023_2026-09-23_1834.log` |
| 2026-09-23 | dreyer2023 | EEGNet-StepType | test batch: 40 batches, 3 epochs | 0.651 | same |
| 2026-09-23 | dreyer2023 | MeanLogReg (upstream floor) | full train split | 0.681 | same |
| 2026-09-24 | **dreyer2023, full train (12,392 windows)** | **EEGNet-StepType** | reference=car, patience=20, 50 epochs; 3 seeds | **0.790 ± 0.002** | `logs/overnight_2026-09-24/` |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | reference=car, patience=10; 3 seeds | 0.786 ± 0.008 | same |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | reference=none, patience=10 / 20; 3 seeds | 0.785 / 0.779 | same |
| 2026-09-24 | dreyer2023 full | EEGNet (upstream braindecode) | 20 epochs; 3 seeds | 0.778 ± 0.013 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN on, reference none / car | 0.754 / 0.746 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN off, reference none / car | 0.592 / 0.585 | same |
| 2026-09-24 | dreyer2023 full | Torch-Linear / MeanLogReg (upstream floors) | — | 0.689 / 0.681 | same |
| 2026-09-24 | dreyer2023 full | **EEGNet-StepType** | **100 epochs, patience 20**, ref none / car; 3 seeds | **0.806 ± 0.016 / 0.803 ± 0.010** | `logs/overnight_2026-09-24_p2/` |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | standardize=True, 50 epochs, ref car / none; 3 seeds | 0.780 / 0.777 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN nfilter 8, ref car / none | 0.760 / 0.759 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN nfilter 4 + 1–40 Hz band-pass / nfilter 2 | 0.71–0.72 / 0.66–0.70 | same |
| 2026-09-25 | dreyer2023 full | EEGNet-StepType | 200 epochs, patience 30, ref none; 3 seeds | 0.810 ± 0.008 | `logs/overnight_2026-09-24_p3/` |
| 2026-10-02 | dreyer2023 full | **EEGNet-StepType-ENS** (5 members, softmax mean), 100 epochs, patience 20, seeds 33-37 | 0.8095 (single run; WU1 0.820) | `logs/warmup_1002_ens/` |
| 2026-09-25 | dreyer2023 full | EEGNet-StepType-WU1 (candidate) | defaults (100 ep, pat 20, none, seed 33); train run / platform replay | 0.82004 / 0.82004 | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on, ref none, blocks off (541 features): baseline re-run | 0.75377 (= phase 1 exactly) | `logs/riemann_blocks_2026-09-25/` |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on + `slow_block` (919 features) | 0.75635 (+0.003) | same |
| 2026-09-25 | dreyer2023 full | **Riemann-StepType** | **xDAWN on + `filterbank`** (2,053 features) | **0.76964 (+0.016)**, best Riemann | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on + `slow_block` + `filterbank` (2,431 features) | 0.76865 (+0.015) | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN **off** + `slow_block` + `filterbank` (2,295 features) | 0.71825 (−0.036; xDAWN off alone was 0.592) | same |

Chance: tangermann2012 = 0.25, dreyer2023 = 0.50. Dreyer test = participants
61–81 (= the Codabench warm-up evaluation subset).
