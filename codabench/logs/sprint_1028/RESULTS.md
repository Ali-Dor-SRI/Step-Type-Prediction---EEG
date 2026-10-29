# Sprint 2026-10-28: task-driven feature blocks on C2f / X4f

Brief: `prompts/2026-10-28_feature_engineering.md` (with addendum 18:40). Measures as
on 10-27: lopo (28 cells) and later (8 cells), cell-averaged balanced accuracy,
paired people bootstrap (10,000). Each arm = base + one block.
Blocks (unwhitened windows, AF3/AF4 excluded): `roi` 48, `asym` 102, `asymabs` 102
(+ asym), `burst` 48.

## Phase 1 — on C2f (primary: decides adoption)

Base: **C2f**. Cell-averaged balanced accuracy; diff = arm - base (points ×100); CI = paired bootstrap over people (10,000); up = people whose mean cell score rose.

| arm | lopo (28) | Graz | BrainHero | calib / online runs | Δ lopo [95 % CI] (up/10) | later (8) | Δ later [95 % CI] (up/2) | fit s |
|---|---|---|---|---|---|---|---|---|
| C2f | 0.6083 | 0.6176 | 0.5991 | 0.6345 / 0.5933 | — | 0.6424 | — | 77 |
| C2f+roi | 0.6092 | 0.6161 | 0.6023 | 0.6277 / 0.5985 | +0.08 [-0.33, +0.60] (5/10) | 0.6357 | -0.67 [-1.63, +0.29] (1/2) | 103 |
| C2f+asym | 0.6095 | 0.6213 | 0.5977 | 0.6283 / 0.5985 | +0.12 [-0.51, +0.95] (8/10) | 0.6528 | +1.03 [+1.02, +1.05] (2/2) | 103 |
| C2f+asymabs | 0.6060 | 0.6168 | 0.5952 | 0.6271 / 0.5938 | -0.23 [-1.05, +0.52] (3/10) | 0.6570 | +1.46 [+1.06, +1.86] (2/2) | 101 |
| C2f+burst | 0.6098 | 0.6161 | 0.6036 | 0.6231 / 0.6019 | +0.15 [-0.31, +0.80] (4/10) | 0.6460 | +0.36 [-0.57, +1.29] (1/2) | 102 |

## Phase 1 — on X4f (secondary: reported)

Base: **X4f**. Cell-averaged balanced accuracy; diff = arm - base (points ×100); CI = paired bootstrap over people (10,000); up = people whose mean cell score rose.

| arm | lopo (28) | Graz | BrainHero | calib / online runs | Δ lopo [95 % CI] (up/10) | later (8) | Δ later [95 % CI] (up/2) | fit s |
|---|---|---|---|---|---|---|---|---|
| X4f | 0.6275 | 0.6302 | 0.6248 | 0.6539 / 0.6125 | — | 0.6735 | — | 69 |
| X4f+roi | 0.6219 | 0.6235 | 0.6203 | 0.6448 / 0.6089 | -0.56 [-1.34, +0.52] (4/10) | 0.6660 | -0.76 [-0.78, -0.73] (0/2) | 100 |
| X4f+asym | 0.6298 | 0.6324 | 0.6272 | 0.6587 / 0.6138 | +0.23 [-0.42, +0.86] (6/10) | 0.6827 | +0.92 [+0.52, +1.31] (2/2) | 102 |
| X4f+asymabs | 0.6279 | 0.6272 | 0.6286 | 0.6502 / 0.6156 | +0.04 [-0.46, +0.70] (5/10) | 0.6839 | +1.04 [+0.77, +1.30] (2/2) | 106 |
| X4f+burst | 0.6181 | 0.6228 | 0.6134 | 0.6375 / 0.6071 | -0.94 [-1.65, -0.12] (3/10) | 0.6735 | -0.00 [-0.77, +0.77] (1/2) | 109 |

## Mechanistic check (per-class recall on lopo, mean over 28 cells; never decisive)

Expected: asym / asymabs -> mi, roi -> sub, burst -> word.

| arm (vs base) | mi recall (Δ) | sub recall (Δ) | word recall (Δ) |
|---|---|---|---|
| C2f+roi (vs C2f) | 0.573 (+0.9) | 0.625 (-0.4) | 0.630 (-0.3) |
| C2f+asym (vs C2f) | 0.575 (+1.1) | 0.623 (-0.6) | 0.631 (-0.2) |
| C2f+asymabs (vs C2f) | 0.574 (+1.0) | 0.614 (-1.5) | 0.631 (-0.2) |
| C2f+burst (vs C2f) | 0.567 (+0.4) | 0.632 (+0.3) | 0.631 (-0.2) |
| X4f+roi (vs X4f) | 0.610 (-1.0) | 0.636 (-0.5) | 0.619 (-0.2) |
| X4f+asym (vs X4f) | 0.629 (+0.9) | 0.645 (+0.3) | 0.616 (-0.5) |
| X4f+asymabs (vs X4f) | 0.627 (+0.7) | 0.635 (-0.6) | 0.621 (+0.0) |
| X4f+burst (vs X4f) | 0.605 (-1.6) | 0.636 (-0.5) | 0.613 (-0.8) |

## Decisions (rules v2, brief § 5)

- **All eight arms: no gain.** Largest lopo change +0.23 (X4f+asym), every CI crosses 0
  except X4f+burst, which is *worse* (−0.94 [−1.65, −0.12]). No block is adopted or lean.
- **Phase 2 skipped** (contingency: nothing lean). Phase 3 runs the nested walk on
  {F64, fb6, C2f, X4f} only, which settles X4f (DIRECTIONS decision 13).
- Mechanism: only `asym` / `asymabs` move the class they target (mi recall +0.7 to
  +1.1), paid for by sub; `roi` does not lift sub, `burst` does not lift word.
- Reading: the existing 10,000 features (six band covariances in tangent space +
  bpt4 + icoh) already carry the regional, lateral and band-power information these
  blocks summarise; per-window relative power and envelope variability add nothing a
  pooled linear model can use across people. `later` rose for most arms (+0.4 to
  +1.5, 2 people) but `later` never decides alone.

## Phase 3 — nested check of the selection path F64 → fb6 → C2f → X4f (rule: addendum 18:40)

Full-lopo cell means: F64 0.5672, fb6 0.5922, C2f 0.6083, X4f 0.6275
Selected on full lopo by the walk: **X4f** (0.6275).
Nested lopo (28 cells): **0.6210** vs selected 0.6275: optimism +0.64 points -> **no over-selection** (flag at > 3). Picks: X4f x5, C2f x5

| outer fold | inner F64 | inner fb6 | inner C2f | inner X4f | chosen | outer score |
|---|---|---|---|---|---|---|
| 0 (P01) | 0.5558 | 0.5798 | 0.5896 | 0.5993 | X4f | 0.6954 |
| 1 (P02) | 0.5558 | 0.5779 | 0.5971 | 0.5971 | C2f | 0.6782 |
| 2 (P03) | 0.5711 | 0.5857 | 0.6160 | 0.6201 | C2f | 0.5598 |
| 3 (P04) | 0.6123 | 0.6444 | 0.6437 | 0.6550 | X4f | 0.5240 |
| 4 (P05) | 0.5706 | 0.5940 | 0.6135 | 0.6222 | C2f | 0.5108 |
| 5 (P06) | 0.5552 | 0.5716 | 0.5974 | 0.6011 | C2f | 0.6685 |
| 6 (P07) | 0.5381 | 0.5684 | 0.5883 | 0.5948 | C2f | 0.7343 |
| 7 (P08) | 0.5571 | 0.5721 | 0.5869 | 0.6021 | X4f | 0.7877 |
| 8 (P09) | 0.5476 | 0.5670 | 0.5913 | 0.6143 | X4f | 0.7144 |
| 9 (P10) | 0.5476 | 0.5674 | 0.5980 | 0.6100 | X4f | 0.5912 |

**Decision.** No over-selection (+0.64 points, flag at > 3): **X4f holds** and replaces C2f
as the recommended sealed model. Inner selection split 5 / 5 between X4f and C2f, so
X4f's edge over C2f is small; the nested lopo (0.6210) still beats C2f's 0.6083.
The new blocks were not in the walk (none passed Phase 1).
