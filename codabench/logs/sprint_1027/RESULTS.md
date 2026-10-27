# Sprint 2026-10-27: sealed-phase accuracy over F64T (PAUSED 13:12 for the user)

Brief: `prompts/2026-10-27_sealed_accuracy.md`. Measures: lopo (10 people left out in
turn, 28 cells) and later (session 1 → later sessions of P04/P10, 8 cells),
cell-averaged balanced accuracy, chance 0.333. Paired people bootstrap (10,000).

Base: **F64**. Cell-averaged balanced accuracy; diff = arm - base (points ×100); CI = paired bootstrap over people (10,000); up = people whose mean cell score rose.

| arm | lopo (28) | Graz | BrainHero | calib / online runs | Δ lopo [95 % CI] (up/10) | later (8) | Δ later [95 % CI] (up/2) | fit s |
|---|---|---|---|---|---|---|---|---|
| F64 | 0.5672 | 0.5573 | 0.5772 | 0.5747 / 0.5626 | — | 0.6005 | — | 70 |
| FI64 | 0.5838 | 0.5908 | 0.5768 | 0.5946 / 0.5777 | +1.65 [+0.43, +2.85] (7/10) | 0.6240 | +2.35 [+1.64, +3.07] (2/2) | 73 |
| fscale | 0.5755 | 0.5677 | 0.5833 | 0.5793 / 0.5729 | +0.82 [-0.18, +1.71] (9/10) | 0.6001 | -0.04 [-0.80, +0.73] (1/2) | 97 |
| logreg | 0.5737 | 0.5692 | 0.5782 | 0.5867 / 0.5657 | +0.64 [-0.07, +1.73] (7/10) | 0.6111 | +1.07 [+0.50, +1.63] (2/2) | 97 |
| dbpt8 | 0.5645 | 0.5632 | 0.5658 | 0.5662 / 0.5633 | -0.27 [-1.44, +1.21] (3/10) | 0.5946 | -0.59 [-1.61, +0.42] (1/2) | 105 |
| bpt8 | 0.5651 | 0.5603 | 0.5700 | 0.5755 / 0.5587 | -0.21 [-0.87, +0.30] (5/10) | 0.6016 | +0.11 [-0.26, +0.48] (1/2) | 107 |
| slow | 0.5688 | 0.5610 | 0.5767 | 0.5801 / 0.5621 | +0.16 [-0.81, +1.18] (5/10) | 0.6124 | +1.19 [+0.00, +2.39] (2/2) | 106 |
| fbd | 0.5781 | 0.5722 | 0.5841 | 0.5963 / 0.5674 | +1.09 [+0.10, +2.14] (7/10) | 0.6096 | +0.91 [+0.25, +1.57] (2/2) | 159 |
| X1 | 0.5995 | 0.6012 | 0.5977 | 0.6198 / 0.5881 | +3.22 [+1.83, +5.05] (9/10) | 0.6458 | +4.53 [+2.72, +6.34] (2/2) | 192 |

FI64 = F64 + icoh (10-16 row, for reference). X1 = F64 + icoh + fscale + logreg0.01.

## Decisions so far (rules in the brief)
- fscale, logreg, dbpt8, bpt8, slow: **no gain** (CI crosses 0 or below +1.0).
- fbd (delta 1–4 Hz added to the filter-bank tangent space): **lean** (+1.09, CI > 0).
- **X1: passes the exploratory rule** (+3.22 [+1.83, +5.05], 9/10 people up; later
  +4.53, both people up). Its parts were chosen after seeing them ("selected after
  seeing parts"), so it needs the nested check before it replaces F64T in a sealed upload.
- Not yet run (paused): fb6 (1 of 11 folds left), X2 = icoh+logreg, X3 = icoh+fscale,
  C2 = fbd+icoh (rule-selected), X4 = C2+fscale+logreg; X1 bake (official + all zips).

## Incident 12:12–12:45
Two 10-thread lanes slowed fits 3–10× (align 30 s → 486 s; X1 fold 0 560 s): every OpenMP
pool thread spin-waited at ~80 % CPU on the i7-12700K (8 P + 4 E cores). Fix: 4 threads
per lane + `OMP_WAIT_POLICY=PASSIVE`; fits then steady 125–210 s, no spikes. The killed
folds were re-run from scratch (resumable keys); no row came from a killed process.

## Resume
`wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1027_resume.sh'` (~2 h): X1 bake then
X3, X4 on one lane; fb6 then X2, C2 on the other.
