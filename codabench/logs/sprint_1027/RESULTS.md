# Sprint 2026-10-27: sealed-phase accuracy over F64T (complete)

Brief: `prompts/2026-10-27_sealed_accuracy.md`. Measures: lopo (10 people left out in
turn, 28 cells) and later (session 1 -> later sessions of P04/P10, 8 cells),
cell-averaged balanced accuracy, chance 0.333. Paired people bootstrap (10,000).
Run 11:00-13:12, paused for the user, resumed 16:19-16:59; bakes 16:19-17:15.

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
| fb6 | 0.5922 | 0.5826 | 0.6018 | 0.6183 / 0.5769 | +2.50 [+0.88, +4.12] (8/10) | 0.6262 | +2.57 [+1.37, +3.78] (2/2) | 173 |
| X1 | 0.5995 | 0.6012 | 0.5977 | 0.6198 / 0.5881 | +3.22 [+1.83, +5.05] (9/10) | 0.6458 | +4.53 [+2.72, +6.34] (2/2) | 192 |
| X2 | 0.5953 | 0.5938 | 0.5968 | 0.6179 / 0.5818 | +2.80 [+1.66, +4.38] (8/10) | 0.6314 | +3.09 [+3.01, +3.18] (2/2) | 72 |
| X3 | 0.5930 | 0.5967 | 0.5892 | 0.6082 / 0.5844 | +2.57 [+0.97, +4.15] (8/10) | 0.6448 | +4.43 [+3.99, +4.87] (2/2) | 67 |
| C2f | 0.6083 | 0.6176 | 0.5991 | 0.6345 / 0.5933 | +4.11 [+3.11, +5.22] (10/10) | 0.6424 | +4.19 [+4.07, +4.32] (2/2) | 77 |
| X4f | 0.6275 | 0.6302 | 0.6248 | 0.6539 / 0.6125 | +6.02 [+4.33, +7.78] (10/10) | 0.6735 | +7.31 [+6.38, +8.24] (2/2) | 69 |

Arms (each = F64 + the change): fscale = per-group/stream feature scaling; logreg =
L2 logistic regression C=0.01 instead of shrinkage LDA; dbpt8 = delta power, 8 bins;
bpt8 = 8 bins instead of 4; slow = 0.1-4 Hz waveform bins; fbd = filter bank + 1-4 Hz;
fb6 = filter bank 1-4, 4-8, 8-13, 13-20, 20-30, 30-45 Hz; FI64 = + icoh (10-16).
X1 = icoh+fscale+logreg; X2 = icoh+logreg; X3 = icoh+fscale; **C2f = fb6+icoh**;
**X4f = fb6+icoh+fscale+logreg**.

## Decisions (pre-registered rules, brief section 4)
- Single arms: fscale, logreg, dbpt8, bpt8, slow **no gain**; fbd lean (+1.09);
  **fb6 adopted** (+2.50 [+0.88, +4.12], 8/10; later +2.57).
- **C2f** (rule-selected: fb6 adopted + icoh lean): +4.11 [+3.11, +5.22], **10/10**;
  later +4.19. Beats fb6 by +1.61 (>= 1.0 -> preferred).
- **X4f** (exploratory: fscale and logreg were picked after seeing them): +6.02
  [+4.33, +7.78], **10/10**; later +7.31. Beats C2f by +1.92 [+0.52, +3.22] (9/10;
  later +3.11) -> preferred by the rule, **pending the nested check**.
- X1 / X2 / X3 pass too but are beaten by C2f / X4f.
- Note: fscale and logreg give nothing alone and ~+1-2 on top of icoh: the combination
  effect is the part most at risk of selection optimism, hence the nested check.

## Zips (all thread-pinned, every gate PASS; see SUBMISSIONS.md for hashes)
`logs/proteus_w1016/proteus_{x1,c2f,x4f}_cand_{official,all}/*.zip`.

## Incident 12:12-12:45
Two 10-thread lanes slowed fits 3-10x (align 30 s -> 486 s): every OpenMP pool thread
spin-waited at ~80 % CPU on the i7-12700K (8 P + 4 E cores). Fix: 4 threads per lane +
`OMP_WAIT_POLICY=PASSIVE`; fits then 50-70 s (3x faster than at 10 threads). Killed
folds re-ran from scratch (resumable keys).
