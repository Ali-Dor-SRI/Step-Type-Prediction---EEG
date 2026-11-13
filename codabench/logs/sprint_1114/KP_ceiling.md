# KP ceiling: oracle person id, `later` split (Sat 11-14 prep; analysis/sprint1114_kp.py, kp.out, kp_table.csv)
Data: 1893 train windows (session 1 of 10 people, ~190 each), 762 test windows, 8 cells (P04, P10: 2 sessions x 2 interfaces).
Base B3 (w 0.6): P04 0.6142 / P10 0.8549 / 8-cell 0.7346 (reproduced; k=1 self-train = B3 to 5e-6). B1 (X4f+nets) 0.7092.
Scores = Delta points vs B3 (P04 / P10 / 8-cell mean). Full grid in kp_table.csv.

| arm | config | d P04 | d P10 | d 8-cell |
|---|---|---|---|---|
| P1 personal LR (session 1 of that person) blended v with B3 | C0.01 v0.4 | +2.85 | +1.85 | +2.35 |
| | C0.03 v0.4 | +3.90 | +2.37 | +3.13 |
| | C0.1 v0.4 | +5.47 | +2.11 | +3.79 |
| | C0.01 v0.6 | +7.30 | +1.10 | +4.20 (best) |
| | C0.003 v0.2 (weakest) | -0.29 | +1.03 | +0.37 |
| P1 head alone (no B3) | C0.03 | +1.15 | -7.78 | -3.32 |
| P2 self-train, person rows xk | k3 / k10 / k30 | -0.52 / -0.82 / -0.01 | -0.25 / -0.22 / 0.00 | -0.38 / -0.52 / -0.00 |
| P3 X4f refit xk, no self-train, + nets | k3 / k10 / k30 | +0.78 / +2.37 / +2.37 | -2.26 / -2.01 / -1.76 | -0.74 / +0.18 / +0.30 |

Picked on the other person (config chosen by the other person's delta): P1 +2.25 (P04 +3.90 via C0.03 v0.4, P10 +0.60 via C0.1 v0.6);
P2 0.00; P3 +0.30 (P04 +2.37, P10 -1.76).
Controls (same blend, v 0.4 / 0.6): head on ALL training rows (pooled) -1.4 to -3.6 8-cell; head of the WRONG person -5 to -15.
So the P1 gain is person-specific, not an ensembling effect.

Reading:
- Ceiling with a oracle id and ONE session of that person: about +2 to +4 (8-cell), P1 only; P2/P3 (reweighting the pooled
  linear head) give ~0. Gain is P04-dominated (+3 to +7, the low-SNR person whose pseudo-labels are poor: A3_P04.md);
  P10 (already 0.855) gains only +1 to +2.4 and stays small whatever the config.
- Realistic: v is picked blind; best-on-other-person gives +2.25 (8 cells, 2 people, so wide uncertainty: a bootstrap over 2 people
  is meaningless). Applies only to the known-person half of sealed (needs a router, accuracy untested here), so expect
  at most about half: roughly +1 to +2 overall, and the 10-16 router experiment lost (below).
- Caveat: v 0.6 is a large weight on a 190-window head; its stand-alone score is poor (-3 to -8 for P10), so the blend
  works through complementarity; session 1 -> sessions 2-3 drift is the same as in sealed (later sessions).

10-16 "router + personal LDA" (LOG.md, 2026-10-16 sprint, Phase 1, arm R; logs/proteus_w1016/RESULTS_p1.md):
- Base B1 = pooled LDA + bpt4 (Riemann/filter bank, early model, lopo 0.4268 / later 0.5279), arm R = PSD router assigns a person per
  window, personal LDA blended w 0.5 with the pooled model.
- Result: lopo 0.3775 (-4.93 [-8.66, -1.17], 4/10 people up), later 0.4420 (-8.60 [-13.70, -3.49]): lost on both,
  also for the known people of `later`; rule 4 -> not adopted (router ids, not oracle ids; a different, weaker base).
- Not directly comparable to the present numbers: router errors and the personal-LDA features were part of that loss.
