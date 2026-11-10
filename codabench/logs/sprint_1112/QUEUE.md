# QUEUE — sprint 11-12 (copy of brief § 4 P4 and § 6; re-ranked after P2 / P3)

Verdict rules (brief § 3), all vs the current base, nested, paired bootstrap over people:
adopt = Δ lopo ≥ +1.0, CI lo > 0, Δ later ≥ 0; lean = Δ lopo > 0, CI lo > −0.5, Δ later ≥ −0.5;
else no gain. Kill: fold-level Δ < −2 after 4 lopo folds. Flagged track separate (label order /
counts / unclear legality).

## Default order (before diagnostics)

| # | lever | build | cost | status |
|---|---|---|---|---|
| Q1 | stream-position experts (calibration vs online expert, hard / soft switch, position feature) | dumps | ≈ 1 h | pending |
| Q2 | interface / run-aware re-centring (change points, per-context references) | sweep | 3–5 h | pending |
| Q3 | split-window blocks (1–3 s and 3–5 s in addition to the full window) | sweep | 3–5 h | pending |
| Q4 | classifier stack on X4f features (RBF-SVM, per-block experts, MDM, GBM, bagging) | dumps | 2–3 h | pending |
| Q5 | stronger in-session self-training (balanced pseudo-labels, last-layer refit) | dumps | 2–3 h | pending |
| Q6 | soft person experts for known people | sweep | 3–4 h | pending |
| Q7 | net-side stack (aligned + unaligned, seeds 38–47, aug, mixup, per-stream BN) | 1 lane | long | filler |
| Q8 | robust covariance / artifacts | sweep | 3–5 h | pending |
| Q9 | frontal slow time course | sweep | 2–4 h | pending |
| QF | flagged track (only if D3 shows label structure) | dumps | 1–2 h | pending |

## § 6 when the queue runs dry

a. combine the two best positive arms, nested weights; b. one level deeper on the best lever;
c. re-read D1, cheapest arm on the largest error cell; d. repeat the desk study; e. tighten the
leader (nested re-selection, 5 more seeds, per-run-type, timing under load); f. revisit a "no gain"
lever only on a new base (state why). No job whose 1.5 × ETA ends after Thu 11:30 (except
checkpointed filler seeds).

## Re-ranked order (after P2 / P3)

(to be written)

## Pre-registered arm lines

(one line per arm before it runs)
