# RESULTS — Sprint A (Thu Nov 12 → Sat Nov 14 10:00 EST)

Brief `prompts/2026-11-12_sprint_a.md`. Base B3 (probe `q5_q0.85_l5.0_s20_t20_f16_C0.03`), nested over w 0.4–0.7:
lopo 0.6868 / later 0.7346 (the brief's 0.6819 used the 0.3–0.7 grid). Comparator `analysis/sprint1114_vsb3.py`
(paired bootstrap over people, vs B3). Raw lines: `eval.out`.

| id | arm | Δ lopo vs B3 [CI] (up /10) | Δ later | verdict |
|---|---|---|---|---|
| A1c1 | pseudo-label ensemble weight wpl 0.3 | −1.54 [−2.68, −0.76] (0) | −0.38 | no gain |
| A1c2 | wpl 0.7 | −0.19 [−0.94, +0.48] (5) | +0.40 | no gain |
| A1c3 | wpl 1.0 (pure co-training from the nets) | −0.86 [−1.79, −0.33] (0) | −0.11 | no gain |
| A2a | refit every 10 | −0.12 [−0.40, +0.14] (3) | −1.45 | no gain |
| A2b | refit every 5 | −0.07 [−0.40, +0.21] (2) | −1.69 | no gain |
| A1a1 | soft weights λ·conf^1 | −0.15 [−0.48, +0.10] (2) | −0.26 | no gain |
| A1a2 | soft weights λ·conf^2 | −0.08 [−0.41, +0.26] (4) | −0.40 | no gain |
| A1a3 | q 1.0 + conf^2 | −0.30 [−0.92, +0.16] (3) | −0.65 | no gain |
| A1b1 | agreement filter (X4f = nets), q 0.85 | −1.17 [−2.18, −0.50] (1) | −1.93 | no gain |
| A1b2 | agreement filter, q 1.0 | −0.31 [−0.99, +0.25] (4) | +0.13 | no gain |
| A1d | per-stream variance normalisation | −0.25 [−0.64, +0.06] (4) | −0.65 | no gain |
| BAGC | bag of self-trained heads over C 0.003–0.3 | −0.10 [−0.42, +0.26] (5) | −0.40 | no gain |
| BAGQ | bag over q 0.7 / 0.85 / 1.0 | −0.34 [−0.97, +0.23] (2) | −1.19 | no gain |
| BAGROB | bag B3 + Huber + Tyler covariance X4f heads | +0.09 [−0.94, +0.78] (6) | −1.25 | no gain |
| BAGALL | bag of 10 heads | −0.03 [−0.41, +0.36] (3) | −0.66 | no gain |
| A3 | P04 diagnosis (`A3_P04.md`) | low-SNR person: signal / covariance normal, pseudo-labels 0.52, confidence uninformative | — | closed, no arm |

**Reading:** B3's self-training sits on a flat optimum: label weight, confidence weighting, filtering, cadence,
feature scale and bagging all land within ±0.3 of it or below. The self-training line is closed.

**Decision (Opus, Thu 10:55):** by the brief's freeze rule nothing qualifies → **slot 4 held** until B3's sealed
score (gate G1). Also: cadence gains nothing, so the GPU 'refit every window' idea for Sprint B loses its main
rationale (Tent / IM were harmful 11-12) — Sprint B needs a new premise.
