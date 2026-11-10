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

## Desk study (P3, 11-10 15:25–15:30; board via in-app browser, web via a read-only agent)

Nothing public on the top methods (no write-ups, repos, forum posts readable; PROTEUS paper not
published). Facts that matter: the sealed container gets **one A100 GPU, 60 min**; the Scientific
FAQ allows causal label-free adaptation inside a session **"including gradient updates"**;
predict() sees EEG only (no person / session / run / interface); runs arrive in run order.
Board: score vs log server duration r = 0.71; ≥ 0.80 entries run ≈ 1,450 s vs our 550 s.

| hypothesis | evidence | local test | cost |
|---|---|---|---|
| H1 net-side test-time adaptation (AdaBN, Tent, pseudo-label self-training) on aligned nets | FAQ allows gradient updates; board time ∝ score; OTTA-for-MI literature | AdaBN probe on saved aligned nets (no retrain); then Tent | minutes → hours |
| H2 much stronger net side (nets alone 0.54–0.60 vs X4f 0.63): aligned FBCNet (best single family, 0.596), more seeds, longer / augmented training | family results 11-10 were unaligned | aligned FBCNet seeds through netsmoke --align stream | 1 lane, hours |
| H3 interface / run change points (Graz ↔ BrainHero inside one stream) | FAQ: interface withheld, state persists across interfaces | Q2 | 3–5 h |
| H4 time-domain / slow ERP (incl. eyes) features | "LDA" entry at 0.808 earlier | **done 15:29: no gain** (below) | — |
| H5 foundation models (REVE etc.) | allowed, organisers ship REVE | needs weight download → out of scope (no downloads) | — |

## Re-ranked order (after D1 / D3 / D5 + P3, 15:35)

D1: errors are not concentrated by run type (calib 0.651 vs online 0.640 lopo), interface or
stream position (1–35: 0.642, 36–70: 0.657, 71+: 0.640); the spread is by **person** (0.54–0.82).
→ Q1 (position experts) drops to the end. D3: runs are class-balanced (54 / 84 runs exactly
10/10/10, 17 exactly 12/12/12), sequences random, repeat rate 0.29 → QF possible (flagged).

1. **H1 AdaBN** (running now), then Tent / pseudo-label refit on the nets (Q5 net side).
2. **H2 aligned FBCNet / aligned more Shallow seeds** (Q7) — the filler lane (long).
3. Q4 classifier stack on the X4f dumps; Q5 balanced self-training on X4f.
4. Q2 run / interface change-point re-centring.
5. Q3 split-window blocks; Q8 robust covariance; Q6 soft person experts.
6. QF flagged (balance constraint) — measure only.
7. Q1 position experts (D1 says no position effect).

## Pre-registered arm lines

| arm | vs | rule | result |
|---|---|---|---|
| TD1 td_dm_all / td_raw_all / td_dm_front / td_dm_b15: time-domain bin means (0.25 / 0.125 s), running feature centring, LR C 0.01; stacked on slot 2, w nested in {0.1..0.5} | slot 2 | § 3 | no gain ×4 (best raw +0.66 [−1.02, +2.27], later −0.15) |
| AB1 aligned Shallow (5 seeds) with causal per-stream BN statistics, prior 8 / 32; X4f + nets, w nested {0.2..0.6} | slot 2 | § 3 | running |
