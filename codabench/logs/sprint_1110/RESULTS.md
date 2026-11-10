# Midweek sprint 11-10 — results (running)

Brief `prompts/2026-11-09_midweek_explore.md`. Started Mon 2026-11-09 17:15 EST.

## W1 desk study — sealed board (GET 2026-11-09 ~17:35, `logs/lb_30483_1109_all.json`, git-ignored)

- 166 owners, one entry each (best shown). Top 0.838, #12 0.806, q75 0.784, **median 0.723**, q25 0.639.
- Ours: X4f all 0.713 = **rank 94 / 166** (was 51/51 on 11-06; the board filled up, 0.713 sits just below median).
- Gap to top 12.5 pts; to the median 1.0. Slot-2 local gain (+1.7 lopo / +2.8 later) would put us ≈ 0.73, near the median.
- Entry #11 is named "LDA" at 0.808: simple models reach the top tier, so the gap is unlikely to be a net-family gap. To re-check after slot 2 (Δsealed vs Δlocal).

## W3 aligned Shallow, wider grid {0.3, 0.4, 0.5, 0.6} (`cmp_align_wide.txt`, ~2 min)

| arm | nested lopo | vs X4f | vs slot-2 ens | later vs slot-2 ens | rule |
|---|---|---|---|---|---|
| aligned Shallow, w 0.5 (10/10) | 0.6612 | +3.37 [+1.34, +5.23] | +1.68 [+0.02, +3.28] (9/10) | −0.16 | **not adopted** (later < 0) |

w is now interior (0.5, not the grid edge). Two of three bars passed; later is a hair negative again.

## W2 families (running; launched 17:21, `scripts/sprint1110_families.sh`, `fam_*.log`, `STATUS.md`)

Smoke (2 epochs lopo0, 3 jobs × 4 threads): Deep4 4.1 s/epoch, FBCNet 9.9, ATCNet 14.3.
ETA (60 ep × 11 splits × 5 seeds): Deep4 ≈ 4 h (~21:30), FBCNet ≈ 9 h (~02:30), ATCNet ≈ 13 h (~06:30 Tue).

### W2a Deep4 — done 20:05 (est ≈ 3.2–4 h from 17:21 → 2 h 44 min, under) — **no gain**

| arm (5 seeds, w grid 0.1–0.5 nested) | nested lopo | vs X4f | vs slot-2 ens | later vs slot-2 ens | rule |
|---|---|---|---|---|---|
| X4f + Deep4 (w 0.1, grid floor) | 0.6263 | −0.12 [−0.61, +0.41] | −1.81 [−2.43, −1.09] | −2.25 | not adopted |
| X4f + (Shallow+Deep4 averaged) | 0.6329 | +0.54 [−0.05, +1.40] | −1.15 [−2.14, −0.04] | −1.21 | not adopted |

Deep4 adds nothing to X4f and dilutes Shallow when mixed in. Files `cmp_deep4.txt`, `cmp_shallow_deep4.txt`.
FBCNet 27/55, ATCNet 13/55 at 20:05.
