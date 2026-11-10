# Midweek sprint 11-10 — results (done Tue 02:20)

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

### W2b FBCNet — done 22:34 (est ~02:30 → 5 h 13 min, under) — **no gain over slot-2 ens**

| arm (5 seeds, w grid 0.1–0.5 nested) | nested lopo | vs X4f | vs slot-2 ens | later vs slot-2 ens | rule |
|---|---|---|---|---|---|
| X4f + FBCNet (w 0.4) | 0.6395 | +1.20 [+0.05, +2.46] | −0.49 [−1.65, +0.85] | −0.40 | not adopted |
| X4f + (Shallow+FBCNet averaged) | 0.6442 | +1.67 [+0.83, +2.46] | −0.02 [−0.49, +0.70] | −0.44 | not adopted |

FBCNet helps X4f (a lean second-family option vs X4f, slightly weaker than Shallow) but adds nothing on top of Shallow.
ATCNet 27/55 at 22:34.

### W2c ATCNet — done 02:14 Tue (est ~06:30 → 8 h 53 min, under) — **no gain**

| arm (5 seeds, w grid 0.1–0.5 nested) | nested lopo | vs X4f | vs slot-2 ens | later vs slot-2 ens | rule |
|---|---|---|---|---|---|
| X4f + ATCNet (w 0.1, grid floor) | 0.6295 | +0.20 [−0.52, +1.07] | −1.50 [−2.75, −0.22] | −1.76 | not adopted |
| X4f + (Shallow+ATCNet averaged) | 0.6355 | +0.80 [+0.23, +1.70] | −0.89 [−1.79, +0.28] | −1.27 | not adopted |

## Verdict (02:20 Tue)

- **W2 families: no gain.** None of FBCNet / Deep4 / ATCNet beats the slot-2 ensemble, alone with X4f or stacked with Shallow (W4 stacks: −0.02 / −1.15 / −0.89). Only FBCNet helps X4f at all (+1.20), less than Shallow.
- **W3 aligned Shallow, w 0.5:** closest arm, +1.68 [+0.02, +3.28] vs slot-2 ens, later −0.16 → fails the later bar by 0.16 points; not adopted.
- **Slot 3: hold** = the slot-2 ensemble stays the best local model. Read slot 2's sealed Δ first (local +1.7 / +2.8 vs X4f). Optional draw for slot 3, if you want one: the aligned-Shallow ensemble (w 0.5, lean vs X4f +3.37; same family, re-centred input) — not baked; ≈ 2 h for bake + gates.
- Brief finished 3½ days early; all pre-registered items ran. Nothing running.
