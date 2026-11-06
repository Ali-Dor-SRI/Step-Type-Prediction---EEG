# DRAFT brief — mid-week exploration sprint, Tue Nov 10 → Fri Nov 13 2026

Status: **draft** (written 2026-11-06 18:30 at the end of the weekend sprint). Review the
"Context" section against the slot-2 sealed score before running.

## Context (verified 2026-11-06)

- Sealed slot 1: X4f all `60a0148c` = 0.713, rank 51 / 51 (top 0.828). Final rank = best of
  the last five sealed uploads; slots left after slot 2: 3 (≈ Sat Nov 14, Tue Nov 17, Thu Nov 19).
- Slot 2 (Tue Nov 10, user): X4f + 5× ShallowNet ensemble `1f6589b8`, local nested lopo
  0.6444 (+1.69 [+0.94, +2.38] vs X4f) / later 0.7013 (+2.78). Results `logs/sprint_1109/RESULTS.md`.
- Already measured, do not repeat: 120-epoch Shallow (+0.40 in the ensemble), stream-aligned
  Shallow with grid {0.2, 0.3, 0.4} (+1.11, later −0.26; w → 0.4 in 10/10), EEGNet (no gain
  alone or in the ensemble), offset augmentation, label-free TTA, train-as-streamed, feature
  blocks roi / asym / burst.
- Tools: `analysis/sprint1106_netsmoke.py` (nets, `--save-weights --align --outdir`),
  `sprint1109_compare.py` (nested arm vs base), `sprint1109_build_ens.py` / `_bake.py` /
  `_w1_gates.py` / `_nobd.py` (any new net must be re-written in plain torch, W0-gated).

## Goal and priority

1. **Desk study of the sealed board** (read-only GET
   `https://www.codabench.org/api/phases/30483/get_leaderboard/`): score spread, the gap to
   the top, how slot 2 moved (Δsealed vs Δlocal +1.7 / +2.8). ≤ 1 h.
2. **Net families judged by their ensemble with X4f** (not alone): FBCNet, Deep4, ATCNet
   (braindecode, 5 seeds, 60 epochs, 4 threads); each vs the Shallow ensemble with
   `sprint1109_compare.py`.
3. **Aligned Shallow with a wider grid** {0.3, 0.4, 0.5, 0.6} (arm (b) sat at the grid edge).
4. **Wider stacks:** X4f + Shallow + best new family, weights chosen nested.

## Rules (pre-registered, rules v2)

- A new arm replaces the slot-2 ensemble for slot 3 if, vs the slot-2 ensemble (both nested),
  Δ lopo ≥ +1.0, CI lower > 0, Δ later ≥ 0. Otherwise slot 3 = a different-family draw only
  if it is at least lean vs X4f, with the risk stated.
- Weights and grids chosen nested only; ties → simpler config; never tune on test data.
- No uploads, posts, merges, tags; push to `personal` and `org` after every commit; ≤ 3 lanes
  × 4 threads, `OMP_WAIT_POLICY=PASSIVE`; estimate / time-log / investigate overruns.

## Time budget (estimates)

| item | estimate |
|---|---|
| desk study | 1 h |
| one family × 5 seeds × 11 splits | Shallow ≈ 25–30 min/seed at 4 threads; Deep4 / ATCNet likely 2–4× → 4–8 h per family |
| plain-torch port + W0 gate + bake + gates | ≈ 2 h per adopted family |
| wrap-up | 15 % |

## Deliverable

Slot-3 recommendation (zip + sha256, gated) or "hold" with the reason; RESULTS, LOG,
DIRECTIONS, HANDOFF updated.
