# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.
WSL gotcha: `wsl.exe bash -lc '…'` expands `$VAR` and `~` in an outer shell — put any command
with variables in a script under `scripts/sprint1112_*.sh` and run that.

## Needs you

- (none)

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`; holds while a job / wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep 0 (never) on AC and DC. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free.

## Now (updated 15:58)

- **Base B1 (adopted 15:45):** X4f + 5× stream-aligned Shallow with AdaBN (prior 32), w 0.5;
  nested lopo 0.6596 (+1.52 [+0.16, +2.86] vs slot 2), later 0.7065. Fixed-prior rows:
  p8 0.6641 / 0.7142 (`adabn2.out`). Compare later arms with
  `sprint1109_compare.py <slot-2 rows> <arm dir>` or against `logs/sprint_1112/adabn/al_p32`.
- **Thursday candidate:** `ab_cand_all/…_2026-11-10.zip` sha256 `73e38638…` — ALL gates PASS
  (quiet timing re-run still to do). Fallback: `al_cand_all` (`e93586d6…`, timing gate under
  load only). Both in SUBMISSIONS.md as ready, not uploaded.
- **Running (3 lanes × 4 thr):**
  - fbcnet aligned 33–37 × 11 splits → `rows_fbc_al/` (`scripts/sprint1112_fam_al.sh`), started 15:31, ETA ≈ 21:00
  - shallow plain 33–37 × 12 splits (incl. all) → `rows_pl/` (`sprint1112_nets.sh`), started 15:58, ETA ≈ 18:30
  - shallow aligned 38–42 × 12 splits → `rows_al2/`, started 15:58, ETA ≈ 18:30
- **Next three:** (1) Q5 pseudo-label self-training on X4f features (dumps, ≤ 2 thr);
  (2) AdaBN on plain nets when `rows_pl` lands, and 10-seed aligned AdaBN (33–42);
  (3) generalise AdaBN to FBCNet (every BN layer) when `rows_fbc_al` lands; then Tent.

## Log of state changes

| time | event |
|---|---|
| 15:14 | P0 start; keep-awake; base row OK (0.6444 / 0.7013) |
| 15:16 | P1 lane: 5 aligned nets on all (done 15:32) |
| 15:20 | aligned solver `ens_x4f_shallow_al.py` built; gates (i)/(ii) ALL PASS |
| 15:23 | D1 / D3 / D5 done; 15:43 D4 / D2 done (gap = per-person adaptation) |
| 15:25 | P3 board + desk study (agent): score ∝ log server time; A100 + gradient updates allowed |
| 15:29 | TD arms ×4 no gain; 15:37 Q4 plda / pens no gain |
| 15:36 | **AdaBN on aligned Shallow: ADOPT**; 15:45 nested prior+w selection +1.52 → base B1 |
| 15:39 | safety-net zip baked (`e93586d6`); 15:49 its CLI rc 0 |
| 15:53 | B1 zip baked (`73e38638`), ALL gates PASS; SUBMISSIONS rows |
| 15:58 | lanes 2 + 3 started (plain Shallow with weights; aligned seeds 38–42) |
