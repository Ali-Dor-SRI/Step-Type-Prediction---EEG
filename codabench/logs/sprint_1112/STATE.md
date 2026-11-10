# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.

## Needs you

- (none yet)

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`: holds while a job or wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep timeout 0 (never) on AC and DC; hibernate 0. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free; nothing running at start.
- Base row reproduced from files (15:20): slot-2 ens nested **lopo 0.6444 / later 0.7013**;
  aligned-Shallow w 0.5 arm 0.6612 / 0.6997 (+1.68 [+0.02, +3.28], later −0.16). OK.

## Now

- **Base:** slot-2 ens (X4f + 5× Shallow, w 0.3), lopo 0.6444 / later 0.7013.
- **Thursday candidate:** P1 safety net (aligned Shallow w 0.5) — being baked.
- **Running:** lane 1 = P1 aligned nets on `all` (`scripts/sprint1112_p1.sh`, log
  `p1_align_all.log`, ETA 15:35).
- **Next:** P1 bake + gates iii–vi; P2 diagnostics (`analysis/sprint1112_diag.py`); P3 desk study.

## Log of state changes

| time | event |
|---|---|
| 15:14 | P0 start; keep-awake; base row OK |
| 15:16 | P1 lane started (5 aligned nets on all) |
| 15:28 | aligned solver `ens_x4f_shallow_al.py` built; gates (i)/(ii) ALL PASS on lopo0 + later (rows_align, w 0.5); 54 ms/window under load (to re-measure quiet) |
