# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.
WSL gotcha: `wsl.exe bash -lc '…'` expands `$VAR` and `~` in an outer shell — put any command
with variables in a script under `scripts/sprint1112_*.sh` and run that.

## Needs you

- **Session stopped 17:28 Tue (usage limit).** Restart with the HANDOFF start line; read this file first.
  Jobs left running in WSL (they finish on their own): lanes 1–3 (fbcnet aligned, shallow plain,
  shallow aligned 38–42, ETA ≈ 21:00) and `sprint1112_b2.sh` (B2 gates + bake → `b2_cand_all/`).

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`; holds while a job / wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep 0 (never) on AC and DC. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free.

## Now (updated 16:52)

- **Base B2 (adopted 16:40):** B1 + Q5 in-stream self-training of X4f's logistic layer
  (train + stream's past windows pseudo-labelled by B1; q 0.7, weight 5, from window 40,
  every 20; w 0.6). Nested over 7 Q5 configs: lopo 0.6785, **+1.70 [+1.05, +2.76] vs B1**
  (10/10), later +2.31. vs slot 2 ≈ +3.4 lopo. Every-40 form: +1.25 vs B1 (nested over 3).
  Probe outputs `q5/<name>_<split>.npz`; nested tool `sprint1112_q5_nested.py`.
- **B1** = X4f + aligned Shallow ×5, AdaBN p32, w 0.5 (zip `ab_cand_all`, `73e38638`, ALL gates PASS).
- **Thursday candidate:** a B2 zip (`st20_cand_all` every 20, w 0.6 / `st40_cand_all`
  every 40, w 0.5; both with a 1,500 s refit budget) — baking now; pick by quiet timing.
  Fallbacks: B1 zip, then the safety net `al_cand_all`.
- **Running:**
  - lane 1 fbcnet aligned 33–37 → `rows_fbc_al/` (ETA ≈ 21:00)
  - lane 2 shallow plain 33–37 (+ all) → `rows_pl/` (ETA ≈ 18:30)
  - lane 3 shallow aligned 38–42 (+ all) → `rows_al2/` (ETA ≈ 18:30)
  - bakes st20 / st40 (p1b, ≈ 17:15); q5_pa (adapted pseudo-label source); Q5-net ×2
- **On resume:** (1) read `st_gates_b2.log` + `b2_cand_all/{bake.log,gate_iii.log,gate_vi.log,cli/}`;
  if ALL PASS (gate 4 only valid quiet: re-time with `sprint1112_quiet_timing.sh b2_cand_all`, must be ≤ 60 ms),
  the B2 zip is the Thursday candidate → SUBMISSIONS row + DIRECTIONS top. Else fall back to `st20_cand_all`
  (`914f97de`), then B1 `ab_cand_all` (`73e38638`).
  (2) q5_pw1 / q5_pw03 gave output IDENTICAL to the unweighted run (+2.00) — suspected no-op bug in
  the `--pw` path of `sprint1112_q5.py`; check before trusting.
  (3) when lanes land: AdaBN on `rows_pl` / `rows_al2` / `rows_fbc_al` (`sprint1112_adabn_gen.py`), then
  `sprint1112_combo.py --x4f q5_q0.85_l5.0_s20_t20_f16 --arm …` vs B2.

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
| 15:59 | Q5a self-training ADOPT vs B1 (+1.02); Tent/IM harmful; Sinkhorn, flagged balance no gain |
| 16:28 | Q5 deeper: q 0.7 / w 5 / every 20 +2.04; x4f all-trained training features verified (2e-14) |
| 16:40 | nested over 7 Q5 configs +1.70 → **base B2**; 16:47 self-training solver gates ALL PASS |
| 16:50 | bakes st20 / st40 relaunched (CRLF fix in p1b.sh); q5_pa and Q5-net probes started |
| 17:20 | quiet timing: B1 23.0, st40 33.6, st20 43.1 ms/window → every 20 OK; nested over 11 Q5 configs +1.96 → B2 config q 0.85 / w 5 / s20 / t20, w 0.6 |
| 17:28 | session stopped (usage limit); B2 gates + bake running |
