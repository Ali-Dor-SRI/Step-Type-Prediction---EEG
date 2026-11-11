# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.
WSL gotcha: `wsl.exe bash -lc '…'` expands `$VAR` and `~` in an outer shell — put any command
with variables in a script under `scripts/sprint1112_*.sh` and run that.

## Needs you

- (none) — session resumed 19:31 after a usage-limit pause (17:28–19:31); all jobs survived.

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`; holds while a job / wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep 0 (never) on AC and DC. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free.

## Now (updated 19:40)

- **Thursday candidate = B2 zip `b2_cand_all` (`1ae19a3b…`), ALL gates PASS** (quiet 50.1 ms/window).
  X4f + 5 aligned Shallow (AdaBN p32), w 0.6, self-training q 0.85 / weight 5 / start 20 / every 20.
  Nested: +1.96 [+1.09, +3.22] vs B1, ≈ +3.7 lopo vs slot 2; later ≈ 0.729. SUBMISSIONS row written.
  Fallbacks: B1 `ab_cand_all` (`73e38638`), safety net `al_cand_all` (`e93586d6`).
- **Running:** fbcnet aligned 33–37 (`rows_fbc_al`, s35 of 5, ETA ≈ 22:00); filler aligned Shallow
  43–47 (`rows_al3`) and plain 38–42 (`rows_pl2`), started 19:38, ETA ≈ 22:30; `nets2` (AdaBN on
  rows_pl / rows_al2 + stacks on B2, `nets2.out`).
- **Next:** (1) nets2 results → if a net stack is adopt/lean on B2, rebake B2 with more nets (the
  solver takes any K aligned nets; plain nets need a second net group — solver work);
  (2) FBCNet aligned + AdaBN when it lands; (3) § 6 items: tighten the leader (per-run-type
  breakdown of B2, nested check done), Q2 change-point re-centring, Q3 split-window.
- **WSL gotchas:** `$VAR`/`~` inside `wsl.exe bash -lc '…'` get expanded by an outer shell → use
  script files; `nohup … &` inside wsl.exe dies → use run_in_background; Python on Windows writes CRLF →
  `sed -i 's/$//'` on edited .sh files.

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
| 17:37 | B2 zip `1ae19a3b` baked; gates (i) ALL PASS |
| 19:31 | resumed; B2 quiet timing 50.1 ms PASS → Thursday candidate; lanes 2/3 done; pw lever void (features session-centred) |
| 19:38 | filler lanes rows_al3 / rows_pl2 started; nets2 (AdaBN on new nets) running |
