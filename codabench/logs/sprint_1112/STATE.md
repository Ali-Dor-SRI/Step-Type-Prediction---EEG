# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.
WSL gotcha: `wsl.exe bash -lc '…'` expands `$VAR` and `~` in an outer shell — put any command
with variables in a script under `scripts/sprint1112_*.sh` and run that.

## Needs you

- (none) — resumed Wed 00:31 after the second usage-limit pause (20:11–00:31); jobs survived.

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`; holds while a job / wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep 0 (never) on AC and DC. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free.

## Now (updated Wed 06:05)

- **Thursday candidate = B3** `submissions/ens_x4f_shallow_al_stC_all_2026-11-10.zip`, sha256 `14541a30…`, ALL gates
  PASS (quiet 47.4 ms/window). DIRECTIONS top note + §2/§4/§5/§6, SUBMISSIONS, HANDOFF, LOG entry, RESULTS
  summary are written (Wed 06:00) — Thursday only re-checks them.
- **Queue: all of P4 done** (QUEUE.md status table); 62 arms + 2 flagged.
- **Running:** filler plain Shallow 43–47 (`rows_pl3`); a timer for the 18:00 board re-check.
- **Left:** (1) Wed 18:00 board re-check (in-app browser JS fetch of `/api/phases/30483/get_leaderboard/?format=json&page_size=200`;
  note new top entries / names; append to RESULTS P3); (2) Thu 07:00 freeze = B3 (nothing beat it);
  re-run `sha256sum` on the submissions copy, glance at gates; (3) Thu 11:30: final LOG line (time
  totals), DIRECTIONS change-log line, HANDOFF ("done"), memory note in the codabench project (not
  the ALS_AI one), commit, push.

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
| 19:31–20:11 | resumed; B2 gated → candidate + upload note; Q2 sliding / Q6 cov weighting / extra nets no gain; C sweep → B3 (nested +2.04); Q3 dumps done; stopped 20:11 (usage limit) |
| 00:31–00:45 | resumed; B3 quiet 47.4 ms → candidate (14541a30); FBCNet / Q3 closed; Q8 chain + filler lane started |
| 05:31–06:05 | Q8 / Sinkhorn-on-B3 no gain → queue done; LOG, DIRECTIONS, RESULTS summary written; filler lane; 18:00 timer |
