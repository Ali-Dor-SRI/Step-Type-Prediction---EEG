# STATE — sprint "to the top" (Tue Nov 10 15:14 → Thu Nov 12 12:00 EST)

Brief: `prompts/2026-11-10_to_the_top.md`. Queue: `QUEUE.md`. Results: `RESULTS.md`.
After a restart / context summary: read this file and QUEUE.md, then continue; do not re-plan.
WSL gotcha: `wsl.exe bash -lc '…'` expands `$VAR` and `~` in an outer shell — put any command
with variables in a script under `scripts/sprint1112_*.sh` and run that.

## Needs you

- (none) — session stopped again 20:11 Tue (usage limit). Resume with the HANDOFF start line.

## Machine (P0, 15:14)

- Keep-awake held (app, `session_idle`; holds while a job / wake-up keeps the session busy).
- `powercfg` SUB_SLEEP: idle sleep 0 (never) on AC and DC. Nothing to flag.
- WSL: 20 cores, 31 GB, 903 GB free.

## Now (updated 20:11)

- **Thursday candidate (gated, in DIRECTIONS top):** B2 `submissions/ens_x4f_shallow_al_st_all_2026-11-10.zip`
  `1ae19a3b…` (quiet 50.1 ms/window). Nested +1.96 vs B1 / later +2.27.
- **B3 = B2 with refit C 0.03** (nested over 16 configs +2.04 [+1.26, +3.34] vs B1, later +2.81): gate + bake
  running (`scripts/sprint1112_b3.sh` → `st_gates_b3.log`, `b3_cand_all/`). **On resume:** if st_gates_b3 ALL
  PASS and gate iii 1–3 / vi / CLI pass, re-time quietly (`sprint1112_quiet_timing.sh b3_cand_all`, ≤ 60 ms) →
  B3 replaces B2 as candidate: copy zip to `submissions/`, update SUBMISSIONS row + DIRECTIONS top note
  (sha256, predicted ≈ 0.77).
- **Q3 dumps done** (`logs/sprint_1106/dump/X4fc02__*`, `X4fc24__*`; crop 0–2 s / 2–4 s of the window): alone
  lopo ≈ 0.54 / 0.58. Next: `sprint1112_q5.py --model X4fc02` / `X4fc24` (deployment config, C 0.03), then a
  3-member stack (X4f + halves, each self-trained, + nets) vs B3 — needs a small extension of
  `sprint1112_combo.py` (several X4f-like members).
- **Running lanes:** fbcnet aligned (rows_fbc_al, ETA ≈ 22:00 → AdaBN via adabn_gen, stack on B3); filler
  Shallow aligned 43–47 / plain 38–42 (ETA ≈ 22:30; net side saturated, filler only).
- Still to do per brief: Wed 18:00 board re-check; Thu 07:00 freeze; Thu 09:15–10:00 final upload note;
  Thu 11:30 wrap-up (LOG entry, DIRECTIONS §2/§4/§6, HANDOFF, memory).

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
