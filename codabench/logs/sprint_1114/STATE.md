# STATE — Sprint A (Thu Nov 12 11:15 → Sat Nov 14 10:00 EST)

Brief `prompts/2026-11-12_sprint_a.md`. After a restart: read this, continue; do not re-plan.
Base B3 = probe `q5_q0.85_l5.0_s20_t20_f16_C0.03` (logs/sprint_1112/q5). Probe npz of new arms: logs/sprint_1112/q5/<name>_<split>.npz.

## Needs you
- Pause Windows Update (asked Thu 10:45). Confirm B3 was uploaded (slot 3).

## Log
| time | event |
|---|---|
| 11:15 | brief written; batch 0 (A1c1-3 wpl, A2a/b cadence) launched on the existing probe |
| 10:44 | batch 0 running (5 arms; launch fixed: MSYS_NO_PATHCONV=1 + /home/ali_d path); Sonnet builder (probe + vs-B3) and Sonnet A3 (P04) agents running; keep-awake held |
| 10:45 | A3 P04 done (Sonnet): B3 0.533 on P04; signal quality / covariance normal; pseudo-labels 0.52 and confidence does not separate right from wrong; causal balance bias −0.6 cohort → **A3 closed, no arm** (logs/sprint_1114/A3_P04.md) |
| 10:46 | probe sprint1114_a.py + sprint1114_vsb3.py built by Sonnet (B3 reproduced exactly on lopo0); batch 1 (A1a1-3, A1b1-2, A1d) launched, ETA 11:00 |
| 10:52 | batch 0+1 + 4 bags scored vs B3: **all no gain** (best BAGROB +0.09, later −1.25); A1/A3 closed; A2b (every 5) still running |
| 10:55 | A2b every 5 no gain; **all 15 arms + A3 closed; slot 4 held pending G1**; asked user for the next direction |
| 11:32 | **G1: B3 sealed 0.755** (id 1049925, 1,074 s, rank 81/207); budget not hit; transfer ≈ 55 %; band 0.740–0.760 = partial → slot 4 held; next = known-person ceiling check (Sonnet) |
