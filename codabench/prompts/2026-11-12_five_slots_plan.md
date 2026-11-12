# Plan — five sealed slots, Thu Nov 12 → Sat Nov 21 17:59 EST

Written Thu 2026-11-12 10:40 EST (Opus), after the Codabench marking fix.

## 0. What changed

- Organisers fixed a marking issue and gave everyone **+2 uploads → 5 left** (was 3).
- Rescored: slot 1 X4f all (id 1033809) **0.713 → 0.710**; slot 2 X4f + Shallow ens (id 1044956) **0.736 → 0.735**.
  Both moved ≈ −0.3, so the "local gains transfer ≈ 1:1" reading still holds (Δ +2.5 vs local +1.7 / +2.8).
- Windows Update rebooted the machine Wed 21:32. Lost: a filler lane only (aligned Shallow seeds 78–95) and the
  Thu 07:00 freeze timer. The freeze is B3 anyway (nothing beat it); its zip sha256 re-checked Thu 10:31 = `14541a30…`.
- **Rule to confirm (user, from the email / Codabench page):** final rank = best of the **last five** sealed uploads.
  With 7 uploads in total, using all 5 remaining pushes slots 1–2 (0.710 / 0.735) out of the counted five.
  Consequence: once B3 scores, the floor is B3's score, not 0.735; if B3 scores < 0.735, one later slot
  re-uploads the slot-2 zip `1f6589b8` (or something gated at least as good) so it stays counted.

## 1. Who does what

| Role | Model | Does |
|---|---|---|
| Lead | **Opus** (main session) | writes each sprint brief and its pre-registered decision rules; adopt / drop on every arm after the nested check; freezes each slot's zip; reads Δsealed vs Δlocal; the risk call on any new code path on the server; updates DIRECTIONS |
| Builders | **Sonnet** subagents | code new arms (analysis scripts, solver changes), bake zips, run gate suites (solver = probe, stateful 1–4, quiet timing, no-braindecode, benchopt CLI), write RESULTS sections |
| Runners | **Haiku** subagents | watch job logs / watchdogs, flag stalls against the estimate (CLAUDE.md overrun rule), re-check sha256, read the board (in-app browser), fill SUBMISSIONS / LOG rows and time logs |

Rules: one Sonnet writer per set of files; Haiku never edits solvers; every arm's verdict comes back to Opus
with its nested numbers before anything is baked. Lanes at 4 threads + `OMP_WAIT_POLICY=PASSIVE`; timing only quiet.

## 2. Timeline

| When (EST) | Slot | What | Owner |
|---|---|---|---|
| **Thu Nov 12, before 12:00** | **3** | **Upload B3** `submissions/ens_x4f_shallow_al_stC_all_2026-11-10.zip` (`14541a30…`), fact-sheet `X4f-ShallowAl-AdaBN-ST`, predicted ≈ 0.77 | **you** |
| Thu 12:00 – 14:00 | — | Machine prep: keep-awake, Windows Update **paused** (it killed the last run), sleep 0; board re-read after the rescore | Haiku |
| Thu 14:00 → Sat 10:00 | — | **Sprint A** (brief by Opus Thu 13:00): levers §3 A1–A3; slot-4 zip gated by Sat 10:00 | Sonnet build / Haiku watch / Opus decide |
| Fri Nov 13 (after B3 scores) | — | **Gate G1**: read B3 sealed score → branch in §4 | Opus |
| **Sat Nov 14, before 12:00** | **4** | Sprint A's best, or the §4 fallback | you |
| Sat 12:00 → Mon 10:00 | — | **Sprint B** = the draw: a higher-variance structural bet (§3 B) — affordable only because of the 2 extra slots | Opus brief, Sonnet build |
| **Mon Nov 16, before 12:00** | **5** | Sprint B's zip if gated and lean, else Sprint A's runner-up | you |
| Mon 12:00 → Wed 10:00 | — | **Sprint C**: stack what transferred (slots 3–5 Δsealed), nested check of the stack | Opus / Sonnet |
| **Wed Nov 18, before 12:00** | **6** | Best stack | you |
| Wed → Fri 10:00 | — | Consolidation only: re-bake the best measured model with every seed / net it can use inside the time budget; no new families | Sonnet / Opus |
| **Fri Nov 20, before 12:00** | **7** | Final: best-expected model (by sealed scores so far, not local) | you |
| Sat Nov 21, before 17:00 | (reserve) | Only if slot 7 failed **and** a failed run does not count as an upload | you |

One upload a day; every slot has a gated zip + one-screen note in DIRECTIONS by 10:00 the same day.

## 3. Work queue (ranked; Opus re-ranks at each gate)

**Sprint A (for slot 4), from DIRECTIONS § 6 and the 11-12 diagnosis (oracle gap ≈ 10 points is per-person):**
- A1. Better pseudo-labels — co-training: a second X4f-type model (different blocks) and the nets each refit on the
  other's confident windows; agreement filter. Sonnet builds on `SelfTrainEnsembleModel`.
- A2. Refit cadence within the CPU budget: every 10 / every 5 windows, warm-started; quiet-time gate must stay
  ≲ 60 ms/window (B3 = 47).
- A3. P04 diagnosis (worst person, gains nothing): artefacts / channel quality; 2 h, Haiku pulls, Opus reads.
- Bar (pre-registered): nested lopo ≥ +1.0 vs B3 with CI lower bound > 0, later not < −0.5; else slot 4 = B3-family
  re-run only if §4 says so.

**Sprint B (for slot 5) — the draw:** GPU path (the server has an A100; refit every window + net self-training on
GPU). Risk: no warm-up to test a GPU path any more (check first, Haiku: does the warm-up phase still accept uploads?).
The zip must fall back to the CPU B3 path when CUDA is absent or slow (time budget), so the worst case is B3.

**Sprint C:** stack only levers whose sealed Δ was seen.

**Not again:** gradient TTA (Tent / IM), more net families, time-domain / window-half features, robust covariances,
balancing tricks (all measured 11-12, no gain).

## 4. Gate G1 — after B3's sealed score (Fri Nov 13)

| B3 sealed | Reading | Slot 4 |
|---|---|---|
| ≥ 0.760 | self-training transfers | Sprint A's best (if it clears its bar), else hold slot 4 to Sun |
| 0.740 – 0.760 | partial transfer | Sprint A focuses on A2 / safer pseudo-labels; slot 4 = its best |
| < 0.740 | does not transfer (unseen people?) | slot 4 = B1 `73e38638` (AdaBN only, +1.5 local); Sprint A switches to diagnosing why |
| failed / timeout | server CPU slower than 2.1× | slot 4 = B3 with a tighter refit budget (re-gate) or B1 |

Record Δsealed vs Δlocal (+3.8 lopo / +3.3 later) in SUBMISSIONS and DIRECTIONS § 1 either way.

## 5. Machine hazards (this week)

Windows Update rebooted mid-sprint Wed 21:32. Before Sprint A: pause updates (Settings → Windows Update →
Pause 1 week — **your click**), keep-awake on, sleep 0 on AC/DC. Every lane resumable (`rows_*` per seed) and
watchdog-logged as in 11-12.
