# Sprint 2026-11-06 → 11-09 (weekend): grow the X4f + ShallowNet ensemble into a slot-2 zip

Time frame: the user's go (or Fri 2026-11-06 15:00) → **Mon 2026-11-09 12:00**; reserve
Mon 10:00–12:00. The user is away; the § 4 rules decide. Run with the `project-sprint`
skill. Read the whole brief first.

**How this branch was chosen.** The 15:00 table of `prompts/2026-11-06_next_session_plan.md`
§ 4 says: P2 no gain → Branch B (explore: adaptation + deep net). Sprint 1 found P2 no gain,
**and** measured Branch B's two levers already: label-free adaptation no gain; a deep net
alone 10 points under X4f, but its **ensemble with X4f is the only lever that moved**
(+1.3 to +1.7 lopo, +3.8 later). Deviation, in the open: the weekend develops that
ensemble (the plan's W3 / W4 / W5 path) instead of re-running Branch B's screens.

## 1. Context (verified 2026-11-06; `logs/sprint_1106/RESULTS.md`)

- Workstream `codabench/`, branch `feat/codabench-track2`, prefix `codabench:`, push to
  `personal` **and** `org` after every commit. WSL: `source ~/codabench/env040.sh`.
  Power: AC / DC idle sleep = never (read 11-06); re-read `powercfg` before launching.
- Sealed upload 1 = X4f all (`60a0148c`): **0.713, rank 51 / 51** (top 0.828, median
  0.775). Final rank = best of the last five; 4 uploads left (user); no warm-up check
  exists any more (a crash or timeout burns a slot). Board: "FBCNet-ensemble-v1" 0.741.
- Local X4f: lopo 0.6275 / later 0.6735, nested 0.6210.
- **Measured 11-06, do not repeat (all no gain):** offset-augmented training windows
  (X4f+aug4 −1.18 [−3.56, +0.50]; X4f+near2s:sep −0.47; C2f+aug4 fold 0 −13.7; Shallow +
  aug8 fold 0 −15.7); label-free TTA of X4f's classifier (EM, bias IM, T3A, class-mean
  shift; best +0.08); EEGNetv4 1× (fold 0 / later −19 / −22). Plus everything in the
  11-06 plan's "already measured" list.
- **The lead (P3b):** ShallowFBCSPNet (braindecode 1.8.1, 1× official windows, per-window
  per-channel z-score, AdamW 1e-3, cosine, 60 epochs, label smoothing 0.1, batch 64)
  alone lopo 0.533 (3-seed average) / later 0.588; **log-prob ensemble with X4f, net
  weight w:**

  | | lopo | Δ [95 % CI] (up/10) | later |
  |---|---|---|---|
  | 3-seed net, w nested (0.3 in 9 / 10) | 0.6406 | +1.32 [−0.12, +2.33] (8) | 0.7117 (+3.81) |
  | 3-seed net, w = 0.3 fixed | 0.6440 | +1.65 [+0.83, +2.40] (9) | 0.7117 (+3.81) |
  | 5-seed net (seeds 33–37), nested | FIVE_SEED_ROW | | |

- Tools: `analysis/sprint1106_netsmoke.py` (nets; rows `logs/sprint_1106/rows/NET.jsonl`,
  probabilities `net_<model>_<split>_s<seed>_o0.npz`), `sprint1106_ensemble.py`
  (`shallow 33_34_35`), `sprint1106_dump.py` (X4f streamed probabilities / features; 11
  dumps in `logs/sprint_1106/dump/`), `proteus_sweep.py --train-offsets`, the aug caches
  `~/neuralbench/proteus_cache_aug/off_<o>/`.
- **Server image:** braindecode is in `requirements.txt` but **unpinned**; torch from the
  CUDA wheel channel (CPU at run time is unknown). A submission must therefore not depend
  on braindecode's class layout: define the net in plain torch inside the submission file
  (or load braindecode's class and verify the state-dict keys at load, failing loudly).

## 2. Goal and priorities

Deliverable: `logs/sprint_1109/RESULTS.md`; an **all-trained, gated ensemble zip** "ready,
not uploaded" in SUBMISSIONS.md with a fact-sheet name (e.g. `X4f-Shallow-ens`), or a
written "hold" with the evidence; LOG / DIRECTIONS / HANDOFF updates.

1. **W1 Make the net stronger, cheaply** (Fri evening → Sat; 1 lane per arm, 25–30 min
   per seed-config on lopo 0–9 + later; 5 seeds per arm ≈ 2.5 h): (a) 120 epochs;
   (b) Euclidean alignment of the net's input per stream (causal running reference, buffer
   64, prior weight 8, as X4f's covariance re-centring; training per session), the E1f idea
   on Shallow; (c) Deep4 / EEGNet only if (a) and (b) both fail. Each arm judged by its
   **ensemble** with X4f (nested w), not alone.
2. **W2 Ensemble design** (Sat; minutes each, from saved probabilities): seed count (3 vs
   5), w grid {0.2, 0.3, 0.4}, nested; optionally a 3-way with C2f only if each pair is
   lean. Pick by § 4.
3. **W3 Submission wrapper** (Sat → Sun; budget 4 h, the critical path): a new solver
   `solvers/bci_decoding/ens_x4f_shallow.py` = X4f's model (unchanged code path, its own
   stream state) + K Shallow nets in plain torch (state dicts in the zip), log-prob mix
   with the baked w; `reset_state()` resets X4f's stream state (the nets are stateless).
   Gates: (i) the wrapper's streamed predictions on lopo fold 0 + later equal the
   post-hoc ensemble exactly (classes; probabilities to 1e-6); (ii) X4f's part alone
   reproduces K2; (iii) `proteus_w1016_gates.py --submission` ALL PASS, unzipped replay =
   training; (iv) threads pinned to 1; ms/window × 13 k ≤ 20 min on a quiet machine
   (Shallow forward ≈ ms; check); (v) zip size < the platform limit (state dicts are
   small); (vi) loads with a clean venv that has no braindecode.
4. **W4 Nested check** (Sun; ≈ 3 h): the ensemble's w and seed count re-chosen inside
   the 9 training people per outer fold (`sprint1106_ensemble.py` nested row is this for
   w; extend to the W1 arm choice if W1 adopted one). Flag over-selection at nested <
   selected − 3.
5. **W5 Bake** (Sun night → Mon 10:00): all-trained X4f (= the sealed `60a0148c` model)
   + K nets trained on all 2,655 windows, same seeds / epochs; sha256; SUBMISSIONS row.

## 3. Rules

- No uploads, posts, merges, tags; push both remotes after every commit; `logs/` git-ignored
  except `RESULTS*.md` (force-add). Frozen: `~/neuralbench/proteus_cache`, the
  benchopt_data_040 cache, `submissions/`, the X4f solver's defaults.
- Evidence: paired people bootstrap (10,000) over the 10 lopo people; `later` reported,
  never decisive alone; seeds 33–37 for nets, 42 for Riemann; nothing tuned on a test
  person or session (w is chosen nested); "no gain" said plainly.
- Every predict-path element causal, label-free, session-bounded (`reset_state()`).
- Resources: ≤ 3 lanes × 4 threads, `OMP_WAIT_POLICY=PASSIVE`; one Shallow seed on all
  11 splits ≈ 25–30 min per lane (2.6–4.4 s / epoch); nights up to 1.7× slower.

## 4. Decision rules (pre-registered; rules v2)

| comparison | adopted | lean | otherwise |
|---|---|---|---|
| W1 arm: its ensemble vs the 3-seed baseline ensemble (both nested w) | Δ lopo ≥ +1.0, CI lower > 0, later ≥ 0 | — | not adopted (keep the 60-epoch, unaligned net) |
| the final ensemble vs X4f (nested w) | Δ lopo ≥ +2.0, CI lower > 0, later > 0 | Δ lopo ≥ +1.0, CI lower > 0 | no gain |

**Slot-2 recommendation:** the ensemble zip is recommended for Tue Nov 10 if it is at least
**lean on the nested row** and W3 / W5 gates all PASS. If it is lean only on the fixed-w
row (as on 11-06), say so and recommend it anyway as a **model-family draw** (a worse
score cannot lower the rank; the board's own FBCNet ensemble at 0.741 suggests nets carry
signal on the sealed test), with the risk stated; the user decides. Never recommend a
zip whose W3 gates did not all pass.

## 5. Timing

| when | phase | ETA |
|---|---|---|
| Fri 15:00–20:00 | W1 (a) 120 epochs × 5 seeds (lane 1), (b) aligned input × 5 seeds (lane 2, after the alignment code + a 1-fold smoke), W3 wrapper drafting in parallel | 2.5–3 h per arm |
| Sat | W2 ensemble design; W3 wrapper + gates (i)–(vi) | 4–6 h |
| Sun | W4 nested check; W5 net training on `all` (5 seeds × ~4 min) + X4f all re-check | 3–4 h |
| Sun night → Mon 10:00 | W5 bake, gates, sha256 | 2 h |
| Mon 10:00–12:00 | reserve: RESULTS, LOG, DIRECTIONS § 1/2/6 + change log, HANDOFF § 00, SUBMISSIONS, final summary | 2 h |

Launch cutoff: nothing starts unless 1.5 × its ETA ends by Mon 10:00. Resumable step
scripts (`step_lib.sh`, `.done` markers), watchdog in a Monitor re-armed every 30 min.

## 6. Contingencies

- W1 arms both fail: keep the 60-epoch net; W3–W5 proceed on the 11-06 config.
- The wrapper cannot match the post-hoc ensemble exactly (gate i): find why before
  anything else (call shape, z-score dtype, thread count); no bake on a failed gate.
- Runtime over budget (gate iv): fewer seeds (3), never a different w.
- Everything blocked: deliver the measured table + "hold slot 2"; C2f all (`7059b103`,
  ready) remains the zero-work measurement option (does the board track lopo?).
