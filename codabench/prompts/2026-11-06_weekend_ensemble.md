# Weekend sprint 2026-11-06 → 11-09: build the X4f + ShallowNet ensemble into the slot-2 zip

**Paste this whole file as the first message of a new session.** Then invoke:

`/project-sprint until Monday 2026-11-09 12:00  build the X4f + ShallowNet ensemble submission (prompts/2026-11-06_weekend_ensemble.md)`

Time frame: from the paste → **Mon 2026-11-09 12:00** (wrap-up reserve Mon 10:00–12:00).
The user is away until Monday 12:00; the rules in § 5 decide everything. Read the whole
brief before starting. Standing constraints of the skill apply: no uploads, posts, merges
or tags; push to `personal` **and** `org` after every commit; defaults bit-identical;
estimate / time-log / investigate overruns (CLAUDE.md).

**Nature of the sprint: building, with one bounded exploration lane.** The critical path
(W0 → W3 → W5) produces a gated, all-trained zip for **sealed slot 2 on Tue Nov 10**. The
exploration lane (W2: two ways to strengthen the net) runs beside it and can only change
the bake if it clears its rule; it never delays it.

## 1. Why this, and where we stand (verified 2026-11-06 13:30)

- Competition: EEG/EMG Foundation Challenge 2026, Track 2 (BCI, PROTEUS, 3 classes,
  streamed one window at a time per session; label-free adaptation inside a session
  allowed). Sealed phase Nov 4 23:01 → **Nov 21 22:59 UTC (17:59 EST)**, 1 upload per UTC
  day (the day turns at 19:00 EST), **final rank = best of each team's last five sealed
  uploads**. No warm-up check exists any more: a crash or timeout burns a slot.
- **Sealed upload 1** = X4f all (`60a0148c`, Nov 4): **0.713, rank 51 / 51** (top 0.828,
  median 0.775, 50th 0.736). 4 uploads left (confirmed on the Codabench page by the user, 2026-11-06). A team's
  "FBCNet-ensemble-v1" scores 0.741, a plain "LDA" 0.808.
- Local X4f (Riemann, per-stream re-centring, logreg): **lopo 0.6275 / later 0.6735**,
  nested 0.6210. Protocols: `lopo` = leave one of 10 people out (28 cells), `later` =
  session 1 of everyone → sessions 2–3 of P04 / P10 (8 cells); metric = balanced
  accuracy per (person, session, interface) cell, averaged; paired people bootstrap.
- **Sprint 11-06** (`logs/sprint_1106/RESULTS.md`, LOG "2026-11-06"): offset-augmented
  training windows **no gain** (X4f+aug4 −1.18, near offsets −0.47); label-free
  classifier adaptation **no gain** (best +0.08); nets alone 10–20 points under X4f;
  **but the X4f + ShallowFBCSPNet log-prob ensemble is lean:**

  | ensemble (net weight w, log-prob average) | lopo | Δ vs X4f [95 % CI] (up/10) | later |
  |---|---|---|---|
  | 3 net seeds (33–35), w nested | 0.6406 | +1.32 [−0.12, +2.33] (8) | 0.7117 (+3.81) |
  | **5 net seeds (33–37), w nested → 0.3 in 10 / 10 folds** | **0.6444** | **+1.69 [+0.94, +2.38] (9)** | **0.7013 (+2.78)** |

  Shallow alone (5-seed average): lopo 0.5312 / later 0.5803. "Nested" = for test person
  k, w maximises the mean cell score over the other nine lopo folds.
- **Do not repeat** (all measured, no gain): offset augmentation of any kind (Riemann and
  Shallow), TTA of X4f's classifier (EM, bias info-max, T3A, class-mean shift), EEGNetv4
  1×, and the 11-06 plan's list (router + personal LDA, no xDAWN, CAR, clipping, crops,
  Euclidean means, buffers / K on F64, dbpt8, bpt8, slow waveform, fbd, ROI / asymmetry
  / burstiness blocks, train-as-streamed, C 0.003 / 0.03, online class balance, X4f + C2f
  average, EEGNet E0f / E1f).

## 2. Assets and tools (all on disk; nothing to download)

| what | where |
|---|---|
| WSL entry point (all work) | `wsl.exe bash -s <<'EOF' … source ~/codabench/env040.sh … EOF` (repo = `~/codabench` → `/mnt/c/Users/Ali D/Documents/ML/codabench`); git only from Git Bash, never inside WSL |
| window cache (frozen) | `~/neuralbench/proteus_cache` (2,655 × 41 × 480, µV, 120 Hz) |
| net training harness | `analysis/sprint1106_netsmoke.py` (braindecode 1.8.1 ShallowFBCSPNet, per-window per-channel z-score, AdamW 1e-3 wd 1e-3, cosine over 60 epochs, label smoothing 0.1, batch 64, `torch.manual_seed(seed)` + randperm generator(seed)); rows `logs/sprint_1106/rows/NET.jsonl`, probabilities `rows/net_shallow_<split>_s<seed>_o0.npz` |
| X4f streamed probabilities / features | `logs/sprint_1106/dump/X4f__<protocol>__<fold>.npz` (`sprint1106_dump.py`; reproduce X4f exactly) |
| ensemble evaluation | `analysis/sprint1106_ensemble.py shallow 33_34_35_36_37` (fixed-w rows + nested row) |
| X4f solver (do not edit) | `solvers/bci_decoding/riemann_sealed_proteus_x4f.py`; the sealed zip `logs/proteus_w1016/proteus_x4f_cand_all/riemann_sealed_proteus_x4f_all_2026-10-27.zip` (sha256 `60a0148c…`) |
| bake / gate pattern | `scripts/sprint1027_bake_x4f.sh`, `scripts/proteus_candidate.sh`, `analysis/proteus_w1016_gates.py --submission`, `analysis/proteus_eval.py` (`load_submission` loads a folder as the platform does) |
| net-in-a-submission precedent | `solvers/bci_decoding/eegnet_stream.py` (self-contained torch solver with stream state) |
| runners | `.claude/skills/project-sprint/scripts/{step_lib.sh,monitor_loop.sh}`; lane examples `scripts/sprint1106_p2.sh` |
| server image | `$COMPET_ROOT/requirements.txt`: torch (unpinned, CUDA wheel channel), braindecode **unpinned**, neuralbench 0.4.0, benchopt ≥ 1.10, python 3.12 |

## 3. Goal and deliverables

1. `solvers/bci_decoding/ens_x4f_shallow.py`: one self-contained submission solver,
   gated, plus the baked all-trained zip under `logs/sprint_1109/ens_cand_all/`, sha256,
   and a SUBMISSIONS.md row "ready, not uploaded", fact-sheet name `X4f-Shallow-ens`.
2. `logs/sprint_1109/RESULTS.md` (force-add), LOG entry, DIRECTIONS § 1 / § 2 (decisions
   15–16) / § 4 / § 6 + change log, HANDOFF top section, memory.
3. The Monday final summary with a clear slot-2 recommendation (§ 5) and the draft brief
   for the mid-week exploration sprint (§ 8).

## 4. Work items (critical path first)

### W0 Plain-torch ShallowNet, proven equal (Fri, ≤ 1.5 h) — critical path
The server's braindecode is unpinned, so the submission defines the net itself.
- Write `ShallowNetT` (plain torch, same layers, init irrelevant) in a small module that
  the solver file will embed verbatim; map braindecode 1.8.1's `ShallowFBCSPNet` state
  dict onto it (print both key lists in the log).
- **Gate W0:** for seed 33 on lopo fold 0 and later, load the braindecode-trained weights
  into `ShallowNetT` and reproduce the saved probabilities (`net_shallow_*_s33_o0.npz`)
  to ≤ 1e-6, eval mode, float32, 1 thread. If braindecode wraps ops (e.g. `Expression`,
  `Ensure4d`, safe log), mirror them exactly. Fail → fix before anything else.
- Retrain nothing here: the saved npz + a re-run of seed 33 fold 0 with weights kept
  (add `--save-weights` to `sprint1106_netsmoke.py`, default off) gives the weights.

### W1 Ensemble solver + gates (Sat, budget 5 h) — critical path
- `ens_x4f_shallow.py`: X4f's model loaded **from the sealed zip's own weights**
  (`weights.joblib` of `60a0148c`, untouched code path copied from the X4f solver, its own
  stream state) + K `ShallowNetT` state dicts; predict = per-window z-score → K nets →
  mean probability P_net → argmax of `(1−w)·log P_X4f + w·log P_net` (probabilities
  clipped at 1e-12, as `sprint1106_ensemble.py`), w baked (0.3 unless W2
  changes it); `reset_state()` resets X4f's stream state (the nets are stateless); every
  thread pool pinned to 1 at load (the F64T lesson: default pools = 30× slower).
- Gates (all must PASS before W4 / W5):
  (i) streamed predictions of the wrapper on lopo fold 0 + later equal the post-hoc
  ensemble (`sprint1106_ensemble.py` with the same seeds / w): classes identical,
  probabilities ≤ 1e-6; (ii) with w = 0 it reproduces X4f (K2 rows) exactly;
  (iii) `proteus_w1016_gates.py --submission` ALL PASS, unzipped replay = training;
  (iv) ms/window on a quiet machine × 13,000 ≤ 20 min (server ran X4f at 27–30 ms/window;
  the 3,600 s limit is per run); (v) zip size recorded, well under the platform limit;
  (vi) loads and predicts in a throwaway venv **without braindecode** (torch + sklearn +
  pyriemann versions of the image), `python -I`.

### W2 Strengthen the net (exploration lane, Fri evening → Sat; ≤ 2 lanes) — optional
Each arm = 5 seeds (33–37) × 11 splits ≈ 2.5 h per lane; judged **only by its ensemble
with X4f** (nested w) against the 5-seed baseline row above.
- (a) 120 epochs (cosine over 120).
- (b) Stream re-centring of the net's input: Euclidean alignment per stream (causal
  running covariance reference, buffer 64, global-prior weight 8, as X4f's covariance
  re-centring; training sessions whitened by their own mean), then z-score. Needs a
  1-fold smoke first; the predict path must stay causal and reset per stream.
- Do not start (c) other families this weekend; they belong to the mid-week sprint (§ 8).

### W3 Nested check of the final choice (Sun, ≈ 1–3 h)
If W2 adopted an arm or W2 changed w / K: re-make every choice (arm, w, K) inside the 9
training people per outer fold; report nested lopo; **flag over-selection if nested <
selected − 3** (then fall back to the 11-06 config: 60 epochs, 5 seeds, w 0.3). If
nothing changed, the 5-seed nested row above *is* the check; say so.

### W4 Train the nets on `all` (Sun, ≈ 30 min)
K nets on all 2,655 windows, same recipe and seeds, fixed epochs (no validation); weights
into `ShallowNetT` format.

### W5 Bake (Sun night → Mon 10:00)
All-trained zip = X4f all weights (`60a0148c`'s, byte-identical file) + the K nets + w;
re-run gates (iii)–(vi) on the baked zip; its official-protocol score is meaningless
(test inside training); sha256; copy to `submissions/` (git-ignored); SUBMISSIONS row.

## 5. Decision rules (pre-registered; rules v2)

| comparison | adopted | otherwise |
|---|---|---|
| W2 arm: its 5-seed ensemble vs the 5-seed baseline ensemble (both nested w) | Δ lopo ≥ +1.0, CI lower > 0, Δ later ≥ 0 | not adopted (keep 60 epochs, unaligned) |
| final ensemble vs X4f (nested w) | **adopted** Δ lopo ≥ +2.0, CI lower > 0, later > 0; **lean** Δ lopo ≥ +1.0, CI lower > 0 | no gain |

Ties → the simpler config (fewer seeds, no alignment). w only from the grid {0.2, 0.3,
0.4}, chosen nested. **Slot-2 recommendation (Tue Nov 10):** the baked zip if the final
ensemble is at least lean on the nested row and every W0 / W1 / W5 gate PASSES. A gate
failure that cannot be fixed by Mon 10:00 → "hold slot 2", with the reason; never
recommend an ungated zip.

## 6. Keeping jobs alive

- Read `powercfg /query SCHEME_CURRENT SUB_SLEEP` first (11-06: idle sleep never on AC and
  battery); do not change it; ask only if it changed.
- Run each phase script in the foreground of a background Bash call (WSL dies when no
  `wsl.exe` stays attached); resumable steps with `.done` markers; STATUS.md ETA rows;
  Monitor watchdog re-armed every 30 min, filter on rows + Traceback / Error / Killed.
- ≤ 3 lanes × 4 threads, `OMP_WAIT_POLICY=PASSIVE`; time ms/window only on a quiet machine.
- Over 1.5–2× an ETA: stop and diagnose (last progress line, swap, a silent fallback).
- Never edit a `.sh` / `.py` a running lane imports; stage edits.

## 7. Time budget

| when | item | ETA |
|---|---|---|
| Fri (paste) → +1.5 h | W0 plain-torch net + gate | 1.5 h |
| Fri evening | W2 (a) lane 1, W2 (b) smoke then lane 2 | 2.5–3 h per arm |
| Sat | W1 solver + gates (i)–(vi); W2 ensemble rows | 5 h |
| Sun | W3 (if needed), W4 nets on `all` | 1–3 h |
| Sun night → Mon 10:00 | W5 bake + gates + sha256 | 2 h |
| Mon 10:00–12:00 | reserve: docs, § 8 draft brief, final summary | 2 h |

Launch cutoff: nothing starts unless 1.5 × its ETA ends by Mon 10:00. Expected slack ≈ 1
day; use it for W2, not for new families.

## 8. After the weekend: the upload calendar and the mid-week sprint (for the Mon report)

Every slot is a free draw (best of the last five; all five of ours count), so every slot
carries a model-family change or a stack, never a ≤ 2-point refinement; each sealed score
is recorded next to its local Δ as a measurement of how lopo maps to the board.

| slot | date (upload before 19:00 EST) | candidate |
|---|---|---|
| 2 | **Tue Nov 10** | this weekend's ensemble zip (if § 5 recommends it) |
| 3 | ~Sat Nov 14 | best result of the mid-week exploration sprint (Tue Nov 10 → Fri Nov 13) |
| 4 | ~Tue Nov 17 | a different family or wider stack, after its nested check |
| 5 | **Thu Nov 19** (Fri Nov 20 18:00 EST at the latest; never the last hours) | the final best; Friday kept free to re-upload after a failure |

On Monday, write `prompts/2026-11-09_midweek_explore.md` (draft, not run) with:
(1) a desk study of the sealed board first (API `https://www.codabench.org/api/phases/30483/get_leaderboard/`:
every fact-sheet name, score, duration, submission count; public pages / FAQ / forum, read
only) → hypotheses for "what does a plain LDA at 0.81 do that X4f does not";
(2) stronger net families judged by their ensemble with X4f (FBCNet, which scores 0.741
on the board; Deep4; ATCNet); (3) wider ensembles (X4f + C2f + nets, nested weights);
(4) how slot 2's score moved versus its local Δ, which decides whether slots 3–4 lean on
nets or on Riemann.

## 9. Contingencies

- W0 cannot reproduce braindecode's outputs: try pinning by vendoring braindecode 1.8.1's
  ShallowFBCSPNet source into the solver file (its license allows it; note it); last
  resort, load braindecode at run time with a loud key check, and report the risk.
- W2 arms both fail: bake the 11-06 config (60 epochs, 5 seeds, w 0.3).
- Runtime over budget (gate iv): fewer seeds (3), never a different w.
- Everything blocked: deliver the measured state + "hold slot 2"; C2f all (`7059b103`,
  ready, all gates PASS) is the zero-work alternative for the user to consider.
- Needs the user (list in the final summary, do not wait): the upload and its fact-sheet
  name. (The remaining-upload counter, 4, was confirmed on 2026-11-06.)
