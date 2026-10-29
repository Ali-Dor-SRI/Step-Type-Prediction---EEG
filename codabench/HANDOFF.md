# HANDOFF

## 0. CURRENT STATE — 2026-10-28 21:15 (read first; overrides everything below where they differ)

**Feature sprint 10-28 done** (brief `prompts/2026-10-28_feature_engineering.md`, table
`logs/sprint_1028/RESULTS.md`, LOG "2026-10-28"):
- New opt-in `xblocks` in `riemann_sealed.py`: `roi`, `asym`, `asymabs`, `burst`
  (computed on unwhitened windows, AF3/AF4 excluded, `align="stream"` only; defaults
  bit-identical: zhou EQUAL, C2f re-run equal). **All no gain** on C2f and X4f.
- **X4f passed its nested check**: nested lopo 0.6210 vs selected 0.6275 (optimism
  +0.64, flag at 3); inner picks X4f ×5, C2f ×5. `analysis/sprint1028_nested.py`.
- **Next (user):** X4f official zip (`9fb7ffe3`) to the warm-up, expect 0.546004 /
  0.545484 / 0.545695; if exact, X4f all (`60a0148c`) to the sealed phase. C2f all
  (`7059b103`) is the fallback. Nothing uploaded to the sealed phase yet (as far as we know).
- Nothing running (checked 21:15).

## 0b. STATE — 2026-10-27 17:35 (superseded where § 0 differs)

**Candidates (accuracy sprint 10-27, brief `prompts/2026-10-27_sealed_accuracy.md`,
table `logs/sprint_1027/RESULTS.md`, LOG "2026-10-27"):**

| model | what it is (all = F64 + 1-thread pools at load) | lopo (28) | later (8) | status |
|---|---|---|---|---|
| F64T | F64 recipe (§ 3) | 0.5672 | 0.6005 | old baseline; warm-up exact, 19.0 s |
| **C2f** | + filter bank `fbands=1to4_4to8_8to13_13to20_20to30_30to45` + `icoh` | **0.6083** (+4.11 [+3.11, +5.22], 10/10) | **0.6424** | **rule-selected; warm-up exact match 0.527602 / 0.526610 / 0.527152, 22.7 s** |
| **X4f** | C2f + `fscale=True` + `clf=logreg0.01` | **0.6275** (+6.02 [+4.33, +7.78], 10/10) | **0.6735** | best; fscale / logreg picked after seeing them → **nested check required** before a sealed upload |

No gain alone: fscale, logreg, dbpt8, bpt8, slow_block; fbd (delta + 4 bands) lean +1.09.

**Zips** (thread-pinned, all gates PASS, 12–14 ms/window; hashes in SUBMISSIONS.md;
copies in `submissions/`, git-ignored; originals `logs/proteus_w1016/proteus_<c>_cand_<p>/`):
C2f official `88dbccc1` (warm-up done), **C2f all `7059b103` = next: first sealed upload**,
X4f official `9fb7ffe3`, X4f all `60a0148c`. Rule: an *official* zip goes only to the
warm-up, an *all* zip only to the sealed phase (on the warm-up it scores ~0.99, meaningless).

**Uploads so far (warm-up):** F64 official (exact, 198 s: the thread bug), F64T official
(exact, 19 s), F64T all (0.988 = memorisation), C2f official (exact, 22.7 s). **Nothing
uploaded to the sealed phase yet.** Uploading is always the user's action.

**Next steps (DIRECTIONS § 2 decision 13):**
1. User: upload C2f all to the sealed phase (once it is open; start date not re-confirmed).
2. **Nested check of X4f** (~3 h on 2 lanes): re-select among {F64, fb6, C2f, X4f} inside
   9 people per outer fold (`proteus_sweep.py --protocols nest`, as `RESULTS_p7c_nested.md`;
   F64's nest rows exist in `logs/proteus_w1016/N*.jsonl`). Flag over-selection if nested
   < selected − 3 points. If X4f holds, X4f all is the next sealed upload.
3. **Feature sprint brief written 10-28, not run:** `prompts/2026-10-28_feature_engineering.md`
   (ROI relative power, asymmetry, |asymmetry|, burstiness on C2f; its Phase 3 nested
   check includes step 2). Start it with the project-sprint skill.
4. Optional: other levers on top of C2f (e.g. logreg C, more bands) only with pre-registered
   rules and a nested check — 10 people, selection optimism is the main risk.

**How to run (changes since § 9):**
- Solver opt-ins added 10-27 to `solvers/bci_decoding/riemann_sealed.py` (defaults
  bit-identical; zhou EQUAL): `fscale`, `clf` (`lda` | `logreg<C>`), `xblocks` `dbpt<K>`,
  `fbands`, `threads` (> 0 pins pools at load_model — **every zip must bake `threads=1`**).
- Sweep: `analysis/proteus_sweep.py` now scores at 1 thread (predictions unchanged).
- **Run lanes at 4 threads + `OMP_WAIT_POLICY=PASSIVE`** (two 10-thread lanes spin-waited
  3–10× slower on this i7-12700K). Bake: `scripts/sprint1027_bake_<c>.sh` →
  `scripts/sprint1027_bake.sh` (4-thread copy of `proteus_w1016_p4v2.sh`), ~13 min.
- g9/g10 in the bake always shows 3 FAILs for adaptive models (by construction, same as
  F64); the real gate is `proteus_w1016_gates.py --submission` (ALL PASS).
- Watchdog: run `monitor_loop.sh` **inside WSL** (from Git Bash it cannot see WSL pids and
  reports false DEAD), e.g. `wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1027_watch.sh <dirs>'`.

**Nothing is running** (checked 17:35). Branch `feat/codabench-track2`, pushed to both
remotes.

---

# HANDOFF (older sections)
(Written earlier on 2026-10-27, before the accuracy sprint. §§ 1–2 and 9–11 still hold; § 0 above supersedes §§ 3–8.)

**(Historical: read § 0 first.)** The user decides from
[DIRECTIONS.md](DIRECTIONS.md) (short, kept current by every session: § 1 status,
§ 2 decisions, § 6 next steps, change log); update it at the end of any session
that changes a decision, a number, the plan or the calendar (rule in `../CLAUDE.md`).
Everything below this section is the history of earlier sessions; its facts hold
unless this section says otherwise.

## 1. The project in five lines

- **What:** NeurIPS 2026 EEG/EMG Foundation Challenge, **Track 2 (BCI decoding)**, on
  Codabench. The user's MSc thesis (EEG step-type classification) lives in the same repo
  but is a separate line of work; every competition file is under `codabench/`.
- **Data: PROTEUS** (NEMAR `nm000290`): 10 people, 14 sessions (P04 and P10 have 3,
  the rest 1), 2,655 windows, 41 EEG channels, 120 Hz, 4 s windows, 3 balanced classes
  (0 = motor imagery, 1 = subtraction, 2 = word generation). Chance 0.333.
- **Contract (0.4.0):** streamed, one window at a time per person-session in recording
  order, a fresh `deepcopy` + `reset_state()` per session; causal, label-free adaptation
  inside a session is **allowed** (FAQ 10-13). Score = balanced accuracy averaged over
  (person, session, interface) cells. Inference only on the server, 60 min limit.
- **Sealed test** = later sessions of the 10 public people **plus 10 unseen people**, no
  calibration data. Locally we measure the two halves with `lopo` (28 cells) and `later`
  (8 cells); never merge them into one number.
- **Machine:** Windows 11 + WSL2 Ubuntu, CPU only (20 cores, no NVIDIA GPU).
  `~/codabench` in WSL is a symlink to this folder.

## 2. Calendar (as of 2026-10-27)

| Date | Event |
|---|---|
| Oct 24 | registration closed (we are registered, Codabench user `adoroodchi`) |
| to **Nov 1** | warm-up, extended from Oct 25 after a Codabench host outage of several days; 5 uploads/day; does not rank |
| **Nov 4, 18:01 EST – Nov 21, 17:59 EST** | **sealed phase**, the only ranked one: 1 upload/day, 10 in total, top 3 audited. Projected dates per the user (2026-10-29); it had **not opened** on 10-29 (was Oct 28 before the host outage) |

## 3. The model: candidate F64 (chosen 10-16 by pre-registered rules)

Pooled shrinkage LDA on filter-bank Riemannian tangent space + xDAWN + log-variance +
`bpt4` band-power time course, with two per-session running re-centrings computed from
the session's own unlabelled windows: covariance whitening (`align="stream"`, buffer
64, prior pseudo-count K = 8) and feature-mean centring (`fcenter=True`). No person ids,
no router.

| | lopo (28 cells) | later (8 cells) |
|---|---|---|
| setup vehicle (pooled LDA) | 0.419 | 0.492 |
| S64 (covariance re-centring only) | 0.484 | 0.530 |
| **F64 = the candidate** | **0.567** | **0.600** |
| FI64 = F64 + icoh (variant, below the 2-point bar) | 0.584 | 0.624 |
| nested check (all choices re-made inside 9 people) | 0.567 (optimism +0.04) | — |

Code: [`solvers/bci_decoding/riemann_sealed_proteus_f64.py`](solvers/bci_decoding/riemann_sealed_proteus_f64.py)
(= `riemann_sealed.py` at b989a06 with the settings baked as defaults, which is what
Codabench runs). Explained for EEG readers, with real-data figures:
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) (rewritten for F64 on 10-26). Every arm and
decision: [LOG.md](LOG.md) "2026-10-16 sprint", `logs/proteus_w1016/RESULTS_p*.md`.

Tried on PROTEUS, no gain: router + personal LDA (−4.9), dropping xDAWN, CAR, clipping,
1–3 s crop, Euclidean means, EEGNet (0.485 / 0.476) and the EEGNet + Riemann ensemble,
buffers 16 / all, K = 2 / 32.

## 4. Zips (all local; sha256 re-verified 2026-10-27)

| Zip (under `logs/proteus_w1016/`) | Role | sha256 | State |
|---|---|---|---|
| `proteus_f64_cand_official/riemann_sealed_proteus_f64_official_2026-10-16.zip` | warm-up format/runtime test (trained on the official split) | `d4b42478…71cd0cd3` | **uploaded 2026-10-26: EXACT match** 0.507944 / 0.506931 / 0.507285 (cell mean / pooled bal. acc. / accuracy), 198.3 s = 263 ms/window on the server |
| `proteus_f64_cand_all/riemann_sealed_proteus_f64_all_2026-10-16.zip` | superseded by F64T all (same predictions, slow threads) | `59121c39…6f78e1822` | not uploaded; do not upload |
| `../threads_1027/f64t_official/riemann_sealed_proteus_f64t_official_2026-10-27.zip` | **F64T warm-up re-time** (F64 + 1-thread pools; 9.4 ms/window, predictions = F64) | `fcfd7c62…80d7430` | ready, not uploaded |
| `../threads_1027/f64t_all/riemann_sealed_proteus_f64t_all_2026-10-27.zip` | **first sealed upload** (F64T, trained on all 14 sessions) | `a774135a…b83cb2a0db8` | ready, not uploaded |
| `proteus_fi64_cand_all/riemann_sealed_proteus_fi64_all_2026-10-16.zip` | sealed variant (F64 + icoh) | `e58d72f0…c039060e1` | ready, not uploaded |
| `proteus_fi64_cand_official/riemann_sealed_proteus_fi64_official_2026-10-16.zip` | FI64 warm-up twin (expected 0.503897 / 0.502830 / 0.503311) | `ebef0a8f…cabba65937` | ready, not uploaded |

Full hashes, sizes and the older superseded rows: [SUBMISSIONS.md](SUBMISSIONS.md).
Uploading is **always the user's action**; never upload, and record every upload there.
Local runtime: 8.2 ms/window on one quiet thread; **server 263 ms/window** (warm-up 10-26), so the 21,000-window upper bound would be ≈ 92 min of 60. Cause confirmed 10-27: default thread pools (local F64 at default threads = 264.7 ms/window); F64T pins them to 1: 9.4 ms/window, identical predictions (`logs/threads_1027/RESULTS.md`).

## 5. What is running

Nothing (checked 2026-10-27: no python / benchopt jobs in WSL). Branch
`feat/codabench-track2`, clean, pushed to **both** remotes (`personal` and `org`,
identical). Not merged to `main` (main = v2.7.0, thesis only).

## 6. What happened since the 10-16 sprint

- **Practice phase** closed Oct 25 with nothing of ours uploaded; then the Codabench host
  was down for days and the **warm-up was extended to Nov 1**.
- **10-26:** the user uploaded the official-trained F64 zip to the warm-up (result pending
  when this was written).
- **10-26:** EOG check: the raw BIDS release has **no EOG** (41 EEG channels only). An
  indirect look found the frontal class signal weakens with distance from the eyes, is as
  strong in delta as in theta and peaks in the second after feedback onset: possibly
  ocular. Ad hoc; no script was committed. Written up in MODEL_ARCHITECTURE § 3.2 and
  DIRECTIONS § 6 step 3.
- **10-26:** MODEL_ARCHITECTURE.md rewritten for F64 (pre-PROTEUS version kept in git at
  bfbeea6); new figures in `figures/` from `analysis/arch_figures.py`.
- **10-27:** this handoff rewritten for a new agent; LOG entry "2026-10-26/27" back-fills
  the items above.

## 7. What the user needs to do / decide (DIRECTIONS § 2)

1. **Decision 9:** read the warm-up result of the 10-26 upload (three numbers + run time +
   Finished/Failed) and send it. If it matches 0.507944 / 0.506931 / 0.507285, the
   adaptive streamed format works on image `v3`. If it **fails or differs**, that is the
   top priority before any sealed upload: get the ingestion log from the user and reproduce
   locally with `proteus_w1016_gates.py --submission` on the unzipped folder.
2. **Decision 10:** first sealed upload = the F64 all-trained zip (recommended, if 9 matches).
3. **Decision 11:** upload plan: F64 first, FI64 as the first variant, keep ≥ 5 of the 10
   uploads for the last week.
4. Decision 8 (adopt the 0.4.0 setup; done in practice), decision 6 (thesis: what are the
   One/Two prompt and the `96` trigger), decision 5 (optional, low priority).
5. Not a decision but theirs: upstream `origin/main` moved past 442edcd (a66bc23 on 10-15);
   our worktree stays at 442edcd until the user says to move it (then re-run self-test + G1).

## 8. Next steps worth doing (ranked; DIRECTIONS § 6)

1. Compare the warm-up result with the local numbers (minutes, needs the user).
2. Frontal / ocular question: (a) delta + finer time bins around feedback onset on lopo
   (a possible lever, ~1–2 h); (b) ICA on resting runs to remove eye components (for the
   write-up, ~2–3 h).
3. Thesis: XGB on the 0.5–2 s window (~3–4 h; needs decision 6).

Not worth it now: EEGNet variants, REVE/LaBraM on this CPU, the personal-LDA path, more
levers tuned on 10 people (buys over-selection, not evidence). Any model change must pass
the pre-registered rules (RELEASE_DAY § 6, rules v2: ≥ 2 points on lopo with the CI
excluding 0, and a gain on later) and the nested check before it replaces F64.

## 9. How to run things

All PROTEUS work runs **inside WSL**. From the Windows-side tools, write a script file and
call `wsl.exe bash -lc 'bash ~/codabench/scripts/<x>.sh'`; never inline loops or
variables (they arrive empty and paths get rewritten). Start WSL heredocs with
`exec 2>&1`. Long heredocs fail in the Bash tool: use the Write tool.

- **Environment:** `source ~/codabench/env040.sh` (must print `neuralbench 0.4.0 cwd
  /home/ali_d/neuralbench/competition-040`; never silence it). The old `env.sh` world is
  frozen; only the zhou regression gate runs there.
- **Self-test first in a new session:** `python ~/codabench/analysis/proteus_selftest.py`
  (349 checks, ~1 min).
- **One arm on lopo + later:** `python ~/codabench/analysis/proteus_sweep.py --lane <X>
  --name <arm> --set k=v ...` (resumable by key; `--protocols nest` = inner folds).
  Summary with paired people bootstrap: `python
  ~/codabench/analysis/proteus_w1016_summary.py --base F64 --arms ... --lanes ...`.
  An arm takes ~10–30 min; a nested arm 3.5–4 h.
- **Build a candidate zip:** `CANDNAME=<x> BAKE="k=v ..." bash
  ~/codabench/scripts/proteus_w1016_p4v2.sh` (bake, official + all chains, stateful
  gates, zhou regression; ~35 min). Clean timing: `scripts/w1016_timing.sh <dir>`.
- **Watch a run:** `bash ~/codabench/scripts/w1016_watch.sh <logdir>`. For a time-boxed
  unattended block use the `project-sprint` skill (`../.claude/skills/project-sprint`).
- **After any edit to `riemann_sealed.py`:** the zhou regression must read `train
  0.770000 replay 0.770000 EQUAL`; new options are opt-in with defaults bit-identical.

## 10. Traps (each one has cost time before)

- `proteus_eval.py g9g10` has three checks that assume a stateless model; they fail by
  construction for F64. Use `analysis/proteus_w1016_gates.py --submission`.
- Timing under CPU contention reads 10–20× slow; measure with other jobs paused.
- Settings must be **baked** into a solver copy: Codabench runs the file's defaults.
- Key on (person, session, run, cue) or the tensor SHA-1, never the loaders' `info` ids.
- benchopt caches on parameters, not data: `--no-cache` after any cache change. It prints
  no scores (read the `--output` parquet).
- Run repo `git` from Windows, not WSL (CRLF noise, no push credentials). `pkill -f` can
  kill its own shell. WSL has no zip/unzip; its /tmp is emptied between calls.
- Machine: OneDrive can revert `.git` on power events; battery idle-sleep kills long jobs
  (disable sleep before long runs). Never pull Stieger 2021 (399 GB) to C:.
- Commit + push to both remotes after major changes; don't commit big logs (`logs/` is
  gitignored apart from `RESULTS*.md`). Author credit: "Ali Doroodchi".
- A stale worktree of the already-merged branch `docs/readme-results` sits in another
  session's scratchpad (`git worktree list`); harmless, can be pruned.

## 11. Document map

| Need | File |
|---|---|
| decisions, status, next steps | [DIRECTIONS.md](DIRECTIONS.md) |
| what exists on PROTEUS, how to run it, gates | [PROTEUS_SETUP.md](PROTEUS_SETUP.md) |
| uploads, zips, hashes, expected server numbers | [SUBMISSIONS.md](SUBMISSIONS.md) |
| the model explained | [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) |
| data structure, contract, pre-registered rules v2 | [RELEASE_DAY.md](RELEASE_DAY.md) |
| every run with estimate vs actual | [LOG.md](LOG.md) |
| last sprint's plan and rules | [prompts/2026-10-16_proteus_weekend_model.md](prompts/2026-10-16_proteus_weekend_model.md) |
| EDA | [reports/proteus_eda/EDA.md](reports/proteus_eda/EDA.md) |
| task neuroscience | [reports/PROTEUS_task_neuroscience_2026-10-15.docx](reports/PROTEUS_task_neuroscience_2026-10-15.docx) |
| pre-release model and proxy evidence (historical) | [SEALED_RECIPE.md](SEALED_RECIPE.md) |

---

# HANDOFF — sprint 2026-10-16: the first real PROTEUS model (complete)

Plan of record: [prompts/2026-10-16_proteus_weekend_model.md](prompts/2026-10-16_proteus_weekend_model.md)
(started 2026-10-15 18:20 EDT, deadline Mon 10-19 08:00; all phases finished Fri 10-16 18:47).
Branch `feat/codabench-track2`, pushed to `personal` after every commit. Logs:
`logs/proteus_w1016/` (rows: `A.jsonl` Riemann lane, `B.jsonl` neural + extras, `C*.jsonl`,
`K*.jsonl`, nested `N*.jsonl`; predictions + probabilities `probs/`; tables `RESULTS_p*.md`
committed). Record: LOG.md "2026-10-16 sprint". Decision brief: DIRECTIONS.md.

## State (updated 2026-10-16 18:50): complete

| Phase | Result |
|---|---|
| 0 harness | `analysis/proteus_sweep.py` (= benchopt route B to 6 dp), solver opt-ins `align="stream"`, `fcenter`, `crop`, `clip` (defaults bit-identical: zhou regression EQUAL after every edit), stateful streaming gates `analysis/proteus_w1016_gates.py` |
| 1 base arms | bpt4 kept; router + personal LDA −4.9 (not adopted); covariance re-centring per stream +5.7 lopo → S64 |
| 2 ablations | nothing adopted (icoh +1.1) |
| 3 EEGNet + ensemble | no gain (EEGNet 0.485 lopo / 0.476 later; ensemble +0.27) |
| 6 feature re-centring | **+8.3 lopo, +7.0 later → candidate F64 (lopo 0.5672, later 0.6005)** |
| 4 candidate zips | F64 sealed (all-trained) + official-trained warm-up zip; FI64 variant; all gates PASS; 8.2 ms/window quiet |
| 5 / 7 nested + icoh | nested lopo 0.5668 (symmetric rule; optimism +0.04); icoh on F64 +1.65 (below the bar) |
| 8 K robustness | K = 2 −1.0, K = 32 −2.4: K = 8 kept |

## What is running

Nothing. Nothing uploaded.

## How to re-run / extend

- One arm: `python ~/codabench/analysis/proteus_sweep.py --lane <X> --name <arm> --set k=v ...`
  (in WSL after `source ~/codabench/env040.sh`; resumable by key; `--protocols nest` for the
  nested inner folds). Summary: `python ~/codabench/analysis/proteus_w1016_summary.py --base F64 --arms ... --lanes ...`.
- Candidate chain: `CANDNAME=<x> BAKE="k=v ..." bash ~/codabench/scripts/proteus_w1016_p4v2.sh`
  (bake, official + all chains, stateful gates, zhou regression); clean timing:
  `scripts/w1016_timing.sh <submission dir>` (pauses lane B by PID).
- Watcher: `bash ~/codabench/scripts/w1016_watch.sh <logdir>...` (one-shot; exits on END /
  crash / STALLED / DEAD / FAILED).

## Facts a resumer needs

- `proteus_eval.py g9g10` has three checks that assume a stateless model (`align == 'none'`;
  B = 1 vs one batch over all streams; reset check after that batch): they fail by
  construction for any adaptive model. Use `proteus_w1016_gates.py --submission` instead.
- Timing measured under CPU contention is 10–20× slower (thread oversubscription); measure
  with the other lane paused.
- Never pass shell loops or variables inline through `wsl.exe bash -lc '…'` from Git Bash
  (variables arrive empty, paths get rewritten): write a script file.

## What the user needs to do

Upload `logs/proteus_w1016/proteus_f64_cand_official/riemann_sealed_proteus_f64_official_2026-10-16.zip`
to the warm-up (expected 0.507944 / 0.506931 / 0.507285) before Oct 25; then decide the first
sealed upload (DIRECTIONS decisions 9–10).

---

# HANDOFF — setup session 2026-10-15: PROTEUS training/test configuration (complete)

**The short version for decisions is [DIRECTIONS.md](DIRECTIONS.md)**; what exists on
PROTEUS and how to run it is [PROTEUS_SETUP.md](PROTEUS_SETUP.md). This section is
the detailed hand-off of the setup session and is resumable from here alone.

Plan of record: [prompts/2026-10-13_proteus_train_test_setup.md](prompts/2026-10-13_proteus_train_test_setup.md)
(started 2026-10-15 17:38:51 EDT, deadline 00:38:51, 7 h; finished 18:52, 1 h 14 min). A setup and verification session on the released PROTEUS data in the
0.4.0 environment: no tuning, no new features, no uploads. Branch
`feat/codabench-track2`, pushed to `personal` after every commit. Logs:
`logs/proteus_setup/` (`RESULTS.md` committed: every gate with its measured value,
the reference table). Record: LOG.md "2026-10-15".

## State of this session (updated 2026-10-15 18:52): complete

| Phase | Status | Result |
|---|---|---|
| 0 orient | done 10caab4 | git state as expected; nothing running in WSL; upstream local head 442edcd; **upstream `origin/main` has moved to a66bc23** (staying on 442edcd: the user's call); `benchmark_utils` symlink intact |
| 1 entry point + reference run | done d75fa12 | `env040.sh`, `config/venv040_freeze.txt`; G0 PASS; G1 PASS 0.395863 / 0.395465 / 0.395672 / 0.396026, 8 cells (`logs/proteus_setup/ref_meanlogreg/`); SETUP.md section + LOG back-fill for the 10-13 setup |
| 2 window cache | done 04a9970 | `analysis/proteus_cache.py` (build 11 s), `analysis/proteus_g4_events.py` (isolated `-I`). G2 22/22, G3 12/12 (max diff 0.0), G4 4/4 PASS. Cache `~/neuralbench/proteus_cache` (209 MB) |
| 3 protocols | done 6a4596d | `analysis/proteus_splits.py`: `split`, `ids_of`, `every_split`, `lopo_summary`. G5a–d 80/80 PASS |
| 4 streamed evaluation, two routes | done 83317ed | route A `analysis/proteus_eval.py`, route B `datasets/proteus_protocols.py`. G5e 78, G6 6, G7 107, G8 26 PASS; both routes = benchopt to 6 decimals, identical `y_pred` |
| 5 one solver through the whole path | done ef945a5 | `reset_state()` in `riemann_sealed.py` (only edit); vehicle `solvers/bci_decoding/riemann_sealed_proteus_pooled.py`; `scripts/proteus_candidate.sh` ran official, all, later, lopo 0..9. G9 / G9b / G10 / G11 / G12 PASS; official vehicle 0.406751 (8 cells, info only); two zips ready, NOT uploaded |
| 6 reference table, self-test, independent review | done | table in RESULTS.md (12 route B `MeanLogReg` runs + the vehicle + constant); `analysis/proteus_selftest.py` **349 / 349 PASS from a clean shell, 55 s**; review by a fresh sub-agent: 8 findings, none making a number wrong, all fixed (script failure reporting and resumption, audit trail, cell-count assert, flag name, `score` guard, environment guard, column union); all 13 runs re-run resumably, self-test 349/349 again (RESULTS.md "Independent review") |
| 7 docs and hand-off | done | `PROTEUS_SETUP.md`; DIRECTIONS § 1/2/5/6 + change log (148 lines); LOG, HANDOFF, SUBMISSIONS; RELEASE_DAY § 5 step 1 and § 6 updated in place; banners on SEALED_RECIPE, TRACK2_BCI, SUBMISSION_GUIDE, COMPETITION; project map; memory |

## What is running

Nothing of this session's. No uploads were made. The zips under `logs/proteus_setup/`
are local only. **Another session** started sweep jobs in this folder at about 18:30
(`scripts/proteus_w1016_p*.sh`, `analysis/proteus_sweep.py`, lanes A and B, logs
`logs/proteus_w1016/`): not this session's, not touched; they share the CPU, so any
timing taken after 18:30 is contended.

## Facts a resumer needs

- **Environment:** every WSL call starts with `source ~/codabench/env040.sh` (prints
  `neuralbench 0.4.0 cwd /home/ali_d/neuralbench/competition-040`; cwd = `$COMPET_ROOT`).
  The old `env.sh` world is frozen; `train_sealed.sh … zhou2016_xsess` still runs there
  as the regression gate after any edit to `riemann_sealed.py` (G11, 2 min; expected
  `gate: train 0.770000 replay 0.770000 EQUAL …`).
- **Self-test:** `python ~/codabench/analysis/proteus_selftest.py` (G0, G2–G8, G10;
  about 1 min; trains nothing; reads the cache, the official loader, the Phase 1
  `MeanLogReg` export and the `official` Riemann export). Run it first in any new session.
- **Protocols** (`analysis/proteus_splits.py`): `official` 1521 / val 379 / 755 (4 streams,
  8 cells); `lopo` fold k = 0..9 holds out P(k+1) (28 cells in all; score = mean over the
  28 cells, fold means beside it); `later` 1893 / 762 (P04 + P10 sessions 2–3); `all`
  2655 / 755 (`score_meaningless`). Never combine `lopo` and `later` into one number.
- **Running a solver on a protocol:** route B by path,
  `-d "$HOME/codabench/datasets/proteus_protocols.py[protocol=lopo,fold=3]"`, with
  `--seed 42 --no-cache --no-plot --no-html --no-display --output <abs>.parquet`, a fresh
  `COMPET_SUBMISSION_DIR` for training, one solver and one fold per call; or the whole
  chain with `bash ~/codabench/scripts/proteus_candidate.sh <protocol> [fold]`
  (resumable; `logs/proteus_setup/cand_<run>/`). Per-window predictions:
  `python ~/codabench/analysis/proteus_eval.py score <export> <protocol> [fold] --out <dir>`.
- **Settings must be baked** into a candidate copy of the solver (Codabench runs a file's
  defaults); the vehicle copy was made by `proteus_candidate.sh` with sed on two lines.
- **Keys:** the official loaders' `info` integers are not stable; everything of ours keys
  on (person, session, run, cue) and on the SHA-1 of the float32 tensor.
- **Plumbing numbers are for information only** (untuned vehicle; sd of a label-blind
  cell mean ≈ 0.025 on 8 cells): reference table in `logs/proteus_setup/RESULTS.md`.
- **Two defects of my own test code** were found by the first run and fixed after the jobs
  ended (see LOG): `equal` compared `objective_name`; the deepcopy check mixed call shapes.
  Both fixed in `proteus_eval.py`; everything re-run.

## What the user needs to do

1. **Decision 7 (model half):** choose the comparison to run on `lopo` + `later`
   (DIRECTIONS § 2 names the four arms). Nothing in this session is evidence for or
   against a model.
2. **Decision 8:** adopt the 0.4.0 setup as the main one (largely done; yes / no).
3. **Upstream moved:** `origin/main` is at a66bc23, our worktree at 442edcd. Say whether to
   move it (then re-run the self-test and G1).
4. **Uploads (optional, before Oct 25):** `logs/proteus_setup/cand_official/riemann_sealed_proteus_pooled_official_2026-10-15.zip`
   (expected 0.406751 / 0.405717 / 0.405298) and/or
   `logs/proteus_setup/ref_meanlogreg/meanlogreg_official_2026-10-15.zip`
   (0.395863 / 0.395672 / 0.396026): proves the streamed format on the `v3` image.

---

# Addendum 2026-10-02 (evening): open sprint, nothing running

- **Thesis** (branch `feat/window-effect-check`, git worktree `C:\Users\Ali D\wt-window`, pushed to `personal`, not merged):
  window-effect check done. No temporal leakage in the shuffled CV; the full-vs-late gap sits in the first 0.5 s after
  t = 0 (the `96` event ~0.55 s after the One/Two prompt). Needs the user: what the prompt is (DIRECTIONS decision 6).
  Results: `docs/sprints/2026-10-02_window_effect_check_RESULTS.md` on that branch.
- **Warm-up:** 5-member EEGNet ensemble 0.8095 (< WU1 0.820): no gain, stopped. The leaderboard's 0.9+ is not reachable honestly.
- Sealed recipe untouched. Nothing uploaded. LOG entry "2026-10-02 (evening)".

---

# HANDOFF — sprint 2026-10-01: dual LDA, strict CV references, release-day rehearsal

**The short version for decisions is [DIRECTIONS.md](DIRECTIONS.md)**: status,
decisions pending, model, next steps. This file is the detailed session handoff.

Plan of record:
[prompts/2026-10-01_dual_lda_strict_wcv.md](prompts/2026-10-01_dual_lda_strict_wcv.md)
(17:36 → 05:36, two addenda). Branch `feat/codabench-track2`, pushed to
`personal` after every commit. Logs: `logs/sealed_s1001/` (STATUS.md,
`RESULTS_*.md`), `logs/sealed_s1001_rel120/` (the rehearsal's ablations),
`logs/s1001_*` (train_sealed flows). Record: LOG.md "2026-10-01".

## State of this sprint (updated 2026-10-01 22:45): sprint complete

| Phase | Status | Result |
|---|---|---|
| 0 orient + brief | ✅ db8b35c | per-subject LDAs were 200 s of a 260-s fold at 500 Hz (bpt4+icoh): picked the dual LDA |
| 1 dual (n < p) shrinkage LDA | ✅ f8e9175 | `_DualLsqrLDA` (both solver files): within 4e-12 of the Cholesky solve, 50–60× faster per subject. D1 PASS |
| 1b/6 deployment-test zip | ✅ 71607ea | Dreyer, bpt4 baked: train = replay 0.615873. 4.1 MiB (the first build was a 2.4 GB zip: `covariance_` is now dropped at every size). NOT uploaded |
| 2 gates | ✅ 07f4883 | regression 0.770000, mock flows 0.483333 / 0.512500, xblocks gate, Scherer icoh rows, ablations: all identical |
| 2 G4 500 Hz sizing | ✅ 26acd44 | recipe 15.7 min (was 19.3), bpt4 1.13×, bpt4+icoh **1.23× PASS** (was 1.78×), strict +12 min. D2: icoh passes rule 3b at 500 Hz |
| 3 strict CV references | ✅ 36902b3 | solver `wcv_ref` = harness `--wref` to 4 dp (pairs and subjects); defaults identical. D3: `WREF=strict` is the release-day setting |
| 4 reviews | ✅ bb280ac | dual LDA (3 minors fixed), strict code (6 minors fixed), integration review (1 pre-existing major fixed: step 2 now reads W1 / HARNESS_FROM from `R1`, and train_sealed.sh stops on a baked weight ≠ the seeded harness weight; checked) |
| 5 full-size 120 Hz rehearsal | ✅ 71607ea | § 5 (a) 12 steps 69 min; § 7 step 1 28 min, MATCH, EQUAL, train = replay 0.676944. Found and fixed: summarizer online-row bug |
| 7 shared-covariance personal LDA | ✅ 05ac612 | **NO GAIN** (Scherer 3-class −2.66 / −0.86) |
| 8 harness ctx_min for router-psdctx | ✅ 71607ea | check PASS; regression identical |
| extra checks | ✅ 4403f5a | the deployment joblib loads identically under sklearn 1.6–1.8, numpy 1.26–2.5, pyriemann 0.7–0.11; strict parity with eeg+eog; final zhou regression identical (0.770000, \|dP\| = 0) |

## What is running

Nothing. No uploads were made. The zips under `logs/` are
local only. The deployment-test zip is for the user to upload (SUBMISSIONS.md).

## Facts a resumer needs (this sprint)

- **`LDA_DUAL`** (both solver files) is on by default.
  - It applies above `LDA_FAST_P` = 4,000 features, with n < p and float64
    input, and only after a once-per-process self-check against sklearn's
    `_class_cov`.
  - Every proxy without icoh is unaffected (bit-identical).
  - Checker: `analysis/lda_dual_check.py`.
- **`WREF=strict`** is in RELEASE_DAY's DEC and § 5 commands. The defaults of
  `train_sealed.sh`, `release_ablations.sh`, the solver and the harness stay
  `all`, so every committed number reproduces.
  - Under strict the harness rows carry `wref=strict` in their config key;
    online-alignment rows stay `all`.
  - `train_sealed.sh` refuses `WREF=strict` for non-router alignments and
    switches it to `all` under `ADAPT=online`.
- **`train_sealed.sh` now prints a fold-score line** ("fold scores per w …
  EQUAL (4 dp) | DIFFER"). It is information; the weight gate stays MATCH.
  On zhou2016_xsess, DIFFER is expected: benchopt trains on 1,176 windows,
  the harness on 1,200.
- **`xsess_lib.read_results(path)`** returns the rows of every
  `results_*.jsonl` in that folder. Filter on `study` (now done in
  `compare_results.py` and `pc_screen.py`).
- **The harness's router-psdctx alignment** now gives a pair with < 16
  training windows its subject's reference, as the solver does
  (`info["ctx_small_pairs"]`).
- **New tools:**
  - `analysis/compare_results.py` (row-by-row equivalence of two runs);
  - `analysis/lda_dual_check.py`;
  - `analysis/ctxmin_check.py`;
  - `analysis/pc_screen.py` (Phase 7, no gain);
  - lane scripts `scripts/sprint1001_*.sh`.

## What the user needs to do

1. Ask the organisers on **Discord** (https://discord.gg/yZv8KqKMpH) whether
   `predict()` may use statistics of unlabelled test windows. The listed email
   `neurips2026-eeg-emg-competition@googlegroups.com` bounced (2026-10-02), and
   GitHub issues get no organiser replies. Order and persistence are answered
   by the public code (LOG 2026-10-02).
   A draft (2026-10-02) also asks whether running statistics count as
   "training" (the guide says not to train during `predict`) and whether the
   model stays loaded between calls. It is still the largest lever.
2. ~~Upload the deployment-test zip~~ **done 2026-10-02**: submission 956971,
   Finished, 0.62, 58.68 s (SUBMISSIONS.md).
3. Review rule 4 (`WREF=strict`) and rule 3b (icoh now passes at 500 Hz) in
   RELEASE_DAY § 6.
4. Optional: merge `docs/sprint0928-consistency`, which fixes the stale
   perf-loop numbers on `main`; REVE/LaBraM weights are still parked.
5. Optional (thesis, for a later sprint): say what the epoch's t = 0 is and
   whether that cue already shows the step direction. The full-CNV window
   (0–2 s, the 0.714 headline) beats the late window (1–2 s) by +0.09 AUC.
   If the t = 0 cue differs between straight and diagonal, the 0–1 s gain may
   be a visual-cue effect. The repo does not document it, so the check could
   not be set up this sprint.

The sections below are the previous sprints' handoffs; their facts still hold
except where this section supersedes them (icoh's 500 Hz verdict, rule 4's
guard, the release-day budget).

---

# HANDOFF — sprint 2026-09-29/30: temporal and spatial feature blocks

Resume from this file alone. Plan of record:
[prompts/2026-09-29_temporal_spatial_features.md](prompts/2026-09-29_temporal_spatial_features.md)
(17:46 → 05:45, with an erratum at the end). Branch `feat/codabench-track2`,
pushed to `personal` after every commit. Logs: `~/codabench/logs/sealed_f0929/`
(STATUS.md, `RESULTS_*.md`, per-study `results_<study>.jsonl`). The lanes
`scripts/sprint0929_f*.sh` are resumable.

## State of this sprint (updated 2026-09-30 02:10): sprint complete

**Outcome:**
- **The band-power time course `bpt4` is adopted** into the release-day recipe
  (`RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4`).
  - Blend_calib on the sealed-like proxy (Scherer WORD / SUB / HAND): +4.35
    (95 % CI +2.05, +6.67).
  - Positive point estimates on all 4 proxies, with the CI above 0 on 3.
  - It passes the 500 Hz sizing gate (1.06×).
  - Caveat: the confirmation reuses the screen's test sessions.
- **Imaginary coherence `icoh` (lagged connectivity) is PROMISING.** It becomes
  the release-day ablation `xa`. Post-hoc, bpt4+icoh gave +6.51. But it
  **fails the 500 Hz sizing gate**: 1.78× the recipe's fit measured alone
  (34 vs 19 min), so rule 3b keeps it out on a 500 Hz cache for now.
- **No gain under the deployed recipe:**
  - time-segment covariances (`tseg3`);
  - topography-type spatial blocks: CSP, regional covariances, band-power maps,
    re-referencing.
- Tables: SEALED_RECIPE § 2 "Phase 7"; LOG 2026-09-29/30.

| Phase | Status | Result |
|---|---|---|
| 0 brief + harness hook | ✅ 8c50541, 7e0d008 | `analysis/xfeat.py` (block registry, `RiemannXModel`), spec keys `x=` / `sl=`, `--split first` |
| 1 build (workflow, 5 agents) | ✅ 9b3719b | 12 verified blocks + 2 EDA-motivated (`tcut1000`, `fbfrom1000`); activation EDA in `reports/features_0929/`; baseline bit-exact; slow block / CAR / Laplacian: no gain |
| 2 screen | ✅ fe0efe4 | tseg3 +3.33, bpt4 +2.59, icoh +2.59 on Scherer 3-cl.; `RESULTS_screen.md` |
| 3 confirm | ✅ 4562ac4 | bpt4 ADOPT, icoh PROMISING, tseg3 NO GAIN (+1.96), tseg3+icoh ADOPT by the letter but not deployed; reverse split: no downgrade; `RESULTS_confirm.md`, `RESULTS_reverse.md` |
| 4 integrate + verify | ✅ e1b7402 | solver opt-in `xblocks` (tseg<K>, bpt<K>, icoh): defaults \|dP\| = 0, harness parity exact. `train_sealed.sh` bakes `RECIPE_SPEC … x=`. `release_ablations.sh` steps `xb` / `xa` (env `XB`), summarizer rule 3b. zhou regression 0.770000 = replay. Mock bpt4 flow: train = replay = harness 0.5125. Mock default flow 0.483333 = committed. 500 Hz sizing: bpt4 1.06× PASS; bpt4+icoh **FAIL** (1.56× contended, **1.78× alone**: 2,062 vs 1,157 s) |
| 4 extras (post-hoc, information) | ✅ e6af308 | bpt4+icoh: +6.51 / +1.81 / +2.16 / +3.33; online-64 Scherer 3-cl.: recipe 0.561, bpt4 0.592, bpt4+icoh 0.599 |
| 5 review (workflow, 6 agents) | ✅ eb39e70, 80d4da3 | 36 confirmed findings fixed. Two majors: the release-day summarizer lost `x=bpt4` (fixed and hardened: reads `config.txt`), and rule 3b's reading of the brief (documented; the icoh rule is now literal). LOG "Corrections" block, brief erratum |

## What is running

Nothing. The last job, the 500 Hz sizing re-measured alone, ended at 02:03
(`logs/sealed_f0929/RESULTS_sizing.md`). No uploads were made. The zips under
`logs/` (mock, regression) are local only; never upload them.

## Facts a resumer needs (this sprint)

- **The recipe is `x=bpt4` only through the release-day commands.**
  - Every script default and the solver default stay `xblocks=""`, so the
    regression gate still reproduces 0.770000.
  - RELEASE_DAY § 5 passes `SPEC=…,x=bpt4 XB=icoh`.
  - § 7 takes `RECIPE_SPEC` from the DECISIONS; the `DEC` default carries
    `x=bpt4`.
- **benchopt:** blocks are joined with `_` (`xblocks="bpt4_icoh"`), because `+`
  may be parsed as arithmetic. `train_sealed.sh` converts `x=bpt4+icoh`.
- **`release_summarize.py`** now reads `spec=` / `xb=` from the run's
  `config.txt` and exits on a mismatching `--spec` / `--xb`.
- **Harness `x=` blocks** run per 512 MiB chunk. Stateless blocks skip the
  full-array fit, and chunked = one-shot exactly.

## What the user needs to do (the previous sprint's items, plus 3 and 4)

1. Post the organiser question (SEALED_RECIPE § 5 step 1).
2. A warm-up deployment-test upload before 2026-10-25 (SUBMISSIONS.md). The
   candidate would now carry bpt4 (`xblocks="bpt4"`); the joblib gains only
   plain numpy arrays for it.
3. Review rule 3b (RELEASE_DAY § 6), in particular the ADOPT reading: bpt4
   stays unless the replica is ≥ +1.0 without it.
4. Decide the icoh sizing policy for a 500 Hz release: it fails the brief's
   relative gate (1.78×) but meets the older absolute rule (fit 34 ≤ 45 min).
   Either accept the absolute rule, or keep icoh out at 500 Hz. The other way
   to lift the block is a restricted icoh (2 bands) or a dual LDA solve, which
   is sprint work.
5. Optional: review / merge `docs/sprint0928-consistency`; approve REVE /
   LaBraM weights (still parked).

The sections below are the previous sprints' handoffs (2026-09-28/29,
sealed-release readiness); every fact in them still holds.

## State of the previous sprint (updated 2026-09-29 02:30): sprint complete

| Phase | Status | Result (details: LOG.md 2026-09-28/29 entries) |
|---|---|---|
| 0 brief | ✅ a44526a | three false claims found in the code before planning |
| 1 build (workflow, 9 agents) | ✅ bd931df | sealed-structure harness, mock sealed study, fast LDA, release scripts; defaults bit-identical |
| 3 online stream/order (real proxies) | ✅ bd931df | online gain not order-robust (+3.0/+4.4 interleaved vs +3.5/+7.5 recording order); no session lag; reset guard no gain |
| 2 dress rehearsal (full-size mock, 120 + 500 Hz) | ✅ ee5377d | every gate exact; 500 Hz sizing passes (fit 30 min, 12 GB); release-split ablations catch the EMG trap |
| 4a solver `align="subject_context"` | ✅ ee5377d, e64c58a | deploys the harness's router-psdctx; adversarially reviewed |
| 4b claims check + `release_eda.py` | ✅ ee5377d | 30 doc mismatches found |
| 4c runbook fix + literal verification | ✅ ceb3827, 5d39463 | RELEASE_DAY.md followed step by step on the mock; § 9 is the log |
| 5 final 3-lens review + fixes | ✅ (last commit) | channel-name guard in the solver; silent-failure paths closed; LOSO bias documented |

## What is running (previous sprint)

Nothing. No uploads were made. The zips under `logs/` are mock or regression
candidates only; never upload them.

## Where things are (previous sprint)

- **[RELEASE_DAY.md](RELEASE_DAY.md)**: the release-day runbook, from download to
  a checked zip. It holds the pre-registered rules, the dress-rehearsal timings
  and the verification log.
- **[SEALED_RECIPE.md](SEALED_RECIPE.md)**: the recipe and the evidence. § 5 lists
  the ranked next steps, including the two added by the final review.
- **Mock study:** `analysis/mock_sealed.py` (caches `mock_sealed_s`, `_120`,
  `_500`) and `datasets/mock_sealed.py` (benchopt, path form only;
  `split=organisers|replica_full`).
- **Harness** (`analysis/xsess_lib.py`, `sealed_run.py`, `sealed_personal.py`):
  splits `last` | `calib:K` | `replica:K` (mock only), context cells,
  `--wcv loso`, `--chans`, `--pool`, `router-psdctx`, `--router_cap`,
  `--wvariant`, `--mmap`.
- **Cache builder** (`xsess_cache.py`): `--picks`, `--sealed`,
  `--eval_subjects`, `--hidden_labelled`.
- **Release tools:**
  - `scripts/train_sealed.sh`. Its env is documented in the header: SPLIT,
    TEST_SUBJECTS, BLEND_W, WCV, CHANS, RECIPE_*, DATASET, GATE, FORCE_ZIP,
    HARNESS_FROM, RUN_NAME, STOP_AFTER, SUBMISSION_DIR.
  - `scripts/release_ablations.sh`.
  - `analysis/release_summarize.py` (the pre-registered rules).
  - `analysis/release_eda.py`.
  - `scripts/monitor_loop.sh`.
- **Solver** (`solvers/bci_decoding/riemann_sealed.py`), new opt-ins:
  - `blend_w="auto"`;
  - `chans`;
  - `align="subject_context"`;
  - channel-name check at load;
  - fast shrinkage LDA above 4,000 features;
  - float32 memory path.
- **Online-order study:** `analysis/sealed_stream.py`; results in
  `logs/sprint0928_p3/`.
- **Thesis docs fixes:** branch `docs/sprint0928-consistency` (853f9c5).

## Facts a resumer needs (previous sprint)

- **The release-day replica** is `SPLIT=calib:3 TEST_SUBJECTS=<the 10 fully
  labelled participants>`. `replica:3` is refused on non-mock caches.
- **Decide on that split only.** On `replica:3` the drifting EMG trap was
  invisible (+0.8); on `calib:3` it costs −3.7 to −4.2 points.
- **Expect 43 EEG channels at 120 Hz.** NeuralBench's default extractor keeps
  EEG channels only and resamples to 120 Hz. EMG/EOG reach `predict()` only if
  the organisers override `neuro.picks`.
- **The fast LDA (Cholesky)** is used only above 4,000 features, so every
  committed proxy number is reproduced bit for bit.
- **LOSO weight search: known bias.** Each held-out session's whitening
  reference includes its own windows: ~2 points of CV bias, never changing a
  weight so far, but it becomes oracle alignment when a (subject, context) pair
  lives in one calibration session. See RELEASE_DAY rule 4 for the interim
  guard. The final all-data run bakes the replica's weight; it does not use
  `auto`.
- **benchopt caches on parameters, not data.** Use `--no-cache` after a cache
  rebuild; train_sealed.sh passes it.
- **Line endings in `2026-competition`.** It is a Windows checkout: 81 of 93
  files are CRLF in the working tree, so plain WSL `git` can list CRLF-only
  diffs. Use `git -c core.autocrlf=true`.
- **WSL calls from the Bash tool:** start the heredoc with `exec 2>&1`.
  Otherwise stderr can overwrite the start of stdout in the captured output.

## What the user needs to do (previous sprint)

1. **Post the organiser question** (SEALED_RECIPE § 5 step 1). It now also asks
   about test-window **order and batching**: the online gain drops from
   +3.5 to +7.5 in recording order to +3.0 to +4.4 when windows arrive
   interleaved.
2. **A deployment-test upload during warm-up** (before 2026-10-25). No
   sklearn/pyriemann joblib has ever run on the scoring image. The command is in
   `SUBMISSIONS.md`.
3. **Review the new pre-registered rule 6 precedence** in RELEASE_DAY § 6
   (router-psdctx is kept over online unless the organisers confirm recording
   order). It was written tonight, before any release data.
4. Optional: review/merge the thesis docs branch `docs/sprint0928-consistency`
   (pushed to `personal`, no PR).
5. Optional: approve the REVE/LaBraM weights (still parked).
