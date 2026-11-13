# Resume status — November 13, 2026

**Ready to resume B3 evaluation and development on the Mac.** The Windows
transfer contains the frozen model and the saved experiment inputs, and the
existing prediction and selection checks reproduce their recorded results.
No new model improvement or competition upload was made during restoration.

## Restored artifacts

Imported 4,582 files, 12,958,795,583 bytes, from the completed Windows copy.
Every copied file was verified with SHA-256. Import took 591.6 seconds
(11:55:23 EST start), within the estimated 8–12 minutes.

NAS root: `/Volumes/Beast_PC/Projects/codabench/windows-transfer/codabench`.
Windows equivalent: `Z:\Projects\codabench\windows-transfer\codabench`.
The per-file receipt is `IMPORT_MANIFEST.json` in that NAS folder.

| Restored folder | Files | Contents |
|---|---:|---|
| `submissions` | 8 | Frozen B3 and earlier submission archives |
| `logs/sprint_1106` | 254 | Feature dumps, classifiers and neural outputs |
| `logs/sprint_1109` | 398 | Alignment references, split weights and ensemble exports |
| `logs/sprint_1110` | 185 | Alternative-family experiment records |
| `logs/sprint_1112` | 3,689 | AdaBN, self-training probes, all-trained weights and candidates |
| `logs/sprint_1114` | 48 | Follow-up probes and diagnostics |

The working checkout uses 805 artifact file/directory links to this NAS store.
Tracked source and result documents were preserved. Existing identical local
artifact files were retained in `logs/mac_setup/pre-transfer-backup-20261113-120530/`.
No conflicting local artifacts were found. Public recordings and the PROTEUS
cache remain in their existing NAS locations.

The transferred source matches Git commit `45f81c5`; no missing uncommitted
Codabench source was found in the inspected source directories. The missing
files were trained/generated artifacts unsuitable for ordinary Git storage.

## Frozen B3

Archive: `submissions/ens_x4f_shallow_al_stC_all_2026-11-10.zip`.

```text
SHA-256: 14541a30357f00c05fff261cbd5bea6839ec031f06189ff077997b9319cfc88b
```

Its NAS copy matches exactly. A verified inference export is also available
at `/Volumes/Beast_PC/Projects/codabench/artifacts/B3-14541a30/`.
The model is X4f plus five aligned ShallowNets, AdaBN prior 32, blend 0.6,
and causal logistic self-training (q 0.85, weight 5, start/every 20, C 0.03).

## Native Mac checks

- Split replay against saved Windows predictions: **ALL PASS**. Zero class
  differences over lopo0 (190 windows) and later (762 windows). Maximum adapted
  X4f probability differences were 2.26e-5 and 4.85e-5; net differences were
  2.42e-5 and 1.28e-5, inside the existing 1e-4 tolerance.
- Frozen archive streaming gates: **ALL PASS**. Streamed/batched predictions
  match on 755 windows; isolated-stream and reset-state checks pass. The
  in-process run was 17.7 ms/window during this restoration, not a new server
  benchmark or a new held-out accuracy estimate.
- Original selection over 16 saved configurations: reproduced **lopo 0.6819 /
  later 0.7346**, +2.04 [1.26, 3.34] points vs B1 on lopo, all-ten choice B3.
- Original selection script runs successfully with the default restored paths;
  it no longer requires the temporary transfer folder.
- A disposable CPU ShallowNet optimization step completed with finite loss and
  parameters. This checks the training backend, not cross-platform retraining
  reproducibility.
- The earlier public-data setup self-test passed 349/349 checks on November 12.

Logs: `logs/mac_setup/b3_split_parity_mac.log`, `b3_frozen_gates_mac.log`,
`b3_selection_reproduction_mac.log`, and `b3_resume_default_paths.log`.

Metric distinction: selecting only the already-fixed B3 blend gives lopo
0.6868 / later 0.7346 with both tested blend grids. The 0.6819 figure comes
from re-selecting among all 16 candidate configurations. This is a difference
in selection procedure, not a new gain or a Mac numerical discrepancy.

## Where development resumes

The latest recorded sealed result remains **0.755**, rank 81/207 in the
November 12 snapshot, with four uploads recorded as remaining. These records
were not refreshed from the live competition site during restoration.

The preceding sprint completed: fifteen follow-up variants did not beat B3;
the known-person routing line was also dropped. Slot 4 was held. There is no
unfinished winning arm to resume. Begin a new hypothesis-driven sprint with
the frozen B3 control, existing person/session splits and saved negative
results; retain the original selection procedure when comparing candidates.

```bash
source ~/codabench/env.mac.sh
python ~/codabench/analysis/sprint1114_vsb3.py
```

For another Mac/Linux checkout, restore code from Git, then connect the same
NAS artifacts with `python3 scripts/link_codabench_artifacts.py`. See
`GIT_WORKFLOW.md`; the native Python environment is configured separately.
