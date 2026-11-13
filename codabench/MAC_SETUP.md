# Native Mac setup

Use bash or zsh:

```bash
source ~/codabench/env.mac.sh
```

This uses Python 3.12.16, NeuralBench 0.4.0, and the upstream competition
snapshot at `442edcd`. All 191 package versions match
`config/venv040_freeze.txt`; the three PyTorch packages use native Mac wheels
instead of the Linux `+cpu` builds. This is a separate environment from the
copied Windows virtual environments.

## Storage

| Contents | Location |
|---|---|
| Source, small setup logs and native environment | `codabench/logs/mac_setup/` (environment and logs ignored by Git) |
| NAS master data folder | `/Volumes/Beast_PC/Projects/codabench` |
| Public PROTEUS recordings | `/Volumes/Beast_PC/Projects/codabench/neural_compet/Dreyer2026Proteus` |
| Official extraction cache | `/Volumes/Beast_PC/Projects/codabench/neural_compet/cache` |
| Rebuilt PROTEUS window cache | `/Volumes/Beast_PC/Projects/codabench/proteus_cache` |

The NAS root is the existing Windows `Z:\Projects\codabench` location,
`\\192.168.50.157\Beast_PC\Projects\codabench`, recorded in LOG.md on
September 25. Earlier datasets were already present there; they were reused.
PROTEUS and B3 were absent when inventoried on November 12.

`~/codabench` links to this checkout's competition folder. `~/neuralbench`
links to its native runtime. The runtime's `benchopt_data_040` links to the
NAS master data folder; this preserves the existing scripts' environment
checks. The native entry point refuses to proceed if that link or the NAS
is unavailable, preventing an accidental large download to local storage.
Keep the NAS mounted while using the data and window cache.

## Existing validation commands

```bash
source ~/codabench/env.mac.sh
python ~/codabench/analysis/proteus_cache.py check
python ~/codabench/analysis/proteus_selftest.py
```

The original Windows/WSL shell scripts still target their original
environment; use the native entry point above on this Mac. Cross-platform
training reproducibility must be checked separately from data integrity.

## Private artifacts still needed

GitHub has no release assets for this repository. The frozen B3 archive is
not tracked by Git and was absent from the established NAS data folder:

```text
ens_x4f_shallow_al_stC_all_2026-11-10.zip
SHA-256: 14541a30357f00c05fff261cbd5bea6839ec031f06189ff077997b9319cfc88b
```

Transfer the original archive from the Windows `codabench/submissions/`
folder to the NAS rather than retraining to recreate its exact weights.
For continuation of the existing experiment sweeps, also preserve the
untracked per-split weights and probability/probe files under
`logs/sprint_1110/`, `logs/sprint_1112/`, and `logs/sprint_1114/`.

## Verification — November 12

The full existing PROTEUS self-test passed **349/349 checks**, including exact
agreement between the rebuilt window cache and the official loader, raw-event
labels, split isolation, saved reference-model replay, and stream-state reset.
This validates the data and reference pipeline; B3 itself still awaits its
private artifact transfer.

| Operation | Start (EST) | Elapsed | Result |
|---|---|---|---|
| Public release download to NAS | 17:41:51 | 75.1 s | Complete, 1.48 GB release |
| Extraction + window cache build | 17:44:24 | 97.5 s | 2,655 × 41 × 480 float32 |
| Full existing self-test | 17:46:12 | 49.9 s | 349 passed, 0 failed |

The overall setup was estimated at 10–20 minutes and finished within that
estimate. Full logs are in `logs/mac_setup/`; the first loader attempts exposed
an unsupported CPU-limit override and macOS's requirement for a file-based
multiprocessing entry point. The successful run uses the original guarded
`proteus_cache.py` script and default extraction settings.
