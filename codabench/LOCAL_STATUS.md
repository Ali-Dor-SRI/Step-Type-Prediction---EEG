# Local checkout status — 2026-11-12

## Git

- Our competition code belongs to the parent ML repository. The ignored
  `2026-competition/` directory is a separate upstream Git repository.
- Current branch: `codex/codabench-mac-setup`, tracking
  `personal/feat/codabench-track2` at `45f81c5` when inspected.
- The original `feat/codabench-track2` remains at `adce572`, preserving its
  unique November 9 documentation commit. No history was rewritten or pushed.
- Both remotes were fetched successfully. The organization branch is at
  `d1b5927`, five commits behind the personal branch.
- Local Git URL rewrites map the unavailable `github-personal` and
  `github-org` SSH aliases to HTTPS. Repository-local `core.autocrlf=true`
  recognizes the existing Windows checkout's line endings: all 436 apparent
  tracked modifications disappeared without rewriting working files.
- The pre-existing untracked `.claude/settings.local.json` was left alone.

## Latest recorded research state

These are the fetched project's records, not a fresh leaderboard check.
The November 12 update at the top of DIRECTIONS.md supersedes older counts
and upload instructions still present farther down that document and HANDOFF.md.

| Submission | Sealed score |
|---|---:|
| X4f, rescored after the marking fix | 0.710 |
| X4f + five ShallowNets, rescored | 0.735 |
| B3: aligned ShallowNets + AdaBN + causal X4f self-training | 0.755 |

B3 ran in 1,074 seconds, ranked 81/207 in the recorded snapshot, and left four
uploads. Its local gains transferred at about 55%. The November 12 follow-up
tested fifteen self-training variants without a qualifying improvement. The
known-person routing idea was dropped after its controls showed substantial
downside. Slot 4 is held pending a new direction. Read
`logs/sprint_1114/STATE.md` and `logs/sprint_1114/RESULTS.md` for the latest state.

## Native runtime and NAS data — configured later the same day

See `MAC_SETUP.md`. Start with `source ~/codabench/env.mac.sh`.

- Python 3.12.16 and all 191 recorded package versions are installed. Native
  Apple Silicon PyTorch wheels replace the Linux `+cpu` builds. Dependency
  compatibility checks pass.
- The validated upstream commit `442edcd` is restored in a separate clone;
  the copied old upstream checkout remains unchanged.
- The established NAS master is `/Volumes/Beast_PC/Projects/codabench`
  (Windows `Z:\Projects\codabench`). Its 4,769 existing files, about
  28.49 GiB, contained the older datasets but no PROTEUS or B3 artifacts.
- The public PROTEUS release downloaded directly to that NAS in 75 seconds.
  The official extraction and window caches also reside on NAS. Rebuilding
  all 2,655 windows took 97.5 seconds, including extraction.
- Compatibility symlinks `~/codabench` and `~/neuralbench` support the existing
  scripts. The native entry point checks the NAS destination before use.
- The exact B3 archive and the newer untracked per-split research artifacts
  still require transfer from Windows/WSL. The archive is
  `ens_x4f_shallow_al_stC_all_2026-11-10.zip`, expected SHA-256
  `14541a30357f00c05fff261cbd5bea6839ec031f06189ff077997b9319cfc88b`.
  The GitHub releases API returned no downloadable release assets.

## Checks performed

- Both GitHub remotes accessible and their branch refs fetched.
- All 107 tracked Codabench Python files parsed without syntax errors.
- All six existing submission ZIPs passed CRC checks; the November 6 ensemble
  matched its recorded full SHA-256 (`1f6589b8…`).
- Native ShallowNet inference passed on a synthetic input. No training,
  leaderboard actions, or uploads were performed. The existing full self-test
  passed all 349 checks in 49.9 seconds; results are recorded in
  `logs/mac_setup/selftest_mac.log`.
