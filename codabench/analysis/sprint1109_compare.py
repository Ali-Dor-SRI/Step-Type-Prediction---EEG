#!/usr/bin/env python
"""Sprint 2026-11-09 W2/W3: X4f + ShallowNet ensembles of two net configs, compared.

For each config (a rows dir with net_shallow_<split>_s<seed>_o0.npz), the
nested ensemble: for lopo test person k, w in GRID maximises the mean cell
score over the other nine folds; `later` uses the w picked on all ten lopo
folds. Prints each config's nested row vs X4f, then arm vs base (paired
people bootstrap, sprint1106_tta_probe.paired).

    python sprint1109_compare.py BASE_DIR ARM_DIR [seeds] [grid]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_ensemble import mix  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

DUMP = Path.home() / "codabench/logs/sprint_1106/dump"
LOPO = [f"lopo{k}" for k in range(10)]


def net_probs(spec, split, seeds):
    """spec = DIR[:MODEL][+DIR[:MODEL]...] (model default shallow); members averaged
    with equal weight, each member = its seed mean (midweek 11-10 generalisation)."""
    out = []
    for m in spec.split("+"):
        d, _, model = m.partition(":")
        out.append(np.mean([np.load(Path(d) / f"net_{model or 'shallow'}_{split}_s{x}_o0.npz")
                            ["probas"] for x in seeds], axis=0))
    return np.mean(out, axis=0)


def nested(rows, rdir, seeds, grid):
    cell, x4f = {}, {}
    for s in LOPO + ["later"]:
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        Pn = net_probs(rdir, s, seeds)
        for w in (0.0,) + grid:
            cell[(s, w)] = cells(rows, d["test_idx"], d["y_test"], mix(d["P_test"], Pn, w))
    picks, out = [], []
    for s in LOPO:
        w = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO if t != s]).score.mean())
        picks.append(w)
        out.append(cell[(s, w)])
    wl = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO]).score.mean())
    base = pd.concat([cell[(s, 0.0)] for s in LOPO])
    return pd.concat(out), cell[("later", wl)], picks, wl, base, cell[("later", 0.0)]


def main(base_dir, arm_dir, seeds="33_34_35_36_37", grid="0.2_0.3_0.4"):
    rows = PC.load().rows
    seeds = seeds.split("_")
    grid = tuple(float(g) for g in grid.split("_"))
    res = {}
    for name, d in (("base", base_dir), ("arm", arm_dir)):
        L, T, picks, wl, X4l, X4t = nested(rows, d, seeds, grid)
        dd, lo, hi, up = paired(X4l, L)
        print(f"{name} ({d}): nested lopo {L.score.mean():.4f} vs X4f {100*dd:+.2f} "
              f"[{100*lo:+.2f}, {100*hi:+.2f}] up {up}; picks {picks}; later w={wl} "
              f"{T.score.mean():.4f} ({100*(T.score.mean()-X4t.score.mean()):+.2f})")
        res[name] = (L, T)
    dd, lo, hi, up = paired(res["base"][0], res["arm"][0])
    dt = res["arm"][1].score.mean() - res["base"][1].score.mean()
    print(f"arm - base: lopo {100*dd:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] up {up}; later {100*dt:+.2f}")
    adopt = dd >= 0.010 and lo > 0 and dt >= 0
    print(f"rule (Δ ≥ +1.0, CI lo > 0, later ≥ 0): {'ADOPTED' if adopt else 'not adopted'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
