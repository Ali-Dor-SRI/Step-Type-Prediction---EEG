#!/usr/bin/env python
"""Sprint 2026-11-12: nested stacking of a new probability source onto a base.

base  = the slot-2 ensemble log-score  (1 - 0.3) log P_X4f + 0.3 log P_shallow
        (P_shallow = mean of seeds 33-37 in logs/sprint_1106/rows), or any base
        given as --base "x4f+DIR:MODEL@w" (w fixed, the nested pick of that base);
arm   = (1 - w) * base + w * log P_new, w from GRID chosen nested: for lopo test
        person k, w maximises the mean cell score over the other nine lopo
        folds; `later` uses the w picked on all ten lopo folds (as
        sprint1109_compare). P_new comes from npz files <DIR>/<prefix>_<split>.npz
        with `probas` (test rows in test_idx order) and `test_idx`.
Prints base row, arm row, arm - base (paired bootstrap over people, n up) and
the verdict by brief § 3 (adopt / lean / no gain), plus the arm alone.

    python sprint1112_stack.py --new DIR:PREFIX [--new DIR2:PREFIX2] [--grid 0.1_0.2_0.3_0.4_0.5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
LOPO = [f"lopo{k}" for k in range(10)]
SPLITS = LOPO + ["later"]


def dump(s):
    proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
    return np.load(DUMP / f"X4f__{proto}__{fold}.npz")


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def net_mean(d, model, s, seeds):
    return np.mean([np.load(Path(d) / f"net_{model}_{s}_s{x}_o0.npz")["probas"] for x in seeds], axis=0)


def base_scores(spec, s, seeds):
    """spec: 'x4f' or 'x4f+DIR:MODEL@w[+DIR:MODEL@w]' (each net term mixed with its w on top of the previous)."""
    L = lg(dump(s)["P_test"])
    for term in spec.split("+")[1:]:
        dm, _, w = term.partition("@")
        d, _, model = dm.partition(":")
        L = (1 - float(w)) * L + float(w) * lg(net_mean(d, model or "shallow", s, seeds))
    return L


def new_scores(specs, s, te):
    Ls = []
    for spec in specs:
        d, _, prefix = spec.partition(":")
        z = np.load(Path(d) / f"{prefix}_{s}.npz")
        assert np.array_equal(z["test_idx"], te), f"{spec} {s}: test_idx mismatch"
        Ls.append(lg(z["probas"]))
    return np.mean(Ls, axis=0)


def verdict(d, lo, dl):
    if d >= 1.0 and lo > 0 and dl >= 0:
        return "ADOPT"
    if d > 0 and lo > -0.5 and dl >= -0.5:
        return "lean"
    return "no gain"


def run(base, news, grid, seeds, label="arm", quiet=False):
    rows = PC.load().rows
    cell = {}
    for s in SPLITS:
        d = dump(s)
        te = d["test_idx"]
        Lb = base_scores(base, s, seeds)
        Ln = new_scores(news, s, te)
        for w in (0.0, 1.0) + grid:
            cell[(s, w)] = cells(rows, te, d["y_test"], ((1 - w) * Lb + w * Ln).argmax(1))
    picks, out = [], []
    for s in LOPO:
        w = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO if t != s]).score.mean())
        picks.append(w)
        out.append(cell[(s, w)])
    wl = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO]).score.mean())
    arm, b = pd.concat(out), pd.concat([cell[(s, 0.0)] for s in LOPO])
    alone = pd.concat([cell[(s, 1.0)] for s in LOPO])
    la, lb, l1 = cell[("later", wl)].score.mean(), cell[("later", 0.0)].score.mean(), cell[("later", 1.0)].score.mean()
    d, lo, hi, up = paired(b, arm)
    dl = (la - lb) * 100
    v = verdict(d * 100, lo * 100, dl)
    msg = (f"{label}: base lopo {b.score.mean():.4f} later {lb:.4f} | new alone lopo {alone.score.mean():.4f} "
           f"later {l1:.4f} | nested lopo {arm.score.mean():.4f} later {la:.4f} (w later {wl}); "
           f"picks {picks}\n  arm - base: lopo {d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}/10; "
           f"later {dl:+.2f} -> {v}")
    if not quiet:
        print(msg, flush=True)
    return dict(d=d * 100, lo=lo * 100, hi=hi * 100, dl=dl, verdict=v, picks=picks, wl=wl,
                lopo=arm.score.mean(), later=la, arm_cells=arm, base_cells=b, msg=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"x4f+{CB}/logs/sprint_1106/rows:shallow@0.3")
    ap.add_argument("--new", action="append", required=True)
    ap.add_argument("--grid", default="0.1_0.2_0.3_0.4_0.5")
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()
    run(a.base, a.new, tuple(float(g) for g in a.grid.split("_")), a.seeds.split("_"), a.label)


if __name__ == "__main__":
    main()
