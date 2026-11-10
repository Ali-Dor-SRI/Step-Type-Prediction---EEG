#!/usr/bin/env python
"""Sprint 2026-11-12: nested selection over several net configs AND the blend weight.

For lopo test person k, (config, w) maximises the mean cell score of X4f + config's
nets (seed mean) over the other nine lopo folds; `later` uses the pick on all ten.
Compares the nested-selected row with the base (sprint1109_compare's nested row of
BASE_DIR) by paired bootstrap over people. Reports what each fold picked.

    python sprint1112_nested_sel.py BASE_DIR DIR1 DIR2 ... [--grid 0.3_..._0.9]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_ensemble import mix  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402
from sprint1109_compare import DUMP, LOPO, nested, net_probs  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    grid = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    if "--grid" in sys.argv:
        grid = tuple(float(g) for g in sys.argv[sys.argv.index("--grid") + 1].split("_"))
        args = [a for a in args if a != sys.argv[sys.argv.index("--grid") + 1]]
    base_dir, dirs = args[0], args[1:]
    seeds = "33_34_35_36_37".split("_")
    rows = PC.load().rows
    b, bl, *_ = nested(rows, base_dir, seeds, (0.2, 0.3, 0.4))
    cell = {}
    for s in LOPO + ["later"]:
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        for di in dirs:
            Pn = net_probs(di, s, seeds)
            for w in grid:
                cell[(s, di, w)] = cells(rows, d["test_idx"], d["y_test"], mix(d["P_test"], Pn, w))
    opts = [(di, w) for di in dirs for w in grid]
    picks, out = [], []
    for s in LOPO:
        o = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO if t != s]).score.mean())
        picks.append((Path(o[0]).name, o[1]))
        out.append(cell[(s, *o)])
    ol = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO]).score.mean())
    arm, la = pd.concat(out), cell[("later", *ol)].score.mean()
    d, lo, hi, up = paired(b, arm)
    print(f"nested-selected over {len(dirs)} configs x {len(grid)} w: lopo {arm.score.mean():.4f} "
          f"later {la:.4f} (pick {Path(ol[0]).name}, w {ol[1]})")
    print(f"  fold picks {picks}")
    print(f"  vs base lopo {b.score.mean():.4f} / later {bl.score.mean():.4f}: lopo {d * 100:+.2f} "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {(la - bl.score.mean()) * 100:+.2f}")


if __name__ == "__main__":
    main()
