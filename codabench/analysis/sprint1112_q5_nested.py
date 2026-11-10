#!/usr/bin/env python
"""Sprint 2026-11-12: nested re-selection over Q5 self-training configs (and w).

For lopo test person k, (config, w) maximises the mean cell score over the other
nine lopo folds; `later` uses the all-ten pick. Base = B1 (X4f + the same AdaBN nets,
w nested). Every config's npz (logs/sprint_1112/q5/<name>_<split>.npz) holds the
adapted X4f probabilities (`probas`), the original (`Px`) and the nets (`Pn`).

    python sprint1112_q5_nested.py NAME1 NAME2 ... [--grid 0.4_0.5_0.6_0.7]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

Q5 = Path.home() / "codabench/logs/sprint_1112/q5"
LOPO = [f"lopo{k}" for k in range(10)]


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def main():
    args = sys.argv[1:]
    grid = (0.4, 0.5, 0.6, 0.7)
    if "--grid" in args:
        i = args.index("--grid")
        grid = tuple(float(g) for g in args[i + 1].split("_"))
        args = args[:i] + args[i + 2:]
    rows = PC.load().rows
    cell = {}
    for s in LOPO + ["later"]:
        for n in args:
            z = np.load(Q5 / f"{n}_{s}.npz")
            te, y = z["test_idx"], z["y"]
            for w in grid:
                cell[(s, n, w)] = cells(rows, te, y, ((1 - w) * lg(z["probas"]) + w * lg(z["Pn"])).argmax(1))
                cell[(s, "B1", w)] = cells(rows, te, y, ((1 - w) * lg(z["Px"]) + w * lg(z["Pn"])).argmax(1))

    def nested(opts):
        out, picks = [], []
        for s in LOPO:
            o = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO if t != s]).score.mean())
            out.append(cell[(s, *o)])
            picks.append(o)
        ol = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO]).score.mean())
        return pd.concat(out), cell[("later", *ol)].score.mean(), picks, ol
    B, bl, _, _ = nested([("B1", w) for w in grid])
    A, al, picks, ol = nested([(n, w) for n in args for w in grid])
    dd, lo, hi, up = paired(B, A)
    print(f"nested over {len(args)} Q5 configs x {len(grid)} w: lopo {A.score.mean():.4f} later {al:.4f} "
          f"(all-ten pick {ol}) vs B1 {B.score.mean():.4f} / {bl:.4f}: {dd * 100:+.2f} [{lo * 100:+.2f}, "
          f"{hi * 100:+.2f}] up {up}; later {(al - bl) * 100:+.2f}")
    print("  fold picks:", [(p[0].replace("q5_", "")[:22], p[1]) for p in picks])


if __name__ == "__main__":
    main()
