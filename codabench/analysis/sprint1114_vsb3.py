#!/usr/bin/env python
"""Sprint 2026-11-14: nested re-selection over probe configs x w, compared with B3.

Base = B3 (q5_q0.85_l5.0_s20_t20_f16_C0.03, w nested over the grid). Option set =
{B3} U given names, each x w (re-selected nested: lopo test person k uses the other
nine lopo folds; `later` uses the all-ten pick). Use --no-b3-option to exclude B3 from
the option set (per-arm comparator). npz: logs/sprint_1112/q5/<name>_<split>.npz.

    python sprint1114_vsb3.py NAME1 NAME2 ... [--grid 0.4_0.5_0.6_0.7] [--no-b3-option]
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
B3 = "q5_q0.85_l5.0_s20_t20_f16_C0.03"


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def main():
    args = sys.argv[1:]
    grid = (0.4, 0.5, 0.6, 0.7)
    if "--grid" in args:
        i = args.index("--grid")
        grid = tuple(float(g) for g in args[i + 1].split("_"))
        args = args[:i] + args[i + 2:]
    b3opt = "--no-b3-option" not in args
    args = [a for a in args if a != "--no-b3-option"]
    rows = PC.load().rows
    cell = {}
    for s in LOPO + ["later"]:
        for n in dict.fromkeys(args + [B3]):
            z = np.load(Q5 / f"{n}_{s}.npz")
            te, y = z["test_idx"], z["y"]
            for w in grid:
                cell[(s, n, w)] = cells(rows, te, y, ((1 - w) * lg(z["probas"]) + w * lg(z["Pn"])).argmax(1))

    def nested(opts):
        out, picks = [], []
        for s in LOPO:
            o = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO if t != s]).score.mean())
            out.append(cell[(s, *o)])
            picks.append(o)
        ol = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO]).score.mean())
        return pd.concat(out), cell[("later", *ol)].score.mean(), picks, ol
    B, bl, _, _ = nested([(B3, w) for w in grid])
    names = list(dict.fromkeys(([B3] if b3opt else []) + args))
    A, al, picks, ol = nested([(n, w) for n in names for w in grid])
    dd, lo, hi, up = paired(B, A)
    dl = (al - bl) * 100
    v = "ADOPT" if dd * 100 >= 1 and lo * 100 > 0 and dl >= -0.5 else (
        "lean" if dd > 0 and lo * 100 > -0.5 and dl >= -0.5 else "no gain")
    print(f"nested over {len(names)} configs x {len(grid)} w: lopo {A.score.mean():.4f} later {al:.4f} "
          f"(all-ten pick {(ol[0].replace('q5_', '')[:30], ol[1])}) vs B3 {B.score.mean():.4f} / {bl:.4f}: "
          f"{dd * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {dl:+.2f} -> {v}")
    print("  fold picks:", [(p[0].replace("q5_", "")[:30], p[1]) for p in picks])


if __name__ == "__main__":
    main()
