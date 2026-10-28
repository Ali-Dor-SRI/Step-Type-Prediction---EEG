#!/usr/bin/env python
"""Sprint 2026-10-28 Phase 3: nested check of the whole selection path
F64 -> fb6 -> C2f -> (C2f+F) -> X4f -> (X4f+F) (brief § 5, rule fixed 18:40 on
2026-10-28 before any Phase 1 number).

For each outer lopo fold k (person P(k+1) held out), the inner score of a
config = cell mean over its 9 inner folds ("nest" rows "k-j"). Walking the
configs in the order above, starting from pick = F64, a config replaces the
pick when its inner score is >= pick + bar and >= F64 + 2.0 points, with
bar = 2.0 if it adds a task-driven block (roi / asym / asymabs / burst) that
the pick does not have, else 1.0. The chosen config's outer score on person k
is its lopo row for fold k. The same walk on the full lopo (10 people, cell
means) gives the "selected" config. Reports nested lopo (28 cells) vs selected;
flags over-selection when nested < selected - 3 points.

    python ~/codabench/analysis/sprint1028_nested.py --configs F64 fb6 C2f C2f+X X4f X4f+X \
        [--out RESULTS_nested.md]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_w1016_summary as S  # noqa: E402

LOPO_LANES = ["B", "G1", "J2", "K2", "P1", "Q1", "R1", "P2", "Q2"]
NEST_LANES = ["N3", "NF1", "NF2", "NF3", "NF4", "NF5", "NF6"]


def has_new(cand, pick, extra):
    return bool(extra.get(cand, set()) - extra.get(pick, set()))


def walk(scores, configs, extra):
    pick = configs[0]
    for c in configs[1:]:
        if scores.get(c) is None:
            continue
        bar = 0.02 if has_new(c, pick, extra) else 0.01
        if scores[c] >= scores[pick] + bar and scores[c] >= scores[configs[0]] + 0.02:
            pick = c
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--blocks", nargs="*", default=[],
                    help="config=block1,block2 for configs that carry task-driven blocks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    extra = {c: set() for c in a.configs}
    for s in a.blocks:
        c, b = s.split("=")
        extra[c] = set(b.split(","))
    cr = PC.load().rows
    lopo = {}
    for r in S.load_rows(LOPO_LANES):
        if r["protocol"] == "lopo" and r["name"] in a.configs:
            lopo.setdefault(r["name"], {})[r["fold"]] = r
    nest = {}
    for r in S.load_rows(NEST_LANES):
        if r["protocol"] == "nest" and r["name"] in a.configs:
            nest.setdefault(r["name"], {})[r["fold"]] = r
    full = {}
    for c in a.configs:
        rs = lopo.get(c, {})
        full[c] = (pd.concat([S.cells_of(rs[f], cr) for f in range(10)]).score.mean()
                   if len(rs) == 10 else None)
    selected = walk(full, a.configs, extra)
    lines = ["| outer fold | " + " | ".join(f"inner {c}" for c in a.configs)
             + " | chosen | outer score |", "|---|" + "---|" * (len(a.configs) + 2)]
    cells, picks = [], []
    for k in range(10):
        sc = {}
        for c in a.configs:
            rs = [v for f, v in nest.get(c, {}).items() if str(f).startswith(f"{k}-")]
            sc[c] = (pd.concat([S.cells_of(r, cr) for r in rs]).score.mean()
                     if len(rs) == 9 else None)
        if sc[a.configs[0]] is None:
            lines.append(f"| {k} | incomplete |")
            continue
        pick = walk(sc, a.configs, extra)
        picks.append(pick)
        oc = S.cells_of(lopo[pick][k], cr)
        cells.append(oc)
        lines.append(f"| {k} (P{k + 1:02d}) | " + " | ".join(
            "—" if sc[c] is None else f"{sc[c]:.4f}" for c in a.configs)
            + f" | {pick} | {oc.score.mean():.4f} |")
    head = ["Full-lopo cell means: " + ", ".join(
        f"{c} {'—' if full[c] is None else f'{full[c]:.4f}'}" for c in a.configs),
        f"Selected on full lopo by the walk: **{selected}** ({full[selected]:.4f})."]
    if len(cells) == 10:
        nested = pd.concat(cells).score.mean()
        flag = "OVER-SELECTION" if nested < full[selected] - 0.03 else "no over-selection"
        head.append(f"Nested lopo (28 cells): **{nested:.4f}** vs selected "
                    f"{full[selected]:.4f}: optimism {100 * (full[selected] - nested):+.2f} "
                    f"points -> **{flag}** (flag at > 3). Picks: "
                    + ", ".join(f"{p} x{picks.count(p)}" for p in dict.fromkeys(picks)))
    else:
        head.append(f"Nested incomplete: {len(cells)}/10 outer folds.")
    text = "\n".join(head + [""] + lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
