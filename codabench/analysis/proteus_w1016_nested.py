#!/usr/bin/env python
"""Phase 5 of sprint 2026-10-16: how optimistic is choosing on `lopo`?

For each outer fold k (person P(k+1) held out), the Phase-1 rule is re-applied
using only the inner lopo over the other nine people (N.jsonl, protocol
"nest", folds "k-j"): inner score of an arm = cell mean over its 9 inner folds'
cells. Rule (as Phase 1, rule 6): the best stream arm by inner score is taken
if it beats B1 by >= 2 points (no CI at this level: 9 people); ties within 0.1
point go to the smaller N (16 < 64 < all). The chosen arm's outer score on
person k comes from A.jsonl (the Phase-1 lopo rows, same configs). Reports the
nested lopo cell mean (28 cells) beside the selected one (S64) and the choice
per fold.

    python ~/codabench/analysis/proteus_w1016_nested.py [--out RESULTS_p5.md]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_w1016_summary as S  # noqa: E402

ARMS = ("B1", "S16", "S64", "S0")
ORDER = {"S16": 0, "S64": 1, "S0": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows_c = PC.load().rows
    inner = [r for r in S.load_rows(["N"]) if r["protocol"] == "nest"]
    outer = [r for r in S.load_rows(["A"]) if r["protocol"] == "lopo"]
    lines = ["| outer fold (person) | inner B1 | inner S16 | inner S64 | inner S0 | chosen | "
             "outer score of chosen | outer S64 |", "|---|---|---|---|---|---|---|---|"]
    chosen_cells, s64_cells, complete = [], [], True
    for k in range(10):
        sc = {}
        for arm in ARMS:
            rs = [r for r in inner if r["name"] == arm and r["fold"].startswith(f"{k}-")]
            if len(rs) != 9:
                sc[arm] = None
                continue
            sc[arm] = pd.concat([S.cells_of(r, rows_c) for r in rs]).score.mean()
        if any(v is None for v in sc.values()):
            complete = False
            lines.append(f"| {k} (P{k + 1:02d}) | incomplete | | | | | | |")
            continue
        best = max(ORDER, key=lambda n: (round(sc[n], 3), -ORDER[n]))
        ties = [n for n in ORDER if sc[best] - sc[n] <= 0.001]
        best = min(ties, key=ORDER.get)
        pick = best if sc[best] - sc["B1"] >= 0.02 else "B1"
        oc = {}
        for arm in (pick, "S64"):
            r = [x for x in outer if x["name"] == arm and x["fold"] == k]
            oc[arm] = S.cells_of(r[0], rows_c)
        chosen_cells.append(oc[pick])
        s64_cells.append(oc["S64"])
        lines.append(f"| {k} (P{k + 1:02d}) | " + " | ".join(f"{sc[x]:.4f}" for x in ARMS)
                     + f" | {pick} | {oc[pick].score.mean():.4f} | {oc['S64'].score.mean():.4f} |")
    if complete:
        n = pd.concat(chosen_cells).score.mean()
        s = pd.concat(s64_cells).score.mean()
        lines += ["", f"**Nested lopo cell mean (28 cells): {n:.4f}**; selected S64 on all folds: "
                  f"{s:.4f}; optimism = {100 * (s - n):+.2f} points. Rule: nested < selected − 3 "
                  "points → flag the candidate as over-selected."]
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
