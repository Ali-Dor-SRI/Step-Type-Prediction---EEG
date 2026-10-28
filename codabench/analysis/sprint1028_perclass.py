#!/usr/bin/env python
"""Sprint 2026-10-28 mechanistic check (brief § 3; reported, never decisive):
per-class recall on lopo (mean over the 28 cells) of each arm vs its base.
Expected: asym / asymabs -> mi (and word via LIF), roi -> sub, burst -> word.

    python ~/codabench/analysis/sprint1028_perclass.py --lanes J2 K2 P1 Q1 R1 \
        --pairs C2f:C2f+roi X4f:X4f+roi ... [--out file.md]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_w1016_summary as S  # noqa: E402

CLASSES = ("mi", "sub", "word")


def recalls(rows, name, cache_rows):
    rs = [r for r in rows if r["name"] == name and r["protocol"] == "lopo"]
    if sorted(r["fold"] for r in rs) != list(range(10)):
        return None
    out = []
    for r in rs:
        z = np.load(S.OUT / "probs" / (r["key"].replace("|", "__") + ".npz"))
        meta = cache_rows.iloc[z["test_idx"]].reset_index(drop=True)
        meta = meta.assign(y=z["y_true"], p=z["y_pred"])
        for _, g in meta.groupby(["person", "session", "interface"]):
            y, p = g["y"].to_numpy(), g["p"].to_numpy()
            out.append([np.mean(p[y == k] == k) for k in range(3)])
    return np.mean(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="base:arm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cache_rows = PC.load().rows
    rows = S.load_rows(a.lanes)
    lines = ["| arm (vs base) | " + " | ".join(f"{c} recall (Δ)" for c in CLASSES) + " |",
             "|---|---|---|---|"]
    for pair in a.pairs:
        b, m = pair.split(":")
        rb, rm = recalls(rows, b, cache_rows), recalls(rows, m, cache_rows)
        if rb is None or rm is None:
            lines.append(f"| {m} (vs {b}) | incomplete | | |")
            continue
        lines.append(f"| {m} (vs {b}) | " + " | ".join(
            f"{rm[k]:.3f} ({100 * (rm[k] - rb[k]):+.1f})" for k in range(3)) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
