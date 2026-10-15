#!/usr/bin/env python
"""Summarise proteus_sweep.py rows: per-arm cell means and paired bootstraps.

Sprint 2026-10-16 (brief ``prompts/2026-10-16_proteus_weekend_model.md``, rules
v2 of RELEASE_DAY § 5). Cells are rebuilt from the saved predictions
(``probs/<key>.npz``) as (person, session, interface), so `lopo` pools its 10
folds into 28 cells and `later` gives 8. For each arm vs a base: the difference
of cell means, a paired bootstrap over **people** (10,000 resamples; a resampled
person brings all of its cells), its 95 % CI, and the number of people whose
mean cell score went up. Also the cell mean on calibration-like vs online runs
is not separable from the cells (a cell holds both): reported per interface.

    python ~/codabench/analysis/proteus_w1016_summary.py --base B1 [--arms S16 S32]
        [--lanes A B] [--out RESULTS_p1.md]

Rows are de-duplicated by key (the newest wins); an arm is summarised only when
every lopo fold and `later` are present (otherwise listed as incomplete).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402

OUT = Path(os.environ.get("SWEEP_OUT", Path.home() / "codabench/logs/proteus_w1016"))
INTERFACES = ("Graz", "BrainHero")


def bacc(y, p, k=3):
    rec = [np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]
    return float(np.mean(rec))


def load_rows(lanes):
    rows = {}
    for lane in lanes:
        f = OUT / f"{lane}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[r["key"]] = r
    return list(rows.values())


def cells_of(row, cache_rows):
    """DataFrame (person, session, interface, score) of one row's test cells."""
    z = np.load(OUT / "probs" / (row["key"].replace("|", "__") + ".npz"))
    idx = z["test_idx"]
    meta = cache_rows.iloc[idx].reset_index(drop=True)
    assert np.array_equal(z["y_true"], meta["label"].to_numpy()), row["key"]
    meta = meta.assign(y=z["y_true"], p=z["y_pred"])
    out = [(pe, se, it, bacc(g["y"].to_numpy(), g["p"].to_numpy()))
           for (pe, se, it), g in meta.groupby(["person", "session", "interface"])]
    return pd.DataFrame(out, columns=["person", "session", "interface", "score"])


def arm_cells(rows, name, protocol, cache_rows):
    rs = [r for r in rows if r["name"] == name and r["protocol"] == protocol]
    if protocol == "lopo":
        if sorted(r["fold"] for r in rs) != list(range(10)):
            return None
    elif len(rs) != 1:
        return None
    return pd.concat([cells_of(r, cache_rows) for r in rs], ignore_index=True)


def paired(base, arm, n_boot=10000, seed=0):
    m = base.merge(arm, on=["person", "session", "interface"], suffixes=("_b", "_a"))
    m["d"] = m["score_a"] - m["score_b"]
    people = m["person"].unique()
    per = {p: m.loc[m.person == p, "d"].to_numpy() for p in people}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(people, len(people), replace=True)
        boots[i] = np.concatenate([per[p] for p in pick]).mean()
    up = sum(per[p].mean() > 0 for p in people)
    return dict(diff=float(m["d"].mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), up=int(up), n_people=len(people),
                n_cells=len(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--lanes", nargs="+", default=["A", "B"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cache_rows = PC.load().rows
    rows = load_rows(a.lanes)
    names = a.arms or sorted({r["name"] for r in rows})
    if a.base not in names:
        names = [a.base] + names
    lines = [f"Base: **{a.base}**. Cell-averaged balanced accuracy; diff = arm - base "
             "(points ×100); CI = paired bootstrap over people (10,000); up = people "
             "whose mean cell score rose.", "",
             "| arm | lopo (28) | Graz | BrainHero | Δ lopo [95 % CI] (up/10) | later (8) | "
             "Δ later [95 % CI] (up/2) | fit s |",
             "|---|---|---|---|---|---|---|---|"]
    base = {p: arm_cells(rows, a.base, p, cache_rows) for p in ("lopo", "later")}
    for name in names:
        c = {p: arm_cells(rows, name, p, cache_rows) for p in ("lopo", "later")}
        if c["lopo"] is None or c["later"] is None:
            lines.append(f"| {name} | incomplete | | | | | | |")
            continue
        fit = np.mean([r["fit_s"] for r in rows if r["name"] == name])
        cell = []
        for p in ("lopo", "later"):
            if name == a.base or base[p] is None:
                cell.append("—")
                continue
            d = paired(base[p], c[p])
            cell.append(f"{100 * d['diff']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] "
                        f"({d['up']}/{d['n_people']})")
        li = c["lopo"]
        lines.append(
            f"| {name} | {li.score.mean():.4f} | "
            f"{li[li.interface == 'Graz'].score.mean():.4f} | "
            f"{li[li.interface == 'BrainHero'].score.mean():.4f} | {cell[0]} | "
            f"{c['later'].score.mean():.4f} | {cell[1]} | {fit:.0f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
