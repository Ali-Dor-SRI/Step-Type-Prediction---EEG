#!/usr/bin/env python
"""Sprint 2026-10-28 Phase 0 unit check of the task-driven blocks (roi / asym /
burst) in solvers/bci_decoding/riemann_sealed.py, on all 2,655 cached windows.

Checks: block sizes (8 x 6 / 17 x 6 / 8 x 6 with the fb6 bands), every value
finite (the cache has +/- 4,000 uV excursions), AF3/AF4 in no block, one asym
value recomputed by hand. Then, descriptive only (decides nothing): the median
over people of each block's best per-feature Fisher ratio, as in the EDA.

    python ~/codabench/analysis/sprint1028_ub_unit.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402

FB6 = ["1to4", "4to8", "8to13", "13to20", "20to30", "30to45"]


def fisher(F, y):
    mu = F.mean(0)
    b = sum((y == k).mean() * (F[y == k].mean(0) - mu) ** 2 for k in np.unique(y))
    w = sum((y == k).mean() * F[y == k].var(0) for k in np.unique(y))
    return b / np.maximum(w, 1e-18)


def main():
    path = Path.home() / "codabench/solvers/bci_decoding/riemann_sealed.py"
    spec = importlib.util.spec_from_file_location("rs_unit", path)
    rs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rs
    spec.loader.exec_module(rs)

    c = PC.load()
    X = np.asarray(c.X, dtype=np.float64)
    y = np.asarray(c.y)
    names = list(c.meta["ch_names"])
    print(f"data X={X.shape}, classes={np.bincount(y).tolist()}, channels={len(names)}")
    parts = {"sfreq": float(c.meta["sfreq"]), "fb_bands": FB6,
             "ublocks": ["roi", "asym", "burst"], "ub_idx": rs._ub_index(names)}
    used = {names[i] for r in parts["ub_idx"]["rois"] for i in r}
    used |= {names[i] for i in parts["ub_idx"]["left"] + parts["ub_idx"]["right"]}
    fails = 0

    def check(label, ok, val=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}  {val}")

    check("AF3/AF4 in no block", not ({"AF3", "AF4"} & used), sorted(used & {"AF3", "AF4"}))
    F = rs._ub_features(parts, X)
    check("shape (2655, 48 + 102 + 48)", F.shape == (len(X), 198), F.shape)
    check("all finite", bool(np.isfinite(F).all()), int((~np.isfinite(F)).sum()))
    blocks = {"roi": F[:, :48], "asym": F[:, 48:150], "burst": F[:, 150:]}
    # asym by hand: C3 vs C4 in 8-13 Hz (band index 2, pair index 3)
    xb = np.concatenate(list(rs._band_chunks(X[:50], parts["sfreq"], "8to13")))
    i3, i4 = names.index("C3"), names.index("C4")
    hand = np.log((xb[:, i3] ** 2).mean(1) + 1e-12) - np.log((xb[:, i4] ** 2).mean(1) + 1e-12)
    got = blocks["asym"][:50, 2 * 17 + 3]
    check("asym C3-C4 8-13 Hz by hand", np.allclose(hand, got, atol=1e-10),
          f"max diff {np.abs(hand - got).max():.2e}")

    rows = c.rows
    print("\nDescriptive only: per block, median over people of the per-feature Fisher "
          "ratio (best feature / block median), features centred per session first")
    Fc = F.copy()
    for _, g in rows.groupby(["person", "session"]).groups.items():
        g = np.asarray(list(g))
        Fc[g] -= Fc[g].mean(0)
    for b, sl in (("roi", slice(0, 48)), ("asym", slice(48, 150)), ("burst", slice(150, 198))):
        per = np.stack([fisher(Fc[rows.person.values == p][:, sl],
                               y[rows.person.values == p]) for p in sorted(rows.person.unique())])
        med = np.median(per, 0)
        print(f"  {b:6s} best {med.max():.3f} (feature {int(med.argmax())}), median {np.median(med):.3f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
