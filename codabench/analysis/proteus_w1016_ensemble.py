#!/usr/bin/env python
"""Phase 3 ensemble of sprint 2026-10-16: Riemann candidate + EEGNet, equal weights.

P_ens = 0.5 * P_riemann + 0.5 * mean over EEGNet seeds of P_eegnet, per test
window (the streamed-replay probabilities saved by proteus_sweep.py), on every
lopo fold and `later`. No weight is tuned. Compared with the Riemann candidate
alone by the same paired people bootstrap as proteus_w1016_summary.py. Also
reports the EEGNet seed average alone. Decision rule (brief Phase 3): ensemble
adopted if >= 2 points on lopo with the CI excluding 0 and no loss on later.

    python ~/codabench/analysis/proteus_w1016_ensemble.py --riemann S64p \
        --eegnet E0f_s33p E0f_s34 E0f_s35 E0f_s36 E0f_s37 [--out RESULTS_p3.md]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_w1016_summary as S  # noqa: E402


def load(rows, name, protocol, fold):
    r = [x for x in rows if x["name"] == name and x["protocol"] == protocol
         and x["fold"] == fold]
    assert len(r) == 1, (name, protocol, fold, len(r))
    z = np.load(S.OUT / "probs" / (r[0]["key"].replace("|", "__") + ".npz"))
    assert "probas" in z.files, f"{name} {protocol} {fold}: no probabilities saved"
    return z


def cells(test_idx, y, p, rows_c):
    meta = rows_c.iloc[test_idx].reset_index(drop=True).assign(y=y, p=p)
    out = [(*k, S.bacc(g["y"].to_numpy(), g["p"].to_numpy()))
           for k, g in meta.groupby(["person", "session", "interface"])]
    return pd.DataFrame(out, columns=["person", "session", "interface", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--riemann", required=True)
    ap.add_argument("--eegnet", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows_c = PC.load().rows
    rows = S.load_rows(["A", "B"])
    res = {}
    for protocol, folds in (("lopo", range(10)), ("later", [None])):
        acc = {"riemann": [], "eegnet": [], "ensemble": []}
        for f in folds:
            zr = load(rows, a.riemann, protocol, f)
            ze = [load(rows, n, protocol, f) for n in a.eegnet]
            for z in ze:
                assert np.array_equal(z["test_idx"], zr["test_idx"])
            Pr = zr["probas"]
            Pe = np.mean([z["probas"] for z in ze], axis=0)
            y, idx = zr["y_true"], zr["test_idx"]
            assert np.array_equal(Pr.argmax(1), zr["y_pred"])
            for k, P in (("riemann", Pr), ("eegnet", Pe), ("ensemble", 0.5 * Pr + 0.5 * Pe)):
                acc[k].append(cells(idx, y, P.argmax(1), rows_c))
        res[protocol] = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    lines = [f"Riemann = {a.riemann}; EEGNet = mean of {', '.join(a.eegnet)} "
             f"({len(a.eegnet)} seeds); ensemble = 0.5 / 0.5 probabilities. Δ vs Riemann "
             "alone, paired bootstrap over people (10,000).", "",
             "| model | lopo (28) | Δ lopo [95 % CI] (up/10) | later (8) | Δ later [95 % CI] (up/2) |",
             "|---|---|---|---|---|"]
    for k in ("riemann", "eegnet", "ensemble"):
        cl = []
        for p in ("lopo", "later"):
            if k == "riemann":
                cl.append("—")
                continue
            d = S.paired(res[p]["riemann"], res[p][k])
            cl.append(f"{100 * d['diff']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] "
                      f"({d['up']}/{d['n_people']})")
        lines.append(f"| {k} | {res['lopo'][k].score.mean():.4f} | {cl[0]} | "
                     f"{res['later'][k].score.mean():.4f} | {cl[1]} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
