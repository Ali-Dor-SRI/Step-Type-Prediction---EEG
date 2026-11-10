#!/usr/bin/env python
"""Sprint 2026-11-12 Q4: other classifiers on the dumped (streamed) X4f features.

Per split: fit on F_train / y_train of logs/sprint_1106/dump/X4f__<split>.npz,
predict the streamed F_test (the features exactly as X4f's classifier saw them).
Kinds:
  svm   StandardScaler + SVC(rbf, C=1, probability) (Platt; seed 0)
  bag   bagging over training people: one LR (X4f's settings) per leave-2-people-out bag
        (all C(9,2) = 36 bags; later: C(10,2) = 45), probabilities averaged
  plda  StandardScaler + PCA(256) + shrinkage LDA
  pens  person-expert mix: one LR per training person, probabilities averaged (uniform)
Writes logs/sprint_1112/q4/<kind>_<split>.npz (probas, y, test_idx); stack with
sprint1112_stack.py.

    python sprint1112_q4.py --kind svm
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_tta_probe import cells  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
OUT = CB / "logs/sprint_1112/q4"
SPLITS = [("lopo", k) for k in range(10)] + [("later", None)]


def log(m):
    print(f"{time.strftime('%T')} {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["svm", "bag", "plda", "pens"])
    ap.add_argument("--splits", default="all")
    a = ap.parse_args()
    c = PC.load()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for proto, fold in SPLITS:
        s = f"lopo{fold}" if proto == "lopo" else proto
        if a.splits != "all" and s not in a.splits.split(","):
            continue
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        base = joblib.load(DUMP / f"X4f__{proto}__{fold}.clf.joblib")
        Ftr, ytr, Fte, te = d["F_train"], d["y_train"], d["F_test"], d["test_idx"]
        tr = np.asarray(PS.split(proto, fold, c.rows).train)
        assert len(tr) == len(ytr) and np.array_equal(c.y[tr], ytr)
        ppl = c.rows.person.iloc[tr].to_numpy()
        if a.kind == "svm":
            P = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=0)
                              ).fit(Ftr, ytr).predict_proba(Fte)
        elif a.kind == "plda":
            P = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                              LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(Ftr, ytr).predict_proba(Fte)
        elif a.kind == "bag":
            Ps = []
            for out2 in itertools.combinations(np.unique(ppl), 2):
                m = ~np.isin(ppl, out2)
                Ps.append(clone(base).fit(Ftr[m], ytr[m]).predict_proba(Fte))
            P = np.mean(Ps, axis=0)
        else:
            P = np.mean([clone(base).fit(Ftr[ppl == p], ytr[ppl == p]).predict_proba(Fte)
                         for p in np.unique(ppl)], axis=0)
        np.savez_compressed(OUT / f"{a.kind}_{s}.npz", probas=P, y=d["y_test"], test_idx=te)
        res[s] = cells(c.rows, te, d["y_test"], P.argmax(1)).score.mean()
        log(f"{a.kind} {s}: {res[s]:.4f} (X4f {float(d['cell_mean']):.4f})")
    lo = [v for k, v in res.items() if k.startswith("lopo")]
    log(f"[row] q4 {a.kind} alone: lopo {np.mean(lo):.4f} later {res.get('later', float('nan')):.4f}")


if __name__ == "__main__":
    main()
