#!/usr/bin/env python
"""Dump the streamed features of a Riemann-Sealed config (sprint 2026-11-06,
prerequisite of the weekend's label-free test-time adaptation probes, W2).

Fits one config on a protocol split exactly as ``proteus_sweep.py`` does (same
``fit_and_score``), then replays every test stream as the contract does (fresh
deep copy + ``reset_state()``, one window per call) with the fitted classifier
wrapped in a recorder, so each window's feature vector *as the classifier sees
it* (after stream whitening, fcenter / fscale) is saved in stream order. The
classifier fit is wrapped the same way to save the training features and
labels it was fitted on. Nothing in the solver changes; the replayed
probabilities must equal the classifier's own (checked) and their argmax the
streamed y_pred (checked by ``stream_probas``' rule).

    python sprint1106_dump.py --name X4f --set <params> --protocols lopo later

Output: ``logs/sprint_1106/dump/<name>__<protocol>__<fold>.npz`` with F_test,
P_test, y_test, stream_id, test_idx, F_train, y_train, and the classifier
(``clf.joblib``) next to it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_sweep as SW  # noqa: E402
import proteus_splits as PS  # noqa: E402
import proteus_cache as PC  # noqa: E402

OUT = Path.home() / "codabench/logs/sprint_1106/dump"


class Recorder:
    """Wraps a fitted classifier; records every input of predict_proba."""

    def __init__(self, clf):
        self.clf, self.F = clf, []

    def predict_proba(self, F):
        self.F.append(np.array(F, copy=True))
        return self.clf.predict_proba(F)

    def __getattr__(self, k):
        return getattr(self.clf, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--solver", default="riemann_sealed")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--protocols", nargs="+", default=["lopo", "later"])
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-offsets", default="")
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        params[k] = SW.parse_value(v)
    lo, hi = (int(v) for v in a.folds.split("-")) if "-" in a.folds else (int(a.folds),) * 2
    jobs = [(p, f) for p in a.protocols for f in (range(lo, hi + 1) if p == "lopo" else [None])]
    Solver = SW.load_solver_class(a.solver)
    mod = sys.modules[Solver.__module__]
    rec = {}
    for fn in ("_fit_lda", "_fit_logreg"):
        orig = getattr(mod, fn)

        def wrapped(F, y, *args, _orig=orig, **kw):
            # the last fit call is the pooled classifier (the router's LDAs come first)
            rec["F"] = np.array(F, copy=True)
            rec["y"] = np.array(y, copy=True)
            return _orig(F, y, *args, **kw)
        setattr(mod, fn, wrapped)
    offs = SW.parse_offsets(a.train_offsets)
    cache = PC.load()
    OUT.mkdir(parents=True, exist_ok=True)
    from benchmark_utils.streaming import fresh_model
    for protocol, fold in jobs:
        out = OUT / f"{a.name}__{protocol}__{fold}.npz"
        if out.exists():
            SW.log(f"skip {out.name}")
            continue
        rec.clear()
        model, sc, y_true, y_pred, info, fit_s = SW.fit_and_score(
            Solver, params, protocol, fold, a.seed, offs, a.train_offsets.endswith(":sep"))
        te = np.asarray(PS.split(protocol, fold).test)
        X = np.asarray(cache.X[te])
        sid = np.asarray(info["stream_id"])
        F, P = [], []
        with SW.one_thread():
            for s in np.unique(sid):
                m = np.where(sid == s)[0]
                f = fresh_model(model)
                r = Recorder(f.parts["lda"])
                f.parts["lda"] = r
                for i in m:
                    P.append(np.asarray(f.predict_proba(X[i:i + 1])).reshape(-1))
                F.extend(np.concatenate(r.F))
        order = np.concatenate([np.where(sid == s)[0] for s in np.unique(sid)])
        Ft = np.empty((len(te), len(F[0])))
        Pt = np.empty((len(te), len(P[0])))
        Ft[order], Pt[order] = np.asarray(F), np.asarray(P)
        if rec["F"].shape[1] != Ft.shape[1]:
            raise RuntimeError(f"training features {rec['F'].shape} are not the classifier's")
        if not np.allclose(model.parts["lda"].predict_proba(Ft), Pt, atol=1e-9):
            raise RuntimeError("recorded features do not reproduce the probabilities")
        if not np.array_equal(Pt.argmax(1), np.asarray(y_pred).reshape(-1)):
            raise RuntimeError("replay argmax != streamed y_pred")
        np.savez_compressed(out, F_test=Ft, P_test=Pt, y_test=np.asarray(y_true), stream_id=sid,
                            test_idx=te, F_train=rec["F"], y_train=rec["y"],
                            cell_mean=sc["balanced_accuracy_cell_mean"])
        import joblib
        joblib.dump(model.parts["lda"], OUT / f"{a.name}__{protocol}__{fold}.clf.joblib")
        SW.log(f"[dump] {out.name}: cell_mean={sc['balanced_accuracy_cell_mean']:.6f} "
               f"F_test {Ft.shape} F_train {rec['F'].shape} fit={fit_s:.0f}s")


if __name__ == "__main__":
    main()
