#!/usr/bin/env python
"""Sprint 2026-11-12: X4f's training features for the all-trained (sealed, 60a0148c) model.

Refits the X4f solver on protocol `all` exactly as proteus_sweep does, with the
module's _fit_logreg wrapped to record the features / labels its pooled classifier
is fitted on; checks the refit classifier equals the sealed zip's
(riemann_sealed.joblib parts["lda"]: scaler mean / scale and LR coef within 1e-9),
then saves Z = scaler.transform(F) as float16 + y to
logs/sprint_1112/x4f_all_trainZ.npz (input of the Q5 solver's refits).
"""

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_sweep as SW  # noqa: E402

CB = Path.home() / "codabench"
X4F_ZIP = CB / "logs/proteus_w1016/proteus_x4f_cand_all/riemann_sealed_proteus_x4f_all_2026-10-27.zip"
OUT = CB / "logs/sprint_1112/x4f_all_trainZ.npz"


def main():
    Solver = SW.load_solver_class("riemann_sealed_proteus_x4f")
    mod = sys.modules[Solver.__module__]
    rec = {}
    orig = mod._fit_logreg

    def wrapped(F, y, *a, **k):
        rec["F"], rec["y"] = np.array(F, copy=True), np.array(y, copy=True)
        return orig(F, y, *a, **k)
    mod._fit_logreg = wrapped
    model, sc, *_ = SW.fit_and_score(Solver, {}, "all", None)
    with zipfile.ZipFile(X4F_ZIP) as z:
        sealed = joblib.load(BytesIO(z.read("riemann_sealed.joblib")))
    a, b = model.parts["lda"], sealed["lda"]
    dm = float(np.abs(a.steps[0][1].mean_ - b.steps[0][1].mean_).max())
    ds = float(np.abs(a.steps[0][1].scale_ - b.steps[0][1].scale_).max())
    dc = float(np.abs(a.steps[-1][1].coef_ - b.steps[-1][1].coef_).max())
    print(f"refit vs sealed classifier: |d mean| {dm:.2e} |d scale| {ds:.2e} |d coef| {dc:.2e}; "
          f"F {rec['F'].shape}", flush=True)
    assert max(dm, ds, dc) < 1e-9, "refit classifier differs from the sealed one"
    Z = b.steps[0][1].transform(rec["F"])
    np.savez_compressed(OUT, Z=Z.astype(np.float16), y=rec["y"].astype(np.int8))
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB); |Z| max {np.abs(Z).max():.1f}", flush=True)


if __name__ == "__main__":
    main()
