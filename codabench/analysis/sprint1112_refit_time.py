"""Sprint 11-12: time one Q5 refit (X4f's logistic layer, warm start, train + 100 pseudo rows) at the current thread count."""
import copy, sys, time
from pathlib import Path
import joblib, numpy as np
CB = Path.home() / "codabench"
d = np.load(CB / "logs/sprint_1106/dump/X4f__later__None.npz")
clf = joblib.load(CB / "logs/sprint_1106/dump/X4f__later__None.clf.joblib")
sc, lr0 = clf.steps[0][1], clf.steps[-1][1]
Z = sc.transform(d["F_train"]); Zt = sc.transform(d["F_test"][:100])
yp = d["P_test"][:100].argmax(1)
for dtype in (np.float64, np.float32):
    X = np.vstack([Z, Zt]).astype(dtype); y = np.concatenate([d["y_train"], yp])
    sw = np.concatenate([np.ones(len(Z)), np.full(100, 3.0)])
    ts = []
    for _ in range(3):
        lr = copy.deepcopy(lr0); lr.set_params(warm_start=True, max_iter=200)
        t0 = time.perf_counter(); lr.fit(X, y, sample_weight=sw); ts.append(time.perf_counter() - t0)
    print(f"{X.shape} {dtype.__name__}: refit {np.median(ts):.2f} s (n_iter {lr.n_iter_})", flush=True)
