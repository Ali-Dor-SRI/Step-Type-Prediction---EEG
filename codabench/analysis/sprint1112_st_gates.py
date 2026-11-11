#!/usr/bin/env python
"""Sprint 2026-11-12 Q5 gates (i) for the self-training solver (ens_x4f_shallow_al.py + "selftrain").

Per split (lopo0, later): folder = aligned AdaBN ensemble (sprint1112_al_gates.make_folder_al)
+ "selftrain" in shallow_nets.pt + x4f_trainZ.npz (the split's dumped X4f training
features through the folder's own scaler, float16). Loaded as the platform does and
streamed; checks
  (i-a) the adapted X4f probabilities == the probe's (sprint1112_q5.py npz `probas`), <= 1e-4;
  (i-b) streamed classes == the post-hoc mix of the probe's adapted X4f + saved nets, w;
  (i-c) the replayed nets' probabilities == the probe's Pn (AdaBN p32 saved), <= 1e-4.
rc 1 on any FAIL.

    python sprint1112_st_gates.py --probe q5_q0.7_l5.0_s40_t40_f16 --q 0.7 --lam 5 --step 40 --w 0.5
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_ensemble import mix  # noqa: E402
from sprint1112_al_gates import make_folder_al  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
W1 = CB / "logs/sprint_1109/w1"


def add_selftrain(folder, F_train, y_train, st):
    d = torch.load(folder / "shallow_nets.pt", map_location="cpu", weights_only=True)
    d["selftrain"] = st
    torch.save(d, folder / "shallow_nets.pt")
    sc = joblib.load(folder / "riemann_sealed.joblib")["lda"].steps[0][1]
    np.savez_compressed(folder / "x4f_trainZ.npz", Z=sc.transform(F_train).astype(np.float16),
                        y=np.asarray(y_train).astype(np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sprint1112_q5 run name (npz prefix in logs/sprint_1112/q5)")
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--wpl", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--adabn", type=float, default=32.0)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--rows", default=str(CB / "logs/sprint_1109/rows_align"))
    ap.add_argument("--splits", nargs="+", default=["lopo0", "later"])
    ap.add_argument("--out", default=str(CB / "logs/sprint_1112/st_gates"))
    a = ap.parse_args()
    PC.require_env040()
    c = PC.load()
    meta = PE.official_meta()
    seeds = "33_34_35_36_37".split("_")
    st = {"q": a.q, "lam": a.lam, "start": a.start, "step": a.step, "wpl": a.wpl, "budget_s": 1e9, **({"C": a.C} if a.C else {})}
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    for split in a.splits:
        proto, fold = ("lopo", int(split[4:])) if split.startswith("lopo") else (split, None)
        folder = Path(a.out) / f"sub_{split}"
        make_folder_al(folder, W1 / f"sub_{split}" / "riemann_sealed.joblib", a.rows, seeds, a.w, split,
                       adabn=a.adabn)
        d = np.load(DUMP / f"X4f__{proto}__{'None' if fold is None else fold}.npz")
        add_selftrain(folder, d["F_train"], d["y_train"], st)
        z = np.load(CB / "logs/sprint_1112/q5" / f"{a.probe}_{split}.npz")
        te = np.asarray(PS.split(proto, fold, c.rows).test)
        assert np.array_equal(z["test_idx"], te)
        ens = PE.load_submission(folder, meta)
        print(f"{split}: loaded {type(ens).__name__} K={len(ens.nets)} w={ens.w}", flush=True)
        # replay per stream (fresh reset), B = 1, to read both parts
        sid = PE.evaluate(ens, PE.test_loader(c, te))[3]["stream_id"]
        sid = np.asarray(sid)
        X = np.asarray(c.X[te])
        Px = np.zeros((len(te), 3))
        Pn = np.zeros((len(te), 3))
        for s in np.unique(sid):
            ens.reset_state()
            for i in np.where(sid == s)[0]:
                px, pn = ens.predict_proba_parts(torch.from_numpy(X[i:i + 1]))
                Px[i], Pn[i] = px[0], pn[0]
        ens.reset_state()
        sc, _, y_pred, _ = PE.evaluate(ens, PE.test_loader(c, te))
        post = mix(z["probas"], z["Pn"], a.w)
        dx = float(np.abs(Px - z["probas"]).max())
        dn = float(np.abs(Pn - z["Pn"]).max())
        check(f"(i-a) {split} adapted X4f probabilities == probe", dx <= 1e-4, f"max|d|={dx:.2e}")
        check(f"(i-c) {split} net probabilities == probe", dn <= 1e-4, f"max|d|={dn:.2e}")
        check(f"(i-b) {split} streamed classes == post-hoc (probe adapted X4f + nets, w {a.w})",
              np.array_equal(np.asarray(y_pred), post), f"{int((np.asarray(y_pred) != post).sum())} of {len(post)} "
              f"differ; streamed cell mean {sc['balanced_accuracy_cell_mean']:.6f}, {sc['ms_per_window']:.1f} "
              f"ms/window, refits {ens.nrefit}, {ens.spent:.0f} s")
    print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
