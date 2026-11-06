#!/usr/bin/env python
"""Sprint 2026-11-09 W1 gates (i) + (ii) for solvers/bci_decoding/ens_x4f_shallow.py.

Per split (lopo0, later): fit X4f (its solver file's defaults, as
proteus_sweep.fit_and_score) and keep its joblib; build a submission folder
= ens_x4f_shallow.py as submission.py + that joblib + shallow_nets.pt (the K
seeds' saved state dicts, w); load it as the platform does
(proteus_eval.load_submission) and stream the split's test windows by the
organisers' predict_streams. Then:
  (i)  classes == the post-hoc ensemble (sprint1106_ensemble.mix of the dumped
       X4f probabilities and the mean of the seeds' saved net probabilities);
       the replayed X4f / net probabilities equal the dumped / saved ones
       (<= 1e-6; net B=1 float noise is ~1e-6, reported);
  (ii) w = 0 reproduces X4f's streamed classes (the dump's argmax) exactly.
rc 1 on any FAIL.

    python sprint1109_w1_gates.py --rows <dir with w_*.pt + net_*.npz> --seeds 33_34_35_36_37 --w 0.3
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_splits as PS  # noqa: E402
import proteus_sweep as SW  # noqa: E402
from sprint1106_ensemble import mix  # noqa: E402

CB = Path.home() / "codabench"
ENS = CB / "solvers/bci_decoding/ens_x4f_shallow.py"
DUMP = CB / "logs/sprint_1106/dump"
OUT = CB / "logs/sprint_1109/w1"


def make_folder(folder, joblib_path, rows, seeds, w, split="all"):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ENS, folder / "submission.py")
    if Path(joblib_path) != folder / "riemann_sealed.joblib":
        shutil.copyfile(joblib_path, folder / "riemann_sealed.joblib")
    sds = [torch.load(Path(rows) / f"w_shallow_{split}_s{s}.pt", map_location="cpu")
           for s in seeds]
    torch.save({"w": float(w), "seeds": [int(s) for s in seeds], "state_dicts": sds,
                "net_kwargs": {}}, folder / "shallow_nets.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(CB / "logs/sprint_1109/w0"))
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--splits", nargs="+", default=["lopo0", "later"])
    a = ap.parse_args()
    seeds = a.seeds.split("_")
    PC.require_env040()
    c = PC.load()
    meta = PE.official_meta()
    X4fSolver = SW.load_solver_class("riemann_sealed_proteus_x4f")
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    for split in a.splits:
        proto, fold = ("lopo", int(split[4:])) if split.startswith("lopo") else (split, None)
        folder = OUT / f"sub_{split}"
        jl = folder / "riemann_sealed.joblib"
        if not jl.exists():
            model, sc, *_ = SW.fit_and_score(X4fSolver, {}, proto, fold)
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model.parts, jl)
            print(f"{split}: X4f fitted, streamed {sc['balanced_accuracy_cell_mean']:.6f}", flush=True)
        make_folder(folder, jl, a.rows, seeds, a.w, split)
        te = np.asarray(PS.split(proto, fold, c.rows).test)
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        assert np.array_equal(d["test_idx"], te)
        Pn_saved = np.mean([np.load(Path(a.rows) / f"net_shallow_{split}_s{s}_o0.npz")["probas"]
                            for s in seeds], axis=0)
        post = mix(d["P_test"], Pn_saved, a.w)

        ens = PE.load_submission(folder, meta)
        print(f"{split}: loaded {type(ens).__name__} K={len(ens.nets)} w={ens.w} "
              f"torch threads={torch.get_num_threads()}", flush=True)
        sc, y_true, y_pred, info = PE.evaluate(ens, PE.test_loader(c, te))
        print(f"{split}: streamed ensemble cell mean {sc['balanced_accuracy_cell_mean']:.6f}, "
              f"{sc['ms_per_window']:.1f} ms/window", flush=True)
        check(f"(i) {split} streamed classes == post-hoc ensemble", np.array_equal(y_pred, post),
              f"{int((y_pred != post).sum())} of {len(post)} differ")
        # replay probabilities stream by stream (fresh copy + reset, B = 1)
        sid = np.asarray(info["stream_id"])
        X = np.asarray(c.X[te])
        Px = np.zeros_like(d["P_test"])
        Pn = np.zeros_like(Pn_saved)
        for s in np.unique(sid):
            f = copy.deepcopy(ens)
            f.reset_state()
            for i in np.where(sid == s)[0]:
                px, pn = f.predict_proba_parts(torch.from_numpy(X[i:i + 1]))
                Px[i], Pn[i] = px[0], pn[0]
        dx = float(np.abs(Px - d["P_test"]).max())
        dn = float(np.abs(Pn - Pn_saved).max())
        check(f"(i) {split} X4f probabilities == dump", dx <= 1e-6, f"max|d|={dx:.2e}")
        check(f"(i) {split} net probabilities == saved (B=1 float noise allowed <= 1e-5)",
              dn <= 1e-5, f"max|d|={dn:.2e}")
        # (ii) w = 0
        ens.w = 0.0
        sc0, _, y0, _ = PE.evaluate(ens, PE.test_loader(c, te))
        check(f"(ii) {split} w=0 == X4f streamed classes", np.array_equal(y0, d["P_test"].argmax(1)),
              f"{int((y0 != d['P_test'].argmax(1)).sum())} differ; cell mean {sc0['balanced_accuracy_cell_mean']:.6f}")
    print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
