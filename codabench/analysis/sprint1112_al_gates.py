#!/usr/bin/env python
"""Sprint 2026-11-12 P1 gates (i) + (ii) for solvers/bci_decoding/ens_x4f_shallow_al.py.

As sprint1109_w1_gates, for stream-aligned nets: per split (lopo0, later) a
submission folder = ens_x4f_shallow_al.py + the split's X4f joblib (reused
from logs/sprint_1109/w1/sub_<split>) + shallow_nets.pt (K aligned seeds'
state dicts, w, align = {Cg, buffer 64, prior 8}); loaded as the platform
does and streamed by the organisers' predict_streams.
  (i)  streamed classes == post-hoc mix of the dumped X4f probabilities and the
       mean of the seeds' saved aligned-net probabilities (sprint1106_netsmoke
       --align stream, batch align_test); probabilities replayed per stream at
       B = 1 equal the dump / saved ones (X4f <= 1e-6, nets <= 1e-4: float32
       eigh + conv noise, reported);
  (ii) w = 0 reproduces X4f's streamed classes exactly.
rc 1 on any FAIL.

    python sprint1112_al_gates.py --rows logs/sprint_1109/rows_align --w 0.5
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_ensemble import mix  # noqa: E402

CB = Path.home() / "codabench"
ENS = CB / "solvers/bci_decoding/ens_x4f_shallow_al.py"
DUMP = CB / "logs/sprint_1106/dump"
W1 = CB / "logs/sprint_1109/w1"
OUT = CB / "logs/sprint_1112/al_gates"


def make_folder_al(folder, joblib_path, rows, seeds, w, split="all", buffer=64, prior=8.0):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ENS, folder / "submission.py")
    if Path(joblib_path) != folder / "riemann_sealed.joblib":
        shutil.copyfile(joblib_path, folder / "riemann_sealed.joblib")
    rows = Path(rows)
    sds = [torch.load(rows / f"w_shallow_{split}_s{s}.pt", map_location="cpu") for s in seeds]
    cgs = [np.load(rows / f"cg_shallow_{split}_s{s}.npy") for s in seeds]
    assert all(np.array_equal(cgs[0], g) for g in cgs), "Cg differs across seeds"
    torch.save({"w": float(w), "seeds": [int(s) for s in seeds], "state_dicts": sds,
                "net_kwargs": {},
                "align": {"kind": "stream", "Cg": torch.from_numpy(cgs[0]),
                          "buffer": int(buffer), "prior": float(prior)}},
               folder / "shallow_nets.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(CB / "logs/sprint_1109/rows_align"))
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--splits", nargs="+", default=["lopo0", "later"])
    a = ap.parse_args()
    seeds = a.seeds.split("_")
    PC.require_env040()
    c = PC.load()
    meta = PE.official_meta()
    fails = 0
    print(f"[data] X {c.X.shape} {c.X.dtype}, rows {len(c.rows)}", flush=True)

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    for split in a.splits:
        proto, fold = ("lopo", int(split[4:])) if split.startswith("lopo") else (split, None)
        folder = OUT / f"sub_{split}"
        make_folder_al(folder, W1 / f"sub_{split}" / "riemann_sealed.joblib", a.rows, seeds,
                       a.w, split)
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
        check(f"(i) {split} aligned-net probabilities == saved (<= 1e-4)", dn <= 1e-4,
              f"max|d|={dn:.2e}")
        ens.w = 0.0
        sc0, _, y0, _ = PE.evaluate(ens, PE.test_loader(c, te))
        check(f"(ii) {split} w=0 == X4f streamed classes", np.array_equal(y0, d["P_test"].argmax(1)),
              f"{int((y0 != d['P_test'].argmax(1)).sum())} differ; cell mean "
              f"{sc0['balanced_accuracy_cell_mean']:.6f}")
    print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
