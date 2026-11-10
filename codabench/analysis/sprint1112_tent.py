#!/usr/bin/env python
"""Sprint 2026-11-12: online test-time adaptation (Tent / information maximisation) on saved nets.

Per test stream (fresh copy of the trained net per stream, B = 1, stream order):
BatchNorm statistics causal per stream (sprint1112_adabn_gen.StreamBN, prior p);
window i is predicted with the current parameters, then ONE optimiser step on
  loss_i = H(p_i) - beta * H(mean of p_1..p_i)     (beta 0 = Tent, > 0 = SHOT-style IM)
(past p's detached) updates only the BN affine parameters (`--params bn`) or all
parameters (`--params all`). Label-free, causal. Writes
<out>/net_<model>_<split>_s<seed>_o0.npz.

    python sprint1112_tent.py --model shallow --rows logs/sprint_1109/rows_align --out ... --lr 1e-3 --beta 1
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_netsmoke import align_test, make, zscore  # noqa: E402
from sprint1112_adabn_gen import wrap  # noqa: E402

SPLITS = [f"lopo{k}" for k in range(10)] + ["later"]


def ent(p):
    return -(p * torch.log(torch.clamp(p, 1e-12))).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shallow")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--prior", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--params", default="bn", choices=["bn", "all"])
    ap.add_argument("--warm", type=int, default=0, help="no update before this many windows")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    c = PC.load()
    rows, out = Path(a.rows), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nwin = 0
    for s in a.splits:
        proto, fold = ("lopo", int(s[4:])) if s.startswith("lopo") else (s, None)
        te = np.asarray(PS.split(proto, fold, c.rows).test)
        Xr = np.asarray(c.X[te])
        r = c.rows.iloc[te]
        keys = (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()
        for seed in a.seeds.split("_"):
            dst = out / f"net_{a.model}_{s}_s{seed}_o0.npz"
            if dst.exists():
                continue
            base = make(a.model, Xr.shape[1], Xr.shape[2])
            base.load_state_dict(torch.load(rows / f"w_{a.model}_{s}_s{seed}.pt", map_location="cpu"))
            base.eval()
            cg = rows / f"cg_{a.model}_{s}_s{seed}.npy"
            X = zscore(align_test(c.rows, Xr, te, np.load(cg)) if cg.exists() else Xr)
            P = np.empty((len(te), 3))
            for k in np.unique(keys):
                net = copy.deepcopy(base)
                ws = wrap(net, a.prior)
                net.eval()                                   # dropout off; BN handled by StreamBN
                if a.params == "bn":
                    ps = [p for w in ws for p in (w.bn.weight, w.bn.bias)]
                else:
                    ps = list(net.parameters())
                for p in net.parameters():
                    p.requires_grad_(False)
                for p in ps:
                    p.requires_grad_(True)
                opt = torch.optim.Adam(ps, lr=a.lr)
                psum = torch.zeros(3)
                for n, i in enumerate(np.where(keys == k)[0]):
                    pr = torch.softmax(net(torch.from_numpy(X[i:i + 1])), -1)[0]
                    P[i] = pr.detach().numpy()
                    nwin += 1
                    if n < a.warm:
                        psum += pr.detach()
                        continue
                    pbar = (psum + pr) / (n + 1)
                    loss = ent(pr) - a.beta * ent(pbar)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    psum += pr.detach()
                    for w in ws:                             # moments are data, not graph
                        w.s1, w.s2 = w.s1.detach(), w.s2.detach()
            np.savez_compressed(dst, probas=P, y=c.y[te], test_idx=te)
        print(f"{time.strftime('%T')} {a.model} {s} done; {1000 * (time.time() - t0) / max(nwin, 1):.1f} ms/window",
              flush=True)


if __name__ == "__main__":
    main()
