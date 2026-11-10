#!/usr/bin/env python
"""Sprint 2026-11-12: model-agnostic causal per-stream AdaBN for saved braindecode nets.

Every BatchNorm{1,2}d module is replaced by a wrapper that normalises with
(prior * training moments + sum of this stream's per-window moments so far,
this window included) / (prior + n), per channel (moments over every axis but
the channel one). Windows are fed one at a time in stream order (B = 1). With
--prior inf the training statistics are used (= frozen; check vs saved).
Inputs aligned by netsmoke's align_test when cg_<model>_<split>_s<seed>.npy exists.
Writes <out>/net_<model>_<split>_s<seed>_o0.npz.

    python sprint1112_adabn_gen.py --model fbcnet --rows logs/sprint_1112/rows_fbc_al --out ... --prior 32
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_netsmoke import align_test, make, zscore  # noqa: E402

SPLITS = [f"lopo{k}" for k in range(10)] + ["later"]


class StreamBN(nn.Module):
    def __init__(self, bn, prior):
        super().__init__()
        self.bn, self.prior = bn, prior
        self.reset()

    def reset(self):
        self.s1 = self.s2 = None
        self.n = 0

    def forward(self, x):
        bn = self.bn
        if math.isinf(self.prior):
            m, v = bn.running_mean, bn.running_var
        else:
            dims = [d for d in range(x.dim()) if d != 1]
            m1, m2 = x.mean(dim=dims), (x ** 2).mean(dim=dims)
            self.s1 = m1 if self.s1 is None else self.s1 + m1
            self.s2 = m2 if self.s2 is None else self.s2 + m2
            self.n += 1
            p, mu0, v0 = self.prior, bn.running_mean, bn.running_var
            m = (p * mu0 + self.s1) / (p + self.n)
            v = torch.clamp((p * (v0 + mu0 ** 2) + self.s2) / (p + self.n) - m ** 2, min=1e-6)
        shape = [1, -1] + [1] * (x.dim() - 2)
        y = (x - m.view(shape)) / torch.sqrt(v.view(shape) + bn.eps)
        if bn.affine:
            y = y * bn.weight.view(shape) + bn.bias.view(shape)
        return y


def wrap(net, prior):
    ws = []
    for name, mod in list(net.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                w = StreamBN(child, prior)
                setattr(mod, cname, w)
                ws.append(w)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--prior", type=float, default=32.0)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    a = ap.parse_args()
    torch.set_num_threads(1)
    c = PC.load()
    rows, out = Path(a.rows), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maxd = 0.0
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
            net = make(a.model, Xr.shape[1], Xr.shape[2])
            net.load_state_dict(torch.load(rows / f"w_{a.model}_{s}_s{seed}.pt", map_location="cpu"))
            net.eval()
            ws = wrap(net, a.prior)
            cg = rows / f"cg_{a.model}_{s}_s{seed}.npy"
            X = zscore(align_test(c.rows, Xr, te, np.load(cg)) if cg.exists() else Xr)
            P = np.empty((len(te), 3))
            with torch.no_grad():
                for k in np.unique(keys):
                    for w in ws:
                        w.reset()
                    for i in np.where(keys == k)[0]:
                        P[i] = torch.softmax(net(torch.from_numpy(X[i:i + 1])), -1).numpy()[0]
            ref = rows / f"net_{a.model}_{s}_s{seed}_o0.npz"
            if math.isinf(a.prior) and ref.exists():
                maxd = max(maxd, float(np.abs(np.load(ref)["probas"] - P).max()))
            np.savez_compressed(dst, probas=P, y=c.y[te], test_idx=te)
        print(f"{time.strftime('%T')} {a.model} {s} done ({len(ws)} BN layers)"
              + (f" frozen max|dP| {maxd:.1e}" if math.isinf(a.prior) else ""), flush=True)


if __name__ == "__main__":
    main()
