#!/usr/bin/env python
"""Sprint 2026-11-12 Q5-net: in-stream pseudo-label fine-tuning of the ShallowNets' final layer.

Per stream and seed (aligned nets, AdaBN prior p as B1): each window's
penultimate features h_i (after BN / square / pool / log) are cached; before
window start, start + step, ... the final conv layer (a fresh copy of the trained
one each time) takes `iters` full-batch Adam steps on
  sum_j lam * CE(final(h_j), yp_j) over the kept past windows / n_kept + mu * ||theta - theta_0||^2
with pseudo-labels yp from the pseudo-label source (default: the B1 ensemble,
X4f dump + AdaBN nets, w 0.5; top q per predicted class). Windows until the next
refit use the tuned layer. Label-free, causal. Writes
<out>/net_shallow_<split>_s<seed>_o0.npz (compare with sprint1109_compare.py).

    python sprint1112_q5net.py --out logs/sprint_1112/q5net/q7 --q 0.7
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
from shallownet_t import ShallowNetT  # noqa: E402
from sprint1106_netsmoke import align_test, zscore  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
SPLITS = [f"lopo{k}" for k in range(10)] + ["later"]


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


@torch.no_grad()
def penult(net, x, st, prior):
    bn = net.bnorm
    h = net.conv_time_spat(x)
    st[0] += h.mean(dim=(0, 2, 3))
    st[1] += (h ** 2).mean(dim=(0, 2, 3))
    st[2] += 1
    mu0, v0 = bn.running_mean, bn.running_var
    m = (prior * mu0 + st[0]) / (prior + st[2])
    v = torch.clamp((prior * (v0 + mu0 ** 2) + st[1]) / (prior + st[2]) - m ** 2, min=1e-6)
    h = (h - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + bn.eps)
    h = h * bn.weight[None, :, None, None] + bn.bias[None, :, None, None]
    h = net.pool(h * h)
    return torch.log(torch.clamp(h, min=1e-6))


def keep_idx(Ppl, q):
    yp, conf = Ppl.argmax(1), Ppl.max(1)
    keep = []
    for c in range(Ppl.shape[1]):
        m = np.where(yp == c)[0]
        if len(m):
            keep.extend(m[np.argsort(-conf[m])][:max(1, int(round(q * len(m))))])
    return np.asarray(sorted(keep)), yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(CB / "logs/sprint_1109/rows_align"))
    ap.add_argument("--nets0", default=str(CB / "logs/sprint_1112/adabn/al_p32"), help="B1 net probabilities (pseudo-labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--prior", type=float, default=32.0)
    ap.add_argument("--q", type=float, default=0.7)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--wpl", type=float, default=0.5)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    a = ap.parse_args()
    torch.set_num_threads(1)
    c = PC.load()
    rows, out = Path(a.rows), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds.split("_")
    t0 = time.time()
    for s in a.splits:
        proto, fold = ("lopo", int(s[4:])) if s.startswith("lopo") else (s, None)
        te = np.asarray(PS.split(proto, fold, c.rows).test)
        d = np.load(DUMP / f"X4f__{proto}__{'None' if fold is None else fold}.npz")
        assert np.array_equal(d["test_idx"], te)
        Pn0 = np.mean([np.load(Path(a.nets0) / f"net_shallow_{s}_s{x}_o0.npz")["probas"] for x in seeds], axis=0)
        L = (1 - a.wpl) * lg(d["P_test"]) + a.wpl * lg(Pn0)
        Ppl = np.exp(L - L.max(1, keepdims=True))
        Ppl /= Ppl.sum(1, keepdims=True)
        Xr = np.asarray(c.X[te])
        r = c.rows.iloc[te]
        keys = (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()
        for seed in seeds:
            dst = out / f"net_shallow_{s}_s{seed}_o0.npz"
            if dst.exists():
                continue
            net = ShallowNetT()
            net.load_state_dict(torch.load(rows / f"w_shallow_{s}_s{seed}.pt", map_location="cpu"))
            net.eval()
            X = zscore(align_test(c.rows, Xr, te, np.load(rows / f"cg_shallow_{s}_s{seed}.npy")))
            P = np.empty((len(te), 3))
            for k in np.unique(keys):
                idx = np.where(keys == k)[0]
                st = [torch.zeros(40), torch.zeros(40), 0]
                fin = net.final_layer
                H = []
                for j, i in enumerate(idx):
                    if j >= a.start and (j - a.start) % a.step == 0:
                        kp, yp = keep_idx(Ppl[idx[:j]], a.q)
                        fin = copy.deepcopy(net.final_layer)
                        th0 = [p.detach().clone() for p in fin.parameters()]
                        opt = torch.optim.Adam(fin.parameters(), lr=a.lr)
                        Hk = torch.cat([H[t] for t in kp])
                        yk = torch.from_numpy(yp[kp].astype(np.int64))
                        for _ in range(a.iters):
                            opt.zero_grad()
                            loss = torch.nn.functional.cross_entropy(fin(Hk), yk) + a.mu * sum(
                                ((p - p0) ** 2).sum() for p, p0 in zip(fin.parameters(), th0))
                            loss.backward()
                            opt.step()
                        fin.eval()
                    h = penult(net, torch.from_numpy(X[i:i + 1]).unsqueeze(-1).permute(0, 3, 2, 1), st, a.prior)
                    H.append(h)
                    with torch.no_grad():
                        P[i] = torch.softmax(fin(h), -1).numpy()[0]
            np.savez_compressed(dst, probas=P, y=c.y[te], test_idx=te)
        print(f"{time.strftime('%T')} {s} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
