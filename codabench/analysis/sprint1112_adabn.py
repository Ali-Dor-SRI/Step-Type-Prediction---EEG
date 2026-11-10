#!/usr/bin/env python
"""Sprint 2026-11-12 Q7/Q5: causal per-stream BatchNorm adaptation (AdaBN) of saved ShallowNets.

For every split and seed, loads the saved state dict (w_shallow_<split>_s<seed>.pt
in --rows; aligned nets also cg_*.npy, inputs then aligned by netsmoke's
align_test), replays each test stream in order and replaces the BatchNorm
statistics by a causal per-stream estimate: for window i, per filter,
  mean_i = (prior * mu_train + sum_{j<=i} m_j) / (prior + i)
  var_i  = (prior * (var_train + mu_train^2) + sum_{j<=i} q_j) / (prior + i) - mean_i^2
where m_j / q_j are window j's mean / mean square of the BN input over time
(the stream's first windows lean on the training statistics). Label-free.
--mode frozen reproduces the saved probabilities (check).
Writes <out>/net_shallow_<split>_s<seed>_o0.npz (probas, y, test_idx), the
layout sprint1109_compare.py reads.

    python sprint1112_adabn.py --rows logs/sprint_1109/rows_align --out logs/sprint_1112/adabn_al_p8 --prior 8
"""

import argparse
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

SPLITS = [f"lopo{k}" for k in range(10)] + ["later"]


def log(m):
    print(f"{time.strftime('%T')} {m}", flush=True)


def keys_of(rows, idx):
    r = rows.iloc[idx]
    return (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()


@torch.no_grad()
def run_stream(net, X, prior, mode, buffer):
    """X (n, C, T) z-scored, stream order -> probas (n, 3)."""
    bn = net.bnorm
    mu0, var0 = bn.running_mean.clone(), bn.running_var.clone()
    out = np.empty((len(X), 3), dtype=np.float64)
    ms, qs = [], []
    for i in range(len(X)):
        x = torch.from_numpy(X[i:i + 1])
        h = x
        while h.dim() < 4:
            h = h.unsqueeze(-1)
        h = h.permute(0, 3, 2, 1)
        h = net.conv_time_spat(h)                     # (1, F, T', 1)
        if mode == "adabn":
            ms.append(h.mean(dim=(0, 2, 3)))
            qs.append((h ** 2).mean(dim=(0, 2, 3)))
            del ms[:-buffer], qs[:-buffer]
            n = len(ms)
            m = (prior * mu0 + torch.stack(ms).sum(0)) / (prior + n)
            q = (prior * (var0 + mu0 ** 2) + torch.stack(qs).sum(0)) / (prior + n)
            v = torch.clamp(q - m ** 2, min=1e-6)
        else:
            m, v = mu0, var0
        h = (h - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + bn.eps)
        h = h * bn.weight[None, :, None, None] + bn.bias[None, :, None, None]
        h = h * h
        h = net.pool(h)
        h = torch.log(torch.clamp(h, min=1e-6))
        out[i] = torch.softmax(net.final_layer(h), -1).numpy()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--mode", default="adabn", choices=["adabn", "frozen"])
    ap.add_argument("--prior", type=float, default=8.0)
    ap.add_argument("--buffer", type=int, default=10000)
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
        keys = keys_of(c.rows, te)
        for seed in a.seeds.split("_"):
            dst = out / f"net_shallow_{s}_s{seed}_o0.npz"
            if dst.exists():
                continue
            net = ShallowNetT()
            net.load_state_dict(torch.load(rows / f"w_shallow_{s}_s{seed}.pt", map_location="cpu"))
            net.eval()
            cg = rows / f"cg_shallow_{s}_s{seed}.npy"
            X = zscore(align_test(c.rows, Xr, te, np.load(cg)) if cg.exists() else Xr)
            P = np.empty((len(te), 3))
            for k in np.unique(keys):
                m = keys == k
                P[m] = run_stream(net, X[m], a.prior, a.mode, a.buffer)
            ref = rows / f"net_shallow_{s}_s{seed}_o0.npz"
            if a.mode == "frozen" and ref.exists():
                maxd = max(maxd, float(np.abs(np.load(ref)["probas"] - P).max()))
            np.savez_compressed(dst, probas=P, y=c.y[te], test_idx=te)
        log(f"{s} done{f' (frozen max|dP| vs saved {maxd:.1e})' if a.mode == 'frozen' else ''}")


if __name__ == "__main__":
    main()
