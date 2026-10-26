#!/usr/bin/env python
"""Figures for codabench/MODEL_ARCHITECTURE.md (written to codabench/figures/).

Real data / real weights unless a figure says "schematic":
  splits      schematic of the warm-up and sealed train/test structure
  router      log-PSD fingerprints + cross-session router confusion (tangermann2012)
  align       tangent-space embedding before / after per-subject whitening (tangermann2012)
  bands       one window's filter-bank covariances (tangermann2012)
  features    feature-vector composition (block sizes measured from the solver code)
  results     the recipe ladder on the four stand-in proxies (numbers: SEALED_RECIPE.md § 2)
  eegnet      learned temporal responses + activation patterns of the uploaded
              EEGNet-StepType-WU1 (weights in submissions/, Dreyer windows in eda_cache)

Run in WSL (the caches live there):
    source ~/codabench/env.sh
    python ~/codabench/analysis/arch_figures.py            # all
    python ~/codabench/analysis/arch_figures.py router align
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
sys.path.insert(0, str(HERE))

BLUE, ORANGE, GREEN, GREY, PURPLE, LBLUE = ("#2a78d4", "#f06a3c", "#12a679", "#8d8d88",
                                            "#7a5cc7", "#9cc3f0")
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 160, "axes.titleweight": "bold", "axes.titlesize": 11,
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {p.relative_to(HERE.parent)} ({p.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
def fig_splits():
    """Schematic: what is trained on and what is scored, warm-up vs sealed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # warm-up: 87 participants, split by person (Dreyer 2023: 52 / 14 / 21)
    n_tr, n_va, n_te = 52, 14, 21
    for i in range(87):
        c = BLUE if i < n_tr else (GREY if i < n_tr + n_va else ORANGE)
        a.add_patch(Rectangle((i % 29, -(i // 29) * 1.0), 0.86, 0.86, color=c, lw=0))
    a.set_xlim(-0.3, 29.3); a.set_ylim(-9.5, 1.2); a.axis("off")
    a.set_title("Warm-up (Dreyer 2023): cross-subject", loc="left")
    a.text(0, -3.35, "one square = one participant (all their windows)", fontsize=8.5, color=GREY)
    a.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE, label=f"train ({n_tr})"),
                      Rectangle((0, 0), 1, 1, color=GREY, label=f"validation ({n_va}): early stopping"),
                      Rectangle((0, 0), 1, 1, color=ORANGE, label=f"test ({n_te}): scored, unseen people")],
             loc="upper left", bbox_to_anchor=(0, 0.52), frameon=False, fontsize=9, ncol=1)
    # sealed: 20 participants x 6 sessions
    for p in range(20):
        for s in range(6):
            hidden = p >= 10 and s >= 3
            b.add_patch(Rectangle((s, -p), 0.9, 0.86, color=ORANGE if hidden else BLUE, lw=0,
                                  alpha=0.95))
    b.set_xlim(-2.4, 6.2); b.set_ylim(-20.6, 2.6); b.axis("off")
    b.set_title("Sealed phase (28 Oct - 21 Nov): cross-session", loc="left")
    for s in range(6):
        b.text(s + 0.45, 1.15, f"S{s + 1}", ha="center", fontsize=9)
    b.text(-0.15, -4.5, "10 fully\nlabelled\nparticipants", ha="right", va="center", fontsize=9)
    b.text(-0.15, -14.5, "10 evaluation\nparticipants", ha="right", va="center", fontsize=9)
    b.plot([-0.1, -0.1], [-9.9, 0.9], color="k", lw=1); b.plot([-0.1, -0.1], [-19.9, -10.1], color="k", lw=1)
    b.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE, label="labelled: train / calibrate"),
                      Rectangle((0, 0), 1, 1, color=ORANGE, label="hidden: scored (subject x session x context cells)")],
             loc="lower center", bbox_to_anchor=(0.45, -0.12), frameon=False, fontsize=9, ncol=1)
    fig.suptitle("Schematic. Same people, later days: the model must survive session-to-session change",
                 fontsize=10, y=0.03, color="#444")
    fig.tight_layout()
    save(fig, "splits.png")


# ---------------------------------------------------------------------------
def _load(study):
    import xsess_lib as XL
    return XL, XL.load_study(study)


def fig_router():
    from scipy.signal import welch
    XL, d = _load("tangermann2012")
    X, subj, sess, sf = d["X"], d["subj"], d["sess"], d["meta"]["sfreq"]
    subjects = np.unique(subj)
    s0, s1 = np.unique(sess)[0], np.unique(sess)[-1]
    log(f"router: X={X.shape} subjects={len(subjects)} sessions={s0!r},{s1!r}")
    f, P = welch(X, fs=sf, nperseg=int(sf), axis=-1)
    m = (f >= 1) & (f <= 45)
    lp = 10 * np.log10(P[..., m] + 1e-12)            # (n, C, F) dB
    freqs = f[m]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    show = [subjects[i] for i in (0, 2, 4, 7)]
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    for s, c in zip(show, cols):
        for sess_id, ls in [(s0, "-"), (s1, "--")]:
            sel = (subj == s) & (sess == sess_id)
            a.plot(freqs, lp[sel].mean((0, 1)), color=c, ls=ls, lw=1.8)
    a.set_xlabel("frequency (Hz)"); a.set_ylabel("log power, mean of 22 channels (dB)")
    a.set_title("Individual spectra are stable across sessions")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=c, lw=2, label=f"participant {int(s) + 1}") for s, c in zip(show, cols)]
             + [Line2D([], [], color="k", ls="-", label="session 1"), Line2D([], [], color="k", ls="--", label="session 2")],
             frameon=False, fontsize=8.5, loc="upper right")
    # router trained on session 1, applied to session 2
    tr, te = sess == s0, sess == s1
    r = XL.SubjectRouter("psd", sf, cap=0.5).fit(X[tr], subj[tr], sess[tr])
    Pp = r.predict_proba(X[te])
    pred = Pp.argmax(1)
    unsure = Pp.max(1) < r.thr
    true = np.searchsorted(r.subjects, subj[te])
    acc = float((pred == true).mean())
    cm = np.zeros((len(subjects), len(subjects) + 1))
    for t, p_, u in zip(true, pred, unsure):
        cm[t, len(subjects) if u else p_] += 1
    cm = cm / cm.sum(1, keepdims=True)
    im = b.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    b.set_xticks(range(len(subjects) + 1)); b.set_yticks(range(len(subjects)))
    b.set_xticklabels([str(i + 1) for i in range(len(subjects))] + ["unsure"], fontsize=8.5)
    b.set_yticklabels([str(i + 1) for i in range(len(subjects))], fontsize=8.5)
    b.set_xlabel("routed to participant"); b.set_ylabel("true participant")
    b.set_title(f"Router trained on session 1, tested on session 2:\n{100 * acc:.1f} % of windows routed correctly")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.05:
                b.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                       color="white" if cm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.03, label="fraction of that participant's windows")
    fig.tight_layout()
    save(fig, "router.png")
    log(f"router: accuracy {acc:.4f}, unsure {unsure.mean():.4f}, thr {r.thr:.3f}")


# ---------------------------------------------------------------------------
def _var_share(V, groups):
    mu = V.mean(0)
    tot = ((V - mu) ** 2).sum()
    ss = sum((groups == g).sum() * ((V[groups == g].mean(0) - mu) ** 2).sum() for g in np.unique(groups))
    return float(ss / tot)


def fig_align():
    from pyriemann.tangentspace import TangentSpace
    from sklearn.decomposition import PCA
    XL, d = _load("tangermann2012")
    X, subj, sess = d["X"], d["subj"], d["sess"]
    subjects, sessions = np.unique(subj), np.unique(sess)
    log("align: window covariances")
    covs = XL.window_covs(X)
    si = np.searchsorted(subjects, subj)
    ss = np.searchsorted(sessions, sess)

    def whiten(kind):
        out = np.empty_like(covs)
        for s in range(len(subjects)):
            if kind == "calib":                       # the recipe: subject reference from session 1 only
                W = XL.inv_sqrtm(XL.mean_cov(covs[(si == s) & (ss == 0)], "riemann"))
                m = si == s
                out[m] = W @ covs[m] @ W.T
            else:                                     # oracle / online: each session with its own statistics
                for k in range(len(sessions)):
                    m = (si == s) & (ss == k)
                    W = XL.inv_sqrtm(XL.mean_cov(covs[m], "riemann"))
                    out[m] = W @ covs[m] @ W.T
        return out

    panels = [("A  raw covariances", covs),
              ("B  whitened with the subject's\nsession-1 reference (the recipe)", whiten("calib")),
              ("C  whitened with each session's\nown statistics (online / oracle)", whiten("session"))]
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where((si == s) & (ss == k))[0], 45, replace=False)
                           for s in range(len(subjects)) for k in range(len(sessions))])
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, C) in zip(axes, panels):
        log(f"align: tangent space {title.splitlines()[0]}")
        V = TangentSpace(metric="riemann").fit(C).transform(C)
        v_subj = _var_share(V, si)
        v_sess = _var_share(V, si * 10 + ss) - v_subj
        Z = PCA(2, random_state=0).fit_transform(V[pick])
        for s in range(len(subjects)):
            for k, mk in [(0, "o"), (1, "^")]:
                sel = (si[pick] == s) & (ss[pick] == k)
                ax.scatter(Z[sel, 0], Z[sel, 1], s=16, marker=mk, color=cmap(s), alpha=0.75, lw=0)
        ax.set_title(title, fontsize=10)
        txt = (f"variance explained by\nparticipant: {100 * v_subj:.0f} %\nsession (within participant): {100 * v_sess:.0f} %"
               if title[0] != "C" else "variance explained by participant\nand session: 0 % by construction\n(every group is re-centred on itself)")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        log(f"align: {title.splitlines()[0]}: participant {v_subj:.3f}, session-in-participant {v_sess:.3f}")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#555", label="session 1 (calibration)"),
                        Line2D([], [], marker="^", ls="", color="#555", label="session 2 (later day)")]
               + [Line2D([], [], marker="s", ls="", color=cmap(s), label=f"P{s + 1}") for s in range(len(subjects))],
               loc="lower center", ncol=11, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Tangermann 2012 (9 people, 22 channels): 45 windows per participant and session, PCA of tangent-space vectors",
                 fontsize=10.5, y=1.02)
    fig.text(0.5, -0.12, "In B the calibration session (circles) is centred by construction, since the reference is computed from it; "
             "the later session (triangles) is not, and stays displaced in person-specific directions.",
             ha="center", fontsize=8.8, color="#444")
    fig.tight_layout()
    save(fig, "alignment.png")


# ---------------------------------------------------------------------------
def fig_bands():
    import torch
    from pyriemann.estimation import Covariances
    XL, d = _load("tangermann2012")            # also puts the solver dir on sys.path
    import riemann_steptype as RS
    X, subj, sess, names = d["X"], d["subj"], d["sess"], d["meta"]["ch_names"]
    s0 = np.unique(subj)[2]
    own = np.where((subj == s0) & (sess == np.unique(sess)[0]))[0]     # the person's calibration windows
    idx = int(own[10])
    w = X[idx:idx + 1].astype(np.float32)
    bands = [("broadband", None)] + [(f"{b.replace('to', '-')} Hz", b) for b in RS.FB_BANDS]
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.35], hspace=0.32, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(w.shape[-1]) / 120.0
    for i in range(w.shape[1]):
        ax.plot(t, w[0, i] * 0.9 + i * 3.2, color="#333", lw=0.6)
    ax.set_yticks(np.arange(w.shape[1]) * 3.2); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(0, 4); ax.set_xlabel("time after cue (s)"); ax.spines["left"].set_visible(False)
    ax.set_title("One 4-s window (22 channels, 120 Hz, robust-scaled): the model sees this and nothing else", loc="left")
    logs = []
    for label, band in bands:
        def cov(x):
            if band is not None:
                with torch.no_grad():
                    x = RS.WindowPreproc([], 120.0, bandpass=band)(torch.from_numpy(x.astype(np.float32))).numpy()
            return Covariances("oas").transform(x.astype(np.float64))
        ref = XL.mean_cov(cov(X[own]), "riemann")            # this person's average covariance in this band
        Rm = XL.inv_sqrtm(ref)
        lam, V = np.linalg.eigh(Rm @ cov(w)[0] @ Rm)          # log-map of the window at the reference
        logs.append((label, (V * np.log(np.maximum(lam, 1e-12))) @ V.T))
    vmax = np.percentile(np.abs(np.stack([l for _, l in logs])), 99)
    for j, (label, S) in enumerate(logs):
        a = fig.add_subplot(gs[1, j])
        im = a.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(label, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_ylabel("whitened coordinate x\nwhitened coordinate")
    fig.colorbar(im, ax=fig.axes[1:], fraction=0.018, pad=0.02, label="log-map (deviation from\nthis person's average)")
    fig.text(0.5, 0.01, "Each panel: how this window's covariance in the band deviates from the person's average covariance in that band "
             "(Riemannian log-map, i.e. what the tangent space hands to the LDA).\nEach 22 x 22 symmetric matrix contributes 253 numbers. "
             "Axes are the person's whitened coordinates (near, but not identical to, the 22 channels): diagonal = change in variance, "
             "off-diagonal = change in coupling.", ha="center", fontsize=8.5, color="#444")
    save(fig, "band_covariances.png")


# ---------------------------------------------------------------------------
def fig_features():
    labels = ["xDAWN", "broadband", "log-variance", "filter bank (4 bands)"]
    cols = [ORANGE, BLUE, GREEN, PURPLE]
    rows = [("Warm-up shape\n27 ch, 2 classes", [136, 378, 27, 1512]),
            ("Sealed shape (expected)\n43 ch, 3 classes", [300, 946, 43, 3784]),
            ("Sealed, if 47 ch delivered\n(43 EEG + 2 EMG + 2 EOG)", [300, 1128, 47, 4512])]
    fig, ax = plt.subplots(figsize=(11, 3.3))
    for r, (name, vals) in enumerate(rows):
        left = 0
        for v, c, lab in zip(vals, cols, labels):
            ax.barh(r, v, left=left, color=c, height=0.62, label=lab if r == 0 else None)
            if v > 250:
                ax.text(left + v / 2, r, f"{v:,}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            else:                                     # too narrow for a label inside the bar
                ax.text(left + v / 2, r - 0.4, f"{v}", ha="center", va="bottom", color=c, fontsize=8)
            left += v
        ax.text(left + 60, r, f"{left:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 6700); ax.set_xlabel("features per window (input to the LDAs)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=9)
    ax.set_title("Feature-vector composition (counts measured from the solver code)", loc="left", pad=28)
    fig.tight_layout()
    save(fig, "feature_composition.png")


# ---------------------------------------------------------------------------
def fig_results():
    # SEALED_RECIPE.md § 2, Phase 3 (router alignment, last session; clean / online)
    data = {"Tangermann\n4 classes, 9 people": (0.25, [0.687, 0.777, 0.802, 0.837]),
            "Scherer\n5 classes, 9 people": (0.20, [0.277, 0.302, 0.322, 0.375]),
            "Zhou\n3 classes, 4 people": (1 / 3, [0.767, 0.688, 0.778, 0.797]),
            "Scherer, 3 sealed-like\nclasses, 9 people": (1 / 3, [0.452, 0.486, 0.486, 0.561])}
    labs = ["shared LDA only", "personal LDA only", "blend (the recipe)", "blend + online re-centring\n(pending the rules)"]
    cols = [GREY, LBLUE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    w = 0.19
    for g, (name, (chance, vals)) in enumerate(data.items()):
        for k, (v, c) in enumerate(zip(vals, cols)):
            x = g + (k - 1.5) * w
            ax.bar(x, v, w * 0.92, color=c, hatch="//" if k == 3 else None, edgecolor="white" if k != 3 else "#0a7a58",
                   label=labs[k] if g == 0 else None)
            ax.text(x, v + 0.008, f"{v:.3f}".lstrip("0"), ha="center", fontsize=8)
        ax.plot([g - 0.45, g + 0.45], [chance, chance], color="k", ls=":", lw=1.3,
                label="chance" if g == 0 else None)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data), fontsize=9)
    ax.set_ylabel("cell-averaged balanced accuracy\n(each person's last session)")
    ax.set_ylim(0, 0.95)
    ax.legend(ncol=5, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Proxy results: personalisation and blending help; online re-centring is the largest lever", loc="left", pad=30)
    fig.tight_layout()
    save(fig, "proxy_results.png")


# ---------------------------------------------------------------------------
def fig_eegnet():
    import mne
    import torch
    HOME = Path.home()
    z = zipfile.ZipFile(HOME / "codabench/submissions/eegnet_steptype_wu1_2026-09-24.zip")
    sd = torch.load(io.BytesIO(z.read("weights.pt")), map_location="cpu")
    Wt = sd["temporal_conv.weight"].numpy()[:, 0, 0, :]          # (8, 60)
    Ws = sd["spatial_depthwise.weight"].numpy()[:, 0, :, 0]      # (16, 27)
    cache = HOME / "neuralbench/eda_cache"
    import json
    meta = json.loads((cache / "meta.json").read_text())
    names, sf = meta["ch_names"], meta["sfreq"]
    Xall = np.load(cache / "X.npy", mmap_mode="r")
    split = np.load(cache / "split.npy")
    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(split == 0)[0], 1500, replace=False)
    Xd = np.asarray(Xall[np.sort(rows)], dtype=np.float64)       # (n, 27, 480)
    log(f"eegnet: windows {Xd.shape}, temporal {Wt.shape}, spatial {Ws.shape}")
    K = Wt.shape[1]
    left = (K - 1) // 2                                          # TF "same" padding
    Xp = np.pad(Xd, ((0, 0), (0, 0), (left, K - 1 - left)))
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(Xp, K, axis=-1)                    # (n, C, T, K)
    freqs = np.fft.rfftfreq(2048, 1 / sf)
    resp, peak, pats = [], [], []
    for k in range(Wt.shape[0]):
        y = win @ Wt[k]                                          # conv2d is cross-correlation: (n, C, T)
        y = y - y.mean(-1, keepdims=True)
        S = np.einsum("nct,ndt->cd", y, y) / (y.shape[0] * y.shape[-1])
        h = np.abs(np.fft.rfft(Wt[k], 2048))
        resp.append(h / h.max()); peak.append(float(freqs[h.argmax()]))
        pk = []
        for dd in range(2):
            w = Ws[2 * k + dd]
            a = S @ w / (w @ S @ w)
            pk.append(a / np.abs(a).max())
        pats.append(pk)
    order = np.argsort(peak)
    info = mne.create_info(names, sf, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    fig, axes = plt.subplots(8, 3, figsize=(7.4, 13.2), gridspec_kw={"width_ratios": [1.9, 1, 1], "wspace": 0.12, "hspace": 0.5})
    for r, k in enumerate(order):
        a = axes[r, 0]
        a.fill_between(freqs, resp[k], color=LBLUE, alpha=0.7); a.plot(freqs, resp[k], color=BLUE, lw=1.2)
        a.set_xlim(0, 60); a.set_ylim(0, 1.05); a.set_yticks([])
        a.axvline(peak[k], color=ORANGE, lw=0.8, ls="--")
        a.text(58, 0.85, f"peak {peak[k]:.1f} Hz", ha="right", fontsize=8.5, color=ORANGE)
        a.set_ylabel(f"temporal\nfilter {r + 1}", rotation=0, ha="right", va="center", fontsize=8.5)
        if r < 7:
            a.set_xticklabels([])
        else:
            a.set_xlabel("frequency (Hz)")
        for dd in range(2):
            ax = axes[r, 1 + dd]
            mne.viz.plot_topomap(pats[k][dd], info, axes=ax, show=False, contours=0, sensors=True,
                                 cmap="RdBu_r", vlim=(-1, 1), sphere=(0, 0, 0, 0.095))
            if r == 0:
                ax.set_title(f"spatial pattern {dd + 1}", fontsize=9)
    fig.suptitle("Uploaded EEGNet-StepType-WU1 (Dreyer warm-up): learned temporal filters and\nspatial activation patterns "
                 "(sign arbitrary), filters sorted by peak frequency", fontsize=10, y=0.94)
    save(fig, "eegnet_learned_filters.png")
    log("eegnet: peak frequencies (Hz), sorted: " + ", ".join(f"{peak[k]:.1f}" for k in order))


FIGS = {"splits": fig_splits, "router": fig_router, "align": fig_align, "bands": fig_bands,
        "features": fig_features, "results": fig_results, "eegnet": fig_eegnet}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    t0 = time.time()
    for n in want:
        log(f"=== {n}")
        FIGS[n]()
    log(f"done in {time.time() - t0:.0f} s")
