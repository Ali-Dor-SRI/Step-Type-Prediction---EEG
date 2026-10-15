"""EEGNet-Stream: EEGNet-StepType with stream alignment, for PROTEUS (Track 2).

Sprint 2026-10-16 (brief ``codabench/prompts/2026-10-16_proteus_weekend_model.md``,
Phase 3), the second model. Self-contained (one ``submission.py``); the network,
init, preprocessing and max-norm are ``eegnet_steptype.py``'s, copied verbatim
(that file is the reference and is not changed). What is new:

- ``align="stream"`` (Euclidean alignment): every training (subject, session) is
  whitened by the inverse square root of its own mean covariance (session ids
  from the dataset's trigger table, as Riemann-Sealed reads them; per subject
  when absent). At test, each stream (one person-session, a fresh deep copy,
  ``reset_state()``) is whitened by a running reference: the mean covariance of
  its windows so far (the last ``buffer``; 0 = all) mixed with the global
  training mean, weight of the stream t = n / (n + stream_k). Causal, label-free,
  within one stream: allowed by the 0.4.0 contract (FAQ 2026-10-13).
- training on the whitened windows held in memory: Adam, early stopping on a
  person-held-out validation set (every 5th training person), then a refit on all
  training windows for the best epoch count.
- ``standardize`` (per-window, per-channel z-scoring inside the net), on here.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

_KERAS_EPSILON = 1e-7
_KERAS_BATCHNORM = {"eps": 1e-3, "momentum": 0.01}   # Keras momentum=0.99
_KERAS_ADAM_EPSILON = 1e-7


# ---------------------------------------------------------------------------
# Per-window preprocessing (identical block in every thesis solver).
# Applied inside the model, so training and Codabench scoring see exactly the
# same transform. Stateless: nothing is fitted on data, so it is safe to run
# on test windows. Both steps are linear (they commute).
# ---------------------------------------------------------------------------
def _parse_band(bandpass):
    """"none" -> None; "8to30" -> (8.0, 30.0). (No "-": benchopt would parse
    "8-30" as arithmetic.)"""
    if bandpass in (None, "none"):
        return None
    lo, hi = (float(v) for v in str(bandpass).split("to"))
    if not 0 < lo < hi:
        raise ValueError(f"bad bandpass {bandpass!r}")
    return lo, hi


def _spatial_matrix(ch_names, reference, lambda2=1e-5, stiffness=4):
    """(C, C) matrix M so that re-referenced X = M @ X, or None.

    Only channels with a standard 10-05 position are re-referenced; any other
    channel (e.g. EMG/EOG in the sealed data) passes through unchanged.
    "laplacian" = MNE spherical-spline surface Laplacian (CSD) with the thesis
    pipeline's lambda2/stiffness; being linear, its matrix is read off by
    transforming an identity "recording". Rescaled to unit median gain.
    """
    if reference in (None, "none"):
        return None
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    known = {c.lower() for c in montage.ch_names}
    n = len(ch_names)
    idx = [i for i, c in enumerate(ch_names) if c.lower() in known]
    if len(idx) < 3:
        idx = list(range(n))
    M = np.eye(n)
    if reference == "car":
        M[np.ix_(idx, idx)] -= 1.0 / len(idx)
        return M
    if reference == "laplacian":
        info = mne.create_info([ch_names[i] for i in idx], 100.0, "eeg")
        epochs = mne.EpochsArray(np.eye(len(idx))[None], info, verbose=False)
        epochs.set_montage(montage, match_case=False, verbose=False)
        csd = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=lambda2, stiffness=stiffness, verbose=False)
        sub = csd.get_data()[0]
        M[np.ix_(idx, idx)] = sub / np.median(np.abs(np.diag(sub)))
        return M
    raise ValueError(f"unknown reference {reference!r}")


class WindowPreproc(nn.Module):
    """Re-reference then zero-phase FFT band-pass, per window, on any device.

    Band-pass: reflect-pad each window by T/2 on both sides, multiply the
    spectrum by a mask that is 1 on [lo, hi] with cosine skirts (min(2, lo/2)
    Hz below, 2 Hz above), inverse FFT, crop. No state, no fitting.
    """

    def __init__(self, ch_names, sfreq, bandpass="none", reference="none"):
        super().__init__()
        M = _spatial_matrix(list(ch_names), reference)
        self.register_buffer(
            "M", None if M is None else torch.as_tensor(M, dtype=torch.float32),
            persistent=False)   # rebuilt from ch_names in load_model
        self.band = _parse_band(bandpass)
        self.sfreq = float(sfreq)
        self._masks = {}

    def _mask(self, n, device):
        key = (n, str(device))
        if key not in self._masks:
            f = np.fft.rfftfreq(n, 1.0 / self.sfreq)
            lo, hi = self.band
            tw_lo, tw_hi = min(2.0, lo / 2), 2.0
            r_lo = np.clip((f - (lo - tw_lo)) / tw_lo, 0.0, 1.0)
            r_hi = np.clip(((hi + tw_hi) - f) / tw_hi, 0.0, 1.0)
            mask = (0.5 - 0.5 * np.cos(np.pi * r_lo)) * (0.5 - 0.5 * np.cos(np.pi * r_hi))
            self._masks[key] = torch.as_tensor(mask, dtype=torch.float32,
                                               device=device)
        return self._masks[key]

    def forward(self, x):
        if self.M is not None:
            x = torch.einsum("ij,bjt->bit", self.M.to(x.device, x.dtype), x)
        if self.band is not None:
            T = x.shape[-1]
            p = min(T // 2, T - 1)
            xp = F.pad(x, (p, p), mode="reflect")
            spec = torch.fft.rfft(xp, dim=-1) * self._mask(xp.shape[-1], x.device)
            x = torch.fft.irfft(spec, n=xp.shape[-1], dim=-1)[..., p:p + T]
        return x


def _keras_max_norm_(weight, max_value, dims):
    with torch.no_grad():
        norms = weight.pow(2).sum(dim=dims, keepdim=True).sqrt()
        weight.mul_(norms.clamp(0.0, max_value) / (_KERAS_EPSILON + norms))


def _glorot_uniform_(weight, *, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def _same_padding(kernel):
    left = (kernel - 1) // 2
    return nn.ZeroPad2d((left, kernel - 1 - left, 0, 0))


def _val_mask(info, n, device):
    """Per-window bool mask: True = held-out validation (every 5th group)."""
    for key in ("subject_id", "record_id"):
        if isinstance(info, dict) and key in info:
            ids = torch.as_tensor(info[key]).reshape(-1)
            if len(ids) == n and bool((ids >= 0).all()):
                return (ids % 5 == 4).to(device)
    return None


class EEGNetTorch(nn.Module):
    """The thesis EEGNet layer stack, multi-class, no tabular branch."""

    def __init__(self, n_channels, n_times, n_classes, *, f1=8,
                 depth_multiplier=2, f2=16, kernel_length=64,
                 separable_kernel_length=16, dropout_rate=0.5,
                 norm_rate=0.25, standardize=False, preproc=None):
        super().__init__()
        self.standardize = bool(standardize)
        self.preproc = preproc          # WindowPreproc or None (no weights)
        kernel_length = max(1, min(int(kernel_length), n_times))
        separable_kernel_length = max(1, min(int(separable_kernel_length),
                                             n_times))
        spatial_filters = f1 * depth_multiplier
        self.norm_rate = float(norm_rate)

        self.temporal_pad = _same_padding(kernel_length)
        self.temporal_conv = nn.Conv2d(1, f1, (1, kernel_length), bias=False)
        self.temporal_bn = nn.BatchNorm2d(f1, **_KERAS_BATCHNORM)
        self.spatial_depthwise = nn.Conv2d(
            f1, spatial_filters, (n_channels, 1), groups=f1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(spatial_filters, **_KERAS_BATCHNORM)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.dropout_1 = nn.Dropout(float(dropout_rate))

        self.separable_pad = _same_padding(separable_kernel_length)
        self.separable_depthwise = nn.Conv2d(
            spatial_filters, spatial_filters, (1, separable_kernel_length),
            groups=spatial_filters, bias=False)
        self.separable_pointwise = nn.Conv2d(spatial_filters, f2, 1,
                                             bias=False)
        self.separable_bn = nn.BatchNorm2d(f2, **_KERAS_BATCHNORM)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.dropout_2 = nn.Dropout(float(dropout_rate))

        n_flat = f2 * ((n_times // 4) // 8)
        if n_flat == 0:
            raise ValueError(f"n_times={n_times} < 32: too short for EEGNet")
        self.classifier = nn.Linear(n_flat, int(n_classes))
        self._reset_parameters(n_channels, f1, depth_multiplier)

    def _reset_parameters(self, n_channels, f1, depth_multiplier):
        k_t = self.temporal_conv.kernel_size[1]
        _glorot_uniform_(self.temporal_conv.weight, fan_in=k_t,
                         fan_out=k_t * f1)
        _glorot_uniform_(self.spatial_depthwise.weight,
                         fan_in=n_channels * f1,
                         fan_out=n_channels * depth_multiplier)
        sf = self.separable_depthwise.in_channels
        k_s = self.separable_depthwise.kernel_size[1]
        _glorot_uniform_(self.separable_depthwise.weight, fan_in=sf * k_s,
                         fan_out=k_s)
        _glorot_uniform_(self.separable_pointwise.weight, fan_in=sf,
                         fan_out=self.separable_pointwise.out_channels)
        _glorot_uniform_(self.classifier.weight,
                         fan_in=self.classifier.in_features,
                         fan_out=self.classifier.out_features)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if self.preproc is not None:
            x = self.preproc(x)
        if self.standardize:   # per-window, per-channel z-score
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        x = x.unsqueeze(1)
        x = self.temporal_bn(self.temporal_conv(self.temporal_pad(x)))
        x = F.elu(self.spatial_bn(self.spatial_depthwise(x)))
        x = self.dropout_1(self.pool_1(x))
        x = self.separable_pointwise(
            self.separable_depthwise(self.separable_pad(x)))
        x = F.elu(self.separable_bn(x))
        x = self.dropout_2(self.pool_2(x))
        return self.classifier(x.permute(0, 2, 3, 1).flatten(1))

    def apply_max_norm(self):
        _keras_max_norm_(self.spatial_depthwise.weight, 1.0, dims=(1, 2, 3))
        _keras_max_norm_(self.classifier.weight, self.norm_rate, dims=(1,))




_CTX_COLUMNS = ("context", "condition", "paradigm")


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _trigger_table(ds):
    for obj in (getattr(ds, "seg_ds", None), ds):
        try:
            trig = getattr(obj, "triggers", None) if obj is not None else None
        except Exception:
            trig = None
        if trig is not None and hasattr(trig, "columns"):
            return trig
    return None


def _covs(X):
    """(n, C, C) sample covariances of (n, C, T) float64 windows (demeaned),
    with a 1e-3 x mean-eigenvalue ridge."""
    Xc = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / X.shape[1]
    return C + 1e-3 * tr * np.eye(X.shape[1])[None]


def _inv_sqrtm(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    return (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T


class EEGNetStreamModel:
    """Network + alignment state; ``predict(X)`` on one stream at a time."""

    def __init__(self, net, device, lr, n_epochs, patience, align, buffer,
                 stream_k, batch_size=64, seed=33, refit=True):
        self.net = net.to(device)
        self.device = device
        self.lr, self.n_epochs, self.patience = lr, n_epochs, patience
        self.align, self.buffer, self.stream_k = align, int(buffer), float(stream_k)
        self.bs, self.seed, self.refit = int(batch_size), int(seed), refit
        self.C_global = None
        self.reset_state()

    def reset_state(self):
        self._buf = []

    # -- training -------------------------------------------------------------
    def _collect(self, train_loader):
        ds = train_loader.dataset
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                         collate_fn=getattr(train_loader, "collate_fn", None))
        Xs, ys, ss = [], [], []
        for Xb, yb, info in dl:
            Xs.append(_to_np(Xb).astype(np.float64))
            ys.append(_to_np(yb).reshape(-1))
            ss.append(_to_np(info["subject_id"]).reshape(-1))
        X, y, subj = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
        trig = _trigger_table(ds)
        sess = None
        if trig is not None and "session" in trig.columns and len(trig) == len(y):
            sess = trig["session"].astype(str).to_numpy()
        return X, y.astype(np.int64), subj, sess

    def _epochs(self, Xt, yt, n_epochs, Xv=None, yv=None):
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=_KERAS_ADAM_EPSILON)
        best, best_state, best_ep, wait = math.inf, None, n_epochs, 0
        for ep in range(n_epochs):
            self.net.train()
            perm = torch.randperm(len(yt), generator=g)
            tl, tn = 0.0, 0
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = F.cross_entropy(self.net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                self.net.apply_max_norm()
                tl += loss.item() * len(b)
                tn += len(b)
            msg = f"[EEGNet-Stream] epoch {ep + 1}/{n_epochs} train_loss={tl / max(tn, 1):.4f}"
            if Xv is not None:
                self.net.eval()
                with torch.no_grad():
                    out = torch.cat([self.net(Xv[i:i + 256]) for i in range(0, len(yv), 256)])
                vl = F.cross_entropy(out, yv).item()
                va = (out.argmax(1) == yv).float().mean().item()
                msg += f" val_loss={vl:.4f} val_acc={va:.3f}"
                if vl < best:
                    best, best_ep, wait = vl, ep + 1, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    wait += 1
            if ep % 10 == 0 or ep == n_epochs - 1:
                print(msg, flush=True)
            if Xv is not None and wait >= self.patience:
                print(f"[EEGNet-Stream] early stop at epoch {ep + 1}, best {best_ep}",
                      flush=True)
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best_ep

    def fit(self, train_loader):
        X, y, subj, sess = self._collect(train_loader)
        covs = _covs(X)
        self.C_global = covs.mean(0)
        ng = 0
        if self.align == "stream":
            g = subj.astype(np.int64) * 1000 + (
                0 if sess is None else np.unique(sess, return_inverse=True)[1].reshape(-1))
            for gg in np.unique(g):
                m = g == gg
                X[m] = np.einsum("ij,njt->nit", _inv_sqrtm(covs[m].mean(0)), X[m])
            ng = len(np.unique(g))
        print(f"[EEGNet-Stream] fitting on X={X.shape}, subjects={len(np.unique(subj))}, "
              f"classes={np.unique(y).tolist()}, align={self.align} ({ng} groups, "
              f"session_ids={'yes' if sess is not None else 'no'}), "
              f"buffer={self.buffer or 'all'}, stream_k={self.stream_k:g}, seed={self.seed}",
              flush=True)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y)
        people = np.unique(subj)
        val = np.isin(subj, people[4::5]) if len(people) >= 5 else np.zeros(len(y), bool)
        init = copy.deepcopy(self.net.state_dict())
        if val.any():
            v = torch.as_tensor(val)
            best_ep = self._epochs(Xt[~v], yt[~v], self.n_epochs, Xt[v], yt[v])
            if self.refit:
                self.net.load_state_dict(init)
                print(f"[EEGNet-Stream] refit on all {len(y)} windows for {best_ep} epochs",
                      flush=True)
                self._epochs(Xt, yt, best_ep)
        else:
            self._epochs(Xt, yt, self.n_epochs)
        return self

    # -- prediction -----------------------------------------------------------
    def _whiten(self, X):
        if self.align != "stream" or self.C_global is None:
            return X
        covs = _covs(X)
        out = np.empty_like(X)
        for i in range(len(X)):
            self._buf.append(covs[i])
            if self.buffer > 0:
                del self._buf[:-self.buffer]
            n = len(self._buf)
            t = n / (n + self.stream_k)
            C = (1 - t) * self.C_global + t * np.mean(self._buf, 0)
            out[i] = _inv_sqrtm(C) @ X[i]
        return out

    @torch.inference_mode()
    def predict_proba(self, X):
        self.net.eval()
        X = self._whiten(_to_np(X).astype(np.float64))
        out = self.net(torch.as_tensor(X, dtype=torch.float32).to(self.device))
        return torch.softmax(out, 1).cpu().numpy()

    def predict(self, X):
        return torch.as_tensor(self.predict_proba(X).argmax(1))


class Solver(CompetSolver):

    name = "EEGNet-Stream"

    parameters = {
        "lr": [1e-3],
        "n_epochs": [150],
        "patience": [25],
        "kernel_s": [0.5],
        "separable_kernel_s": [0.125],
        "dropout_rate": [0.5],
        "standardize": [True],
        "bandpass": ["none"],
        "reference": ["none"],
        "align": ["stream"],          # "stream" | "none"
        "buffer": [0],                # running reference length (0 = all so far)
        "stream_k": [8],
        "batch_size": [64],
        "seed": [33],
    }

    def load_model(self, meta):
        torch.manual_seed(int(self.seed))
        sfreq = float(meta["sfreq"])
        net = EEGNetTorch(
            n_channels=meta["n_chans"], n_times=meta["n_times"],
            n_classes=meta["n_classes"],
            kernel_length=round(self.kernel_s * sfreq),
            separable_kernel_length=round(self.separable_kernel_s * sfreq),
            dropout_rate=self.dropout_rate,
            standardize=self.standardize,
            preproc=WindowPreproc(meta["ch_names"], sfreq, self.bandpass, self.reference),
        )
        model = EEGNetStreamModel(net, self.device, self.lr, self.n_epochs, self.patience,
                                  self.align, self.buffer, self.stream_k,
                                  batch_size=self.batch_size, seed=self.seed)
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists() and self.train_loader is None:
            state = torch.load(weights, map_location="cpu")
            model.C_global = state.pop("_C_global").numpy().astype(np.float64)
            net.load_state_dict(state)
        return model

    def fit(self, model, train_loader):
        torch.manual_seed(int(self.seed))
        model.fit(train_loader)

    def save_model(self, model, path):
        state = dict(model.net.state_dict())
        state["_C_global"] = torch.as_tensor(model.C_global)
        torch.save(state, path / "weights.pt")
