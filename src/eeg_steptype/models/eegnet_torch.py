"""EEGNet — PyTorch port of the Keras ``eegnet`` model.

A layer-for-layer port of :mod:`eeg_steptype.models.eegnet`, the Keras
implementation that produced the recorded EEGNet results (it is left exactly
as it is). Registered as its own model, ``eegnet_torch``, so the two can be
run and compared side by side.

Two pieces:

* :class:`EEGNetTorch` -- a plain ``torch.nn.Module``. ``forward`` takes
  ``(batch, n_channels, n_times)`` -- braindecode's input convention, so a
  braindecode/eegdash training loop can take the module as it is -- plus an
  optional ``(batch, n_tabular)`` tensor for the hybrid fusion branch, and
  returns ``(batch, 1)`` logits. braindecode itself is never imported here.
* :class:`EEGNetTorchClassifier` -- a scikit-learn estimator around it
  (``fit`` / ``predict`` / ``predict_proba``), so it runs unchanged under the
  nested-CV driver's GridSearchCV and ``scripts/08_tensor_model_diagnostics.py``.
  It accepts a 3-D tensor or the 2-D hybrid layout the driver builds
  (flattened tensor followed by the tabular columns).

Mirrored from the Keras file (Keras 3 semantics):

* the layer sequence, including BatchNorm after each convolution and the
  tabular/fusion branch;
* the max-norm constraints, re-applied after every optimizer step with the
  Keras formula over the same axes (depthwise spatial filters to 1.0, fusion
  and classifier dense weights to ``norm_rate``) -- PyTorch has no constraint
  API;
* TensorFlow "same" padding (an even kernel's extra sample goes on the
  right), the channels-last flatten order, glorot-uniform init with Keras's
  fan convention, BatchNorm momentum/epsilon, Adam's epsilon, and the L2
  penalty on the tabular dense kernel;
* the training loop: Adam, epochs, batch size, early stopping on
  ``val_loss`` with ``restore_best_weights``, and the validation split taken
  as Keras takes it -- the *trailing* fraction of the training fold, before
  shuffling.

Unlike the Keras path (scikeras is left unseeded), every fit seeds torch and
numpy from ``modeling.random_state``. CPU only.

PyTorch is imported when this module loads; ``models.train`` registers the
model through a lazy factory, so importing the training driver for a
classical run does not import torch.
"""

from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.nn import functional as F

from ..logging_utils import get_logger
from .cnn import ExponentialMovingStandardizer, HybridTensorFeatureStandardizer


log = get_logger(__name__)

# Keras 3 values this port reproduces: keras.config.epsilon() (used by the
# MaxNorm constraint), the BatchNormalization and Adam defaults, and the L2
# factor hard-coded on the Keras model's tabular dense layer.
_KERAS_EPSILON = 1e-7
_KERAS_BATCHNORM = {"eps": 1e-3, "momentum": 0.01}   # Keras momentum=0.99 == PyTorch 0.01
_KERAS_ADAM_EPSILON = 1e-7
_TABULAR_L2 = 1e-4
# Keras ``predict`` batch size; also bounds memory on 64 x 2049 epochs.
_PREDICT_BATCH_SIZE = 32


def make_normalizer(cfg: dict, n_features=None):
    """Fold-local standardizer -- the same transformer the Keras ``eegnet`` uses."""
    ecfg = cfg.get("modeling", {}).get("eegnet_torch", {}).get("standardize", {})
    hybrid = cfg.get("_neural_hybrid_input")
    if hybrid:
        return HybridTensorFeatureStandardizer(
            n_channels=int(hybrid["n_channels"]),
            n_times=int(hybrid["n_times"]),
            n_tabular_features=int(hybrid["n_tabular_features"]),
            factor_new=float(ecfg.get("factor_new", 0.001)),
            init_block_size=int(ecfg.get("init_block_size", 1000)),
            eps=float(ecfg.get("eps", 1e-4)),
        )
    return ExponentialMovingStandardizer(
        factor_new=float(ecfg.get("factor_new", 0.001)),
        init_block_size=int(ecfg.get("init_block_size", 1000)),
        eps=float(ecfg.get("eps", 1e-4)),
    )


# ---------------------------------------------------------------------------
# Keras semantics
# ---------------------------------------------------------------------------
def keras_max_norm_(weight: torch.Tensor, max_value: float, dims: tuple[int, ...]) -> None:
    """In place, Keras ``MaxNorm``: ``w * clip(||w||, 0, max_value) / (eps + ||w||)``.

    ``dims`` are the axes the L2 norm is taken over -- the PyTorch counterpart
    of the Keras constraint's ``axis``. Every slice along the remaining axes
    ends up with norm <= ``max_value``.
    """
    with torch.no_grad():
        norms = weight.pow(2).sum(dim=dims, keepdim=True).sqrt()
        weight.mul_(norms.clamp(0.0, max_value) / (_KERAS_EPSILON + norms))


def keras_validation_split(n_samples: int, validation_split: float):
    """Keras ``fit(validation_split=...)``: the last fraction of samples, before shuffling.

    Returns ``(train_idx, val_idx)``; ``val_idx`` is ``None`` when the split is
    0. Raises, as Keras does, when either side would be empty.
    """
    if not validation_split:
        return np.arange(n_samples), None
    split_at = int(math.floor(n_samples * (1.0 - float(validation_split))))
    if split_at == 0 or split_at == n_samples:
        raise ValueError(
            f"Training data contains {n_samples} samples, which is not sufficient "
            f"to split into a training and validation set with "
            f"validation_split={validation_split}."
        )
    return np.arange(split_at), np.arange(split_at, n_samples)


class KerasEarlyStopping:
    """Keras 3 ``EarlyStopping(monitor="val_loss", restore_best_weights=True)``.

    Same bookkeeping as the Keras callback: ``wait`` counts epochs since the
    last strict improvement, training stops once ``wait >= patience``, and the
    best epoch's weights are restored when training ends -- whether or not it
    stopped early.
    """

    def __init__(self, patience: int):
        self.patience = int(patience)
        self.wait = 0
        self.best = math.inf
        self.best_epoch: int | None = None
        self.best_state: dict | None = None
        self.stopped_epoch = 0

    def update(self, epoch: int, value: float, module: nn.Module) -> bool:
        """Record one epoch's monitored value; return True when training should stop."""
        if self.best_state is None:
            self.best_state = copy.deepcopy(module.state_dict())
            self.best_epoch = epoch
        self.wait += 1
        if value < self.best:
            self.best = value
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(module.state_dict())
            self.wait = 0
            return False
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            return True
        return False

    def restore(self, module: nn.Module) -> None:
        if self.best_state is not None:
            module.load_state_dict(self.best_state)


def _glorot_uniform_(weight: torch.Tensor, *, fan_in: int, fan_out: int) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def _same_padding(kernel: int) -> nn.ZeroPad2d:
    """TensorFlow ``padding="same"`` along time: an even kernel's extra zero goes right."""
    left = (kernel - 1) // 2
    return nn.ZeroPad2d((left, kernel - 1 - left, 0, 0))


@contextlib.contextmanager
def _seeded(seed: int | None):
    """Seed torch's CPU RNG inside the block without disturbing the global stream."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class EEGNetTorch(nn.Module):
    """The Keras ``eegnet`` layer stack as a ``torch.nn.Module``.

    ``x`` is ``(batch, n_channels, n_times)``; ``x_tabular`` is
    ``(batch, n_tabular)`` and is required iff ``n_tabular > 0``. Returns
    ``(batch, 1)`` logits -- the Keras model ends in a sigmoid; here the
    sigmoid lives in the loss and in ``predict_proba``.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        *,
        n_tabular: int = 0,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        dropout_rate: float = 0.5,
        tabular_units: int = 32,
        fusion_units: int = 32,
        norm_rate: float = 0.25,
    ):
        super().__init__()
        n_channels, n_times, n_tabular = int(n_channels), int(n_times), int(n_tabular)
        f1, depth_multiplier, f2 = int(f1), int(depth_multiplier), int(f2)
        # Keras clamps both kernels to the epoch length (``_valid_kernel``).
        kernel_length = max(1, min(int(kernel_length), n_times))
        separable_kernel_length = max(1, min(int(separable_kernel_length), n_times))
        spatial_filters = f1 * depth_multiplier
        self.n_channels, self.n_times, self.n_tabular = n_channels, n_times, n_tabular
        self.norm_rate = float(norm_rate)

        # Block 1: temporal conv -> BN -> depthwise spatial conv -> BN -> ELU
        #          -> average-pool (1, 4) -> dropout.
        self.temporal_pad = _same_padding(kernel_length)
        self.temporal_conv = nn.Conv2d(1, f1, (1, kernel_length), bias=False)
        self.temporal_bn = nn.BatchNorm2d(f1, **_KERAS_BATCHNORM)
        self.spatial_depthwise = nn.Conv2d(
            f1, spatial_filters, (n_channels, 1), groups=f1, bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(spatial_filters, **_KERAS_BATCHNORM)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.dropout_1 = nn.Dropout(float(dropout_rate))

        # Block 2: separable conv (depthwise temporal + pointwise) -> BN -> ELU
        #          -> average-pool (1, 8) -> dropout.
        self.separable_pad = _same_padding(separable_kernel_length)
        self.separable_depthwise = nn.Conv2d(
            spatial_filters, spatial_filters, (1, separable_kernel_length),
            groups=spatial_filters, bias=False,
        )
        self.separable_pointwise = nn.Conv2d(spatial_filters, f2, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(f2, **_KERAS_BATCHNORM)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.dropout_2 = nn.Dropout(float(dropout_rate))

        n_flat = f2 * ((n_times // 4) // 8)
        if n_flat == 0:
            raise ValueError(
                f"n_times={n_times} is too short: EEGNet average-pools time by 4 "
                "and then 8, so it needs at least 32 samples."
            )

        # Head: optional tabular branch fused with the conv features, then the
        # single-logit classifier.
        if n_tabular > 0:
            self.tabular_dense = nn.Linear(n_tabular, int(tabular_units))
            self.tabular_dropout = nn.Dropout(float(dropout_rate))
            self.fusion_dense = nn.Linear(n_flat + int(tabular_units), int(fusion_units))
            head_in = int(fusion_units)
        else:
            self.tabular_dense = self.tabular_dropout = self.fusion_dense = None
            head_in = n_flat
        self.classifier = nn.Linear(head_in, 1)
        self._reset_parameters(n_channels, f1, depth_multiplier)

    def _reset_parameters(self, n_channels: int, f1: int, depth_multiplier: int) -> None:
        """Keras ``glorot_uniform`` kernels and zero biases, with Keras's fans.

        Keras computes fans from its own kernel layouts. They coincide with
        PyTorch's for every layer except the depthwise spatial conv, whose
        Keras kernel is ``(n_channels, 1, f1, depth_multiplier)``.
        """
        k_temporal = self.temporal_conv.kernel_size[1]
        _glorot_uniform_(self.temporal_conv.weight, fan_in=k_temporal, fan_out=k_temporal * f1)
        _glorot_uniform_(
            self.spatial_depthwise.weight,
            fan_in=n_channels * f1,
            fan_out=n_channels * depth_multiplier,
        )
        spatial_filters = self.separable_depthwise.in_channels
        k_separable = self.separable_depthwise.kernel_size[1]
        _glorot_uniform_(
            self.separable_depthwise.weight,
            fan_in=spatial_filters * k_separable,
            fan_out=k_separable,
        )
        _glorot_uniform_(
            self.separable_pointwise.weight,
            fan_in=spatial_filters,
            fan_out=self.separable_pointwise.out_channels,
        )
        for dense in (self.tabular_dense, self.fusion_dense, self.classifier):
            if dense is not None:
                _glorot_uniform_(dense.weight, fan_in=dense.in_features, fan_out=dense.out_features)
                nn.init.zeros_(dense.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional blocks -> flat features (the Keras ``flatten`` output)."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {tuple(x.shape)}")
        x = x.unsqueeze(1)                     # Keras adds a trailing image axis; here it leads
        x = self.temporal_bn(self.temporal_conv(self.temporal_pad(x)))
        x = F.elu(self.spatial_bn(self.spatial_depthwise(x)))
        x = self.dropout_1(self.pool_1(x))
        x = self.separable_pointwise(self.separable_depthwise(self.separable_pad(x)))
        x = F.elu(self.separable_bn(x))
        x = self.dropout_2(self.pool_2(x))
        # Keras flattens channels-last (height, time, filter). Permute first so
        # the flat layout -- and so the dense-weight layout -- is the same.
        return x.permute(0, 2, 3, 1).flatten(1)

    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None) -> torch.Tensor:
        z = self.features(x)
        if self.fusion_dense is not None:
            if x_tabular is None:
                raise ValueError("this network has a tabular branch (n_tabular > 0); pass x_tabular")
            tab = self.tabular_dropout(F.elu(self.tabular_dense(x_tabular)))
            z = F.elu(self.fusion_dense(torch.cat([z, tab], dim=1)))
        return self.classifier(z)

    def apply_max_norm(self) -> None:
        """Re-impose the Keras max-norm constraints; call after every optimizer step.

        Keras ``max_norm`` defaults to ``axis=0`` of its kernel layout:

        * depthwise spatial kernel ``(n_channels, 1, f1, D)`` -> each spatial
          filter's norm across electrodes, i.e. dims (1, 2, 3) of the PyTorch
          weight ``(f1 * D, 1, n_channels, 1)``, bounded by 1.0;
        * dense kernels ``(in, out)`` -> each output unit's incoming weights,
          i.e. dim 1 of the PyTorch weight ``(out, in)``, bounded by ``norm_rate``.

        Biases are unconstrained, as in Keras.
        """
        keras_max_norm_(self.spatial_depthwise.weight, 1.0, dims=(1, 2, 3))
        if self.fusion_dense is not None:
            keras_max_norm_(self.fusion_dense.weight, self.norm_rate, dims=(1,))
        keras_max_norm_(self.classifier.weight, self.norm_rate, dims=(1,))

    def regularization_loss(self) -> torch.Tensor:
        """Keras ``kernel_regularizer=l2(1e-4)`` on the tabular dense kernel (0 without one)."""
        if self.tabular_dense is None:
            return torch.zeros(())
        return _TABULAR_L2 * self.tabular_dense.weight.pow(2).sum()


# ---------------------------------------------------------------------------
# scikit-learn estimator
# ---------------------------------------------------------------------------
class EEGNetTorchClassifier(ClassifierMixin, BaseEstimator):
    """Binary scikit-learn classifier around :class:`EEGNetTorch`.

    ``X`` is either ``(n_epochs, n_channels, n_times)`` or the driver's 2-D
    hybrid layout ``(n_epochs, n_channels * n_times + n_tabular)``; the latter
    needs ``n_channels`` and ``n_times``. Hyperparameter names and defaults
    are those of ``build_fn`` / ``KerasClassifier`` in ``models/eegnet.py``.
    """

    def __init__(
        self,
        n_channels: int | None = None,
        n_times: int | None = None,
        n_tabular: int = 0,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        dropout_rate: float = 0.5,
        tabular_units: int = 32,
        fusion_units: int = 32,
        learning_rate: float = 1e-3,
        norm_rate: float = 0.25,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2,
        patience: int = 10,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_tabular = n_tabular
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2
        self.kernel_length = kernel_length
        self.separable_kernel_length = separable_kernel_length
        self.dropout_rate = dropout_rate
        self.tabular_units = tabular_units
        self.fusion_units = fusion_units
        self.learning_rate = learning_rate
        self.norm_rate = norm_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    # -- public API ---------------------------------------------------------
    def fit(self, X, y):
        arr = np.asarray(X, dtype=np.float32)
        self._resolve_shape(arr)
        tensor, tabular = self._split_input(arr)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"EEGNetTorchClassifier is binary; got {classes.size} class(es).")
        self.classes_ = classes
        targets = (y == classes[1]).astype(np.float32)

        with _seeded(self.random_state):
            rng = np.random.default_rng(self.random_state)
            self.module_ = EEGNetTorch(
                self.n_channels_,
                self.n_times_,
                n_tabular=int(self.n_tabular),
                f1=self.f1,
                depth_multiplier=self.depth_multiplier,
                f2=self.f2,
                kernel_length=self.kernel_length,
                separable_kernel_length=self.separable_kernel_length,
                dropout_rate=self.dropout_rate,
                tabular_units=self.tabular_units,
                fusion_units=self.fusion_units,
                norm_rate=self.norm_rate,
            )
            self.history_ = self._train(tensor, tabular, targets, rng)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "module_")
        tensor, tabular = self._split_input(np.asarray(X, dtype=np.float32))
        module = self.module_.eval()
        x_all = torch.from_numpy(tensor)
        tab_all = None if tabular is None else torch.from_numpy(tabular)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), _PREDICT_BATCH_SIZE):
                rows = slice(start, start + _PREDICT_BATCH_SIZE)
                logits = module(x_all[rows], None if tab_all is None else tab_all[rows])
                chunks.append(torch.sigmoid(logits).squeeze(1))
        positive = torch.cat(chunks).numpy().astype(np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    # -- internals ----------------------------------------------------------
    def _resolve_shape(self, arr: np.ndarray) -> None:
        if arr.ndim == 3:
            if int(self.n_tabular):
                raise ValueError("3-D tensor input carries no tabular features; set n_tabular=0.")
            for name, value, actual in (
                ("n_channels", self.n_channels, arr.shape[1]),
                ("n_times", self.n_times, arr.shape[2]),
            ):
                if value is not None and int(value) != actual:
                    raise ValueError(f"{name}={value} but X has {actual}.")
            self.n_channels_, self.n_times_ = int(arr.shape[1]), int(arr.shape[2])
        elif arr.ndim == 2:
            if self.n_channels is None or self.n_times is None:
                raise ValueError("2-D (hybrid) input needs n_channels and n_times.")
            self.n_channels_, self.n_times_ = int(self.n_channels), int(self.n_times)
        else:
            raise ValueError(f"X must be 3-D (epochs, channels, times) or 2-D hybrid; got {arr.ndim}-D.")
        self.n_features_in_ = int(arr.shape[1])

    def _split_input(self, arr: np.ndarray):
        """Return ``(tensor (n, C, T), tabular (n, n_tabular) or None)`` as float32."""
        if arr.ndim == 3:
            if arr.shape[1:] != (self.n_channels_, self.n_times_):
                raise ValueError(
                    f"X has shape {arr.shape[1:]}; fitted on ({self.n_channels_}, {self.n_times_})."
                )
            return np.ascontiguousarray(arr), None
        n_tensor = self.n_channels_ * self.n_times_
        expected = n_tensor + int(self.n_tabular)
        if arr.ndim != 2 or arr.shape[1] != expected:
            raise ValueError(f"hybrid input must be 2-D with {expected} columns; got {arr.shape}.")
        tensor = np.ascontiguousarray(arr[:, :n_tensor]).reshape(-1, self.n_channels_, self.n_times_)
        tabular = np.ascontiguousarray(arr[:, n_tensor:]) if int(self.n_tabular) else None
        return tensor, tabular

    def _train(self, tensor, tabular, targets, rng) -> list[dict]:
        module = self.module_
        batch_size = int(self.batch_size)
        train_idx, val_idx = keras_validation_split(len(targets), float(self.validation_split))
        optimizer = torch.optim.Adam(
            module.parameters(), lr=float(self.learning_rate), eps=_KERAS_ADAM_EPSILON,
        )
        stopper = KerasEarlyStopping(self.patience) if val_idx is not None else None
        x_all = torch.from_numpy(tensor)
        tab_all = None if tabular is None else torch.from_numpy(tabular)
        y_all = torch.from_numpy(targets)

        history: list[dict] = []
        for epoch in range(int(self.epochs)):
            module.train()
            order = rng.permutation(train_idx)          # Keras shuffle=True, every epoch
            loss_sum = 0.0
            for start in range(0, order.size, batch_size):
                rows = torch.from_numpy(order[start:start + batch_size])
                optimizer.zero_grad()
                loss = self._loss(module, x_all, tab_all, y_all, rows)
                loss.backward()
                optimizer.step()
                module.apply_max_norm()                 # Keras constrains right after each update
                loss_sum += float(loss.detach()) * len(rows)
            record = {"epoch": epoch, "loss": loss_sum / order.size}
            if stopper is not None:
                record["val_loss"] = self._validation_loss(module, x_all, tab_all, y_all, val_idx)
            history.append(record)
            if self.verbose:
                log.info("eegnet_torch epoch %d/%d %s", epoch + 1, int(self.epochs), record)
            if stopper is not None and stopper.update(epoch, record["val_loss"], module):
                break

        if stopper is not None:
            stopper.restore(module)
            self.best_epoch_ = stopper.best_epoch
        module.eval()
        return history

    @staticmethod
    def _loss(module, x_all, tab_all, y_all, rows) -> torch.Tensor:
        logits = module(x_all[rows], None if tab_all is None else tab_all[rows]).squeeze(1)
        return F.binary_cross_entropy_with_logits(logits, y_all[rows]) + module.regularization_loss()

    def _validation_loss(self, module, x_all, tab_all, y_all, val_idx) -> float:
        """Keras ``val_loss``: inference-mode BCE over the split plus the L2 penalty."""
        module.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, val_idx.size, int(self.batch_size)):
                rows = torch.from_numpy(val_idx[start:start + int(self.batch_size)])
                logits = module(x_all[rows], None if tab_all is None else tab_all[rows]).squeeze(1)
                total += float(F.binary_cross_entropy_with_logits(logits, y_all[rows], reduction="sum"))
            return total / val_idx.size + float(module.regularization_loss())


# ---------------------------------------------------------------------------
# Registry hooks (same contract as models/eegnet.py)
# ---------------------------------------------------------------------------
def make_eegnet_torch(cfg: dict, *, input_shape: tuple[int, int] | int, **_kwargs):
    """Return an :class:`EEGNetTorchClassifier` for the training driver.

    ``input_shape`` is ``(n_channels, n_times)`` for tensor-only runs, or the
    flattened hybrid feature count when ``cfg["_neural_hybrid_input"]`` is set.
    """
    mcfg = cfg.get("modeling", {})
    ecfg = mcfg.get("eegnet_torch", {})
    monitor = ecfg.get("early_stopping_monitor", "val_loss")
    if monitor != "val_loss":
        raise ValueError(f"eegnet_torch early-stops on 'val_loss' only; got {monitor!r}.")

    hybrid = cfg.get("_neural_hybrid_input")
    if hybrid:
        n_channels = int(hybrid["n_channels"])
        n_times = int(hybrid["n_times"])
        n_tabular = int(hybrid["n_tabular_features"])
        if int(input_shape) != n_channels * n_times + n_tabular:
            raise ValueError(
                f"hybrid input_shape={input_shape} does not match "
                f"{n_channels} x {n_times} + {n_tabular} tabular features."
            )
    else:
        n_channels, n_times = int(input_shape[0]), int(input_shape[1])
        n_tabular = 0

    seed = mcfg.get("random_state")
    return EEGNetTorchClassifier(
        n_channels=n_channels,
        n_times=n_times,
        n_tabular=n_tabular,
        epochs=int(ecfg.get("epochs", 50)),
        batch_size=int(ecfg.get("batch_size", 16)),
        validation_split=float(ecfg.get("validation_split", 0.2)),
        patience=int(ecfg.get("patience", 10)),
        random_state=None if seed is None else int(seed),
        verbose=int(ecfg.get("verbose", 0)),
    )


def param_grid(cfg: dict) -> dict:
    ecfg = cfg.get("modeling", {}).get("eegnet_torch", {})
    grid = ecfg.get("param_grid")
    if grid:
        # Accept a grid cloned from configs/eegnet.yaml: drop scikeras's
        # ``model__`` routing prefix, which the torch estimator does not use.
        return {key.removeprefix("model__"): value for key, value in grid.items()}
    return {
        "f1": [8],
        "depth_multiplier": [2],
        "f2": [16],
        "kernel_length": [64],
        "separable_kernel_length": [16],
        "dropout_rate": [0.5],
        "tabular_units": [32],
        "fusion_units": [32],
        "learning_rate": [1e-3],
        "norm_rate": [0.25],
    }
