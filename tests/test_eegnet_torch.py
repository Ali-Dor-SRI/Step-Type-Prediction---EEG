"""PyTorch EEGNet port (``models/eegnet_torch.py``).

Covers the network (output shape, max-norm constraints), parity with the Keras
original in ``models/eegnet.py``, the scikit-learn wrapper, and an end-to-end
``--model eegnet_torch`` training run.

Parity is checked in two layers:

* against constants recorded from the Keras model -- these run everywhere,
  including CI, which installs no TensorFlow;
* live against the Keras model (parameter counts, the max-norm constraint, and
  a forward pass with the Keras weights copied in) -- skipped wherever
  TensorFlow is not importable.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.base import clone

from eeg_steptype.models.eegnet_torch import (
    EEGNetTorch,
    EEGNetTorchClassifier,
    KerasEarlyStopping,
    keras_max_norm_,
    keras_validation_split,
    param_grid,
)


# Trainable / non-trainable (BatchNorm moving statistics) parameter counts of
# the Keras EEGNet in models/eegnet.py at its default hyperparameters (f1=8,
# depth_multiplier=2, f2=16, kernel_length=64, separable_kernel_length=16,
# tabular_units=32, fusion_units=32). Recorded 2026-09-11 from
# make_eegnet(cfg, input_shape=...).model() under TensorFlow 2.21.0 /
# Keras 3.14.1 (.venv312). Key: (n_channels, n_times, n_tabular); n_tabular > 0
# is the hybrid (tensor + tabular) input.
KERAS_RECORDED_PARAM_COUNTS = {
    (8, 256, 0): (1361, 80),
    (8, 256, 12): (6833, 80),
    (4, 97, 5): (3985, 80),        # odd length: pooling floors
    (3, 40, 0): (977, 80),         # kernel_length 64 clamped to n_times
    (64, 2049, 0): (3153, 80),     # the real tensor: 64 channels x 2049 samples (0-2 s at 1024 Hz)
    (64, 2049, 300): (45617, 80),
}
LIVE_FORWARD_SHAPES = [(8, 256, 0), (4, 97, 5), (64, 2049, 300)]


def _shape_id(shape: tuple[int, int, int]) -> str:
    return "x".join(str(value) for value in shape)


def _param_counts(module: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = sum(
        b.numel() for name, b in module.named_buffers() if not name.endswith("num_batches_tracked")
    )
    return trainable, non_trainable


def _require_keras() -> None:
    try:
        import scikeras  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError:
        pytest.skip(
            "live Keras parity needs TensorFlow + scikeras (the .venv312 environment); "
            "the recorded-constant tests cover environments without them, such as CI"
        )


def _keras_eegnet(n_channels: int, n_times: int, n_tabular: int):
    """The Keras model exactly as the training driver builds it (default hyperparameters)."""
    _require_keras()
    from eeg_steptype.models.eegnet import make_eegnet

    cfg: dict = {"modeling": {"eegnet": {}}}
    if n_tabular:
        cfg["_neural_hybrid_input"] = {
            "n_channels": n_channels,
            "n_times": n_times,
            "n_tabular_features": n_tabular,
        }
        return make_eegnet(cfg, input_shape=n_channels * n_times + n_tabular).model()
    return make_eegnet(cfg, input_shape=(n_channels, n_times)).model()


def _randomize_batchnorm(keras_model, rng: np.random.Generator) -> None:
    """Give every Keras BatchNorm non-trivial parameters and statistics to copy."""
    for name in ("temporal_bn", "spatial_bn", "separable_bn"):
        layer = keras_model.get_layer(name)
        gamma, beta, mean, var = layer.get_weights()
        layer.set_weights([
            rng.uniform(0.5, 1.5, gamma.shape),
            rng.normal(0.0, 0.2, beta.shape),
            rng.normal(0.0, 0.2, mean.shape),
            rng.uniform(0.5, 2.0, var.shape),
        ])


def _copy_keras_weights(keras_model, module: EEGNetTorch) -> None:
    """Load Keras (channels-last) kernels into the PyTorch (channels-first) layers."""

    def weights(name):
        return keras_model.get_layer(name).get_weights()

    def load(param, array):
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)))

    (kernel,) = weights("temporal_conv")                     # (1, K, 1, f1)
    load(module.temporal_conv.weight, kernel.transpose(3, 2, 0, 1))
    (kernel,) = weights("spatial_depthwise")                 # (C, 1, f1, D)
    n_channels, _, f1, depth = kernel.shape
    # Keras orders depthwise outputs f1_index * D + d, as PyTorch's grouped conv does.
    load(
        module.spatial_depthwise.weight,
        kernel.reshape(n_channels, 1, f1 * depth).transpose(2, 1, 0)[..., None],
    )
    depthwise, pointwise = weights("separable_conv")          # (1, K2, f1*D, 1), (1, 1, f1*D, f2)
    load(module.separable_depthwise.weight, depthwise.transpose(2, 3, 0, 1))
    load(module.separable_pointwise.weight, pointwise.transpose(3, 2, 0, 1))
    for name, bn in (
        ("temporal_bn", module.temporal_bn),
        ("spatial_bn", module.spatial_bn),
        ("separable_bn", module.separable_bn),
    ):
        gamma, beta, mean, var = weights(name)
        load(bn.weight, gamma)
        load(bn.bias, beta)
        load(bn.running_mean, mean)
        load(bn.running_var, var)
    for name, dense in (
        ("tabular_dense", module.tabular_dense),
        ("fusion_dense", module.fusion_dense),
        ("class_probability", module.classifier),
    ):
        if dense is not None:
            kernel, bias = weights(name)                      # (in, out), (out,)
            load(dense.weight, kernel.T)
            load(dense.bias, bias)


def _constrained_weight_norms(module: EEGNetTorch) -> dict:
    """``{layer: (per-filter/unit L2 norms, bound)}`` for every Keras max-norm weight."""
    return {
        "spatial_depthwise": (module.spatial_depthwise.weight.flatten(1).norm(dim=1), 1.0),
        "fusion_dense": (module.fusion_dense.weight.norm(dim=1), module.norm_rate),
        "classifier": (module.classifier.weight.norm(dim=1), module.norm_rate),
    }


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_tabular", [0, 7])
def test_forward_returns_one_logit_per_epoch(n_tabular):
    module = EEGNetTorch(6, 128, n_tabular=n_tabular).eval()
    x = torch.randn(5, 6, 128)
    x_tabular = torch.randn(5, n_tabular) if n_tabular else None

    assert module(x, x_tabular).shape == (5, 1)
    assert module.features(x).shape == (5, 16 * (128 // 4 // 8))   # f2 x pooled time steps
    if n_tabular:
        with pytest.raises(ValueError, match="x_tabular"):
            module(x)


def test_keras_max_norm_matches_the_keras_formula():
    rng = np.random.default_rng(0)
    weight = rng.normal(0.0, 1.0, (6, 1, 5, 1)).astype(np.float32)
    norms = np.sqrt((weight ** 2).sum(axis=(1, 2, 3), keepdims=True))
    expected = weight * np.clip(norms, 0.0, 1.0) / (1e-7 + norms)   # keras MaxNorm.__call__

    out = torch.from_numpy(weight.copy())
    keras_max_norm_(out, 1.0, dims=(1, 2, 3))

    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)
    assert np.sqrt((out.numpy() ** 2).sum(axis=(1, 2, 3))).max() <= 1.0 + 1e-6


def test_max_norm_bounds_hold_after_training_steps(monkeypatch):
    rng = np.random.default_rng(1)
    n, n_channels, n_times, n_tabular = 32, 4, 64, 6
    X = np.concatenate(
        [rng.normal(size=(n, n_channels * n_times)), rng.normal(size=(n, n_tabular))], axis=1,
    )
    y = np.tile([0, 1], n // 2)
    # lr=0.5 moves every weight by ~0.5 per Adam step, so without the constraint
    # these norms blow far past their bounds within the 6 steps (2 batches x 3
    # epochs). No validation split: the check sees the last step's weights.
    clf = EEGNetTorchClassifier(
        n_channels=n_channels, n_times=n_times, n_tabular=n_tabular,
        learning_rate=0.5, epochs=3, validation_split=0.0, random_state=0,
    )

    with monkeypatch.context() as patched:          # control: constraint switched off
        patched.setattr(EEGNetTorch, "apply_max_norm", lambda self: None)
        unconstrained = clone(clf).fit(X, y).module_
    for name, (norms, bound) in _constrained_weight_norms(unconstrained).items():
        assert norms.max() > bound, f"control never exceeded the {name} bound; the test is not sensitive"

    constrained = clone(clf).fit(X, y).module_
    for name, (norms, bound) in _constrained_weight_norms(constrained).items():
        assert norms.max() <= bound * (1 + 1e-5), f"{name}: max norm {norms.max():.5f} > {bound}"


# ---------------------------------------------------------------------------
# Parity with the Keras original
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", sorted(KERAS_RECORDED_PARAM_COUNTS), ids=_shape_id)
def test_parameter_counts_match_recorded_keras_eegnet(shape):
    n_channels, n_times, n_tabular = shape
    module = EEGNetTorch(n_channels, n_times, n_tabular=n_tabular)
    assert _param_counts(module) == KERAS_RECORDED_PARAM_COUNTS[shape]


@pytest.mark.parametrize("shape", sorted(KERAS_RECORDED_PARAM_COUNTS), ids=_shape_id)
def test_parameter_counts_match_live_keras_eegnet(shape):
    keras_model = _keras_eegnet(*shape)
    keras_counts = (
        sum(int(np.prod(w.shape)) for w in keras_model.trainable_weights),
        sum(int(np.prod(w.shape)) for w in keras_model.non_trainable_weights),
    )
    assert keras_counts == KERAS_RECORDED_PARAM_COUNTS[shape], "recorded constant is stale"
    n_channels, n_times, n_tabular = shape
    assert _param_counts(EEGNetTorch(n_channels, n_times, n_tabular=n_tabular)) == keras_counts


@pytest.mark.parametrize("shape", LIVE_FORWARD_SHAPES, ids=_shape_id)
def test_forward_matches_live_keras_with_copied_weights(shape):
    keras_model = _keras_eegnet(*shape)
    import keras

    n_channels, n_times, n_tabular = shape
    rng = np.random.default_rng(0)
    _randomize_batchnorm(keras_model, rng)
    module = EEGNetTorch(n_channels, n_times, n_tabular=n_tabular).eval()
    _copy_keras_weights(keras_model, module)

    x = rng.normal(size=(5, n_channels, n_times)).astype(np.float32)
    x_tabular = rng.normal(size=(5, n_tabular)).astype(np.float32)
    keras_input = np.concatenate([x.reshape(5, -1), x_tabular], axis=1) if n_tabular else x
    keras_flatten = keras.Model(keras_model.inputs, keras_model.get_layer("flatten").output)
    with torch.no_grad():
        torch_flat = module.features(torch.from_numpy(x)).numpy()
        logits = module(torch.from_numpy(x), torch.from_numpy(x_tabular) if n_tabular else None)
        torch_proba = torch.sigmoid(logits).numpy().ravel()

    np.testing.assert_allclose(
        torch_flat, np.asarray(keras_flatten([keras_input], training=False)), atol=1e-5,
    )
    np.testing.assert_allclose(
        torch_proba, np.asarray(keras_model([keras_input], training=False)).ravel(), atol=1e-5,
    )


def test_max_norm_matches_the_live_keras_constraint():
    _require_keras()
    import keras

    rng = np.random.default_rng(2)
    depthwise = rng.normal(0.0, 1.0, (5, 1, 3, 2)).astype(np.float32)   # Keras (C, 1, f1, D)
    dense = rng.normal(0.0, 1.0, (7, 4)).astype(np.float32)             # Keras (in, out)
    keras_depthwise = np.asarray(keras.constraints.MaxNorm(1.0)(depthwise))
    keras_dense = np.asarray(keras.constraints.MaxNorm(0.25)(dense))

    torch_depthwise = torch.from_numpy(                                  # PyTorch (f1*D, 1, C, 1)
        np.ascontiguousarray(depthwise.reshape(5, 1, 6).transpose(2, 1, 0)[..., None])
    )
    keras_max_norm_(torch_depthwise, 1.0, dims=(1, 2, 3))
    torch_dense = torch.from_numpy(np.ascontiguousarray(dense.T))       # PyTorch (out, in)
    keras_max_norm_(torch_dense, 0.25, dims=(1,))

    np.testing.assert_allclose(
        torch_depthwise.numpy()[:, 0, :, 0].T.reshape(5, 1, 3, 2), keras_depthwise,
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(torch_dense.numpy().T, keras_dense, rtol=1e-6, atol=1e-7)


def test_validation_split_takes_the_trailing_fraction_like_keras():
    train, val = keras_validation_split(10, 0.2)
    assert train.tolist() == list(range(8)) and val.tolist() == [8, 9]
    train, val = keras_validation_split(21, 0.2)                 # floor(21 * 0.8) = 16
    assert train.tolist() == list(range(16)) and val.tolist() == list(range(16, 21))
    assert keras_validation_split(5, 0.0)[1] is None
    with pytest.raises(ValueError, match="not sufficient"):
        keras_validation_split(1, 0.2)


def test_early_stopping_follows_keras_patience_and_restores_best():
    module = torch.nn.Linear(1, 1)

    def run(losses, patience):
        stopper = KerasEarlyStopping(patience)
        stopped = None
        for epoch, loss in enumerate(losses):
            with torch.no_grad():
                module.weight.fill_(float(epoch))            # tag the weights with the epoch
            if stopper.update(epoch, loss, module):
                stopped = epoch
                break
        stopper.restore(module)
        return stopped, stopper.best_epoch, module.weight.item()

    # Best at epoch 1; epochs 2-4 do not improve, so wait reaches patience=3 at epoch 4.
    assert run([1.0, 0.8, 0.9, 0.85, 0.95, 0.7], patience=3) == (4, 1, 1.0)
    # No early stop, but Keras 3 still restores the best epoch when training ends.
    assert run([0.5, 0.6, 0.7], patience=10) == (None, 0, 0.0)


# ---------------------------------------------------------------------------
# scikit-learn wrapper and the training driver
# ---------------------------------------------------------------------------
def test_sklearn_wrapper_fits_and_predicts_on_synthetic_epochs(synthetic_epoch_tensor):
    X, y = synthetic_epoch_tensor
    labels = np.where(y == 1, "Two", "One")                  # the driver's condition names
    clf = EEGNetTorchClassifier(epochs=30, learning_rate=1e-2, random_state=0)

    params = clf.get_params()
    assert (params["f1"], params["norm_rate"], params["patience"]) == (8, 0.25, 10)
    assert clone(clf).set_params(f1=4).get_params()["f1"] == 4

    clf.fit(X, labels)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert list(clf.classes_) == ["One", "Two"]
    assert (clf.predict(X) == labels).mean() >= 0.8          # the class-1 ramp is easy to fit
    assert clf.best_epoch_ is not None and 1 <= len(clf.history_) <= 30
    # Same random_state -> same fit.
    np.testing.assert_allclose(clone(clf).fit(X, labels).predict_proba(X), proba, atol=1e-6)


def test_runs_under_the_training_driver_search(synthetic_epoch_tensor):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from eeg_steptype.models.cnn import HybridTensorFeatureStandardizer
    from eeg_steptype.models.train import MODEL_FACTORIES, _make_search_estimator

    tensor, y = synthetic_epoch_tensor
    n, n_channels, n_times = tensor.shape
    tabular = np.random.default_rng(3).normal(size=(n, 3))
    X = np.concatenate([tensor.reshape(n, -1), tabular], axis=1)
    cfg = {
        "modeling": {"random_state": 1, "eegnet_torch": {"epochs": 3}},
        "_neural_hybrid_input": {
            "n_channels": n_channels, "n_times": n_times, "n_tabular_features": 3,
        },
    }
    estimator, grid = _make_search_estimator(
        MODEL_FACTORIES["eegnet_torch"], cfg, "eegnet_torch",
        scale_pos_weight=1.0, n_features=X.shape[1],
    )

    assert isinstance(estimator.named_steps["normalize"], HybridTensorFeatureStandardizer)
    classifier = estimator.named_steps["classifier"]
    assert isinstance(classifier, EEGNetTorchClassifier)
    assert (classifier.random_state, classifier.epochs, classifier.n_tabular) == (1, 3, 3)
    assert grid and all(key.startswith("classifier__") for key in grid)

    search = GridSearchCV(
        estimator, grid, scoring="roc_auc", cv=StratifiedKFold(2, shuffle=True, random_state=1),
    ).fit(X, y)
    assert search.best_estimator_.predict_proba(X).shape == (n, 2)


def test_overlay_clones_the_keras_eegnet_overlay(project_root):
    def load(name):
        return yaml.safe_load((project_root / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))

    keras_cfg, torch_cfg = load("eegnet"), load("eegnet_torch")
    keras_model_cfg = keras_cfg["modeling"].pop("eegnet")
    torch_model_cfg = torch_cfg["modeling"].pop("eegnet_torch")
    keras_grid = keras_model_cfg.pop("param_grid")
    torch_grid = torch_model_cfg.pop("param_grid")

    assert keras_cfg["modeling"].pop("default_model") == "eegnet"
    assert torch_cfg["modeling"].pop("default_model") == "eegnet_torch"
    assert torch_cfg == keras_cfg              # features, window, CV, search, logging
    assert torch_model_cfg == keras_model_cfg  # epochs, batch, split, patience, standardizer, tabular
    assert torch_grid == {key.removeprefix("model__"): value for key, value in keras_grid.items()}
    assert param_grid({"modeling": {"eegnet_torch": {"param_grid": keras_grid}}}) == torch_grid
    assert set(torch_grid) <= set(EEGNetTorchClassifier().get_params())


def test_end_to_end_train_cli_on_smoke_config_plus_overlay(tmp_path, project_root, smoke_config_path):
    """``scripts/04_train.py --model eegnet_torch`` on smoke.yaml + the overlay, synthetic data."""
    from eeg_steptype.config import apply_feature_bin_width, apply_prediction_window, load_config
    from eeg_steptype.io import src_csv_path

    from .test_smoke_pipeline import _write_synthetic_epochs

    test_overlay = tmp_path / "test_paths.yaml"
    test_overlay.write_text(yaml.safe_dump({
        "paths": {
            "data_dir": (tmp_path / "data").as_posix(),
            "outputs_dir": (tmp_path / "outputs").as_posix(),
        },
        "modeling": {"eegnet_torch": {"epochs": 3}},       # keep the run to seconds
    }), encoding="utf-8")
    overlays = [
        str(smoke_config_path),
        str(project_root / "configs" / "eegnet_torch.yaml"),
        str(test_overlay),
    ]
    # Resolve the config as the CLI will, to put the synthetic inputs where it looks.
    cfg = apply_feature_bin_width(apply_prediction_window(load_config(overlays), None), None)
    (participant,) = cfg["participants"]
    n_epochs = 14
    for condition in cfg["conditions"]:
        _write_synthetic_epochs(cfg, participant, condition, n_epochs=n_epochs)
        # The overlay requires source-space columns (require_source: true), so
        # write a synthetic eLORETA label time-course CSV for the src block.
        rng = np.random.default_rng(0 if condition == "One" else 1)
        src = {"epoch": np.arange(n_epochs)}
        for label in ("G_synthetic-lh", "G_synthetic-rh"):
            for bin_index in range(32):                    # 0-2 s in 0.0625 s bins
                src[f"{label}_bin_{bin_index}"] = rng.normal(size=n_epochs)
        path = src_csv_path(cfg, participant, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(src).to_csv(path, index=False)

    result = subprocess.run(
        [
            sys.executable, str(project_root / "scripts" / "04_train.py"),
            "--model", "eegnet_torch", "--run-id", "e2e_eegnet_torch", "--config", *overlays,
        ],
        cwd=project_root,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "model=eegnet_torch" in output and "on 1 participants" in output

    metrics = pd.read_csv(tmp_path / "outputs" / "runs" / "e2e_eegnet_torch" / "metrics.csv")
    assert set(metrics["model"]) == {"eegnet_torch"}
    assert len(metrics) == 2                            # overlay CV: 2 outer folds x 1 repeat
    assert metrics["auc"].between(0.0, 1.0).all()
    assert (metrics["n_features_final"] > 10 * 400).all()   # flattened 10 x 400 tensor + tabular
    assert "classifier__f1" in metrics["best_params"].iloc[0]
