"""Pytest fixtures shared across smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `eeg_steptype` importable when running tests without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def smoke_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "smoke.yaml"


@pytest.fixture
def synthetic_epoch_tensor() -> tuple[np.ndarray, np.ndarray]:
    """Small ``(n_epochs, n_channels, n_times)`` tensor with a learnable class signal.

    Class 1 carries a slow negative ramp on two channels (a CNV-like drift) on
    top of unit Gaussian noise. Epochs are shuffled, so any contiguous slice --
    e.g. a Keras-style trailing validation split -- holds both classes.
    """
    rng = np.random.default_rng(0)
    n_per_class, n_channels, n_times = 16, 4, 64
    y = np.repeat([0, 1], n_per_class)
    X = rng.normal(0.0, 1.0, (y.size, n_channels, n_times)).astype(np.float32)
    X[y == 1, :2, :] -= np.linspace(0.0, 1.5, n_times, dtype=np.float32)
    order = rng.permutation(y.size)
    return X[order], y[order]
