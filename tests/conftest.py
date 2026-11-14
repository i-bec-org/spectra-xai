# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures: a headless backend and a small synthetic dataset."""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from spectraxai.dataset import Dataset  # noqa: E402

WAVELENGTHS = np.arange(400, 2500, 50)  # 42 bands


def make_synthetic(n_samples: int = 80, n_outputs: int = 1, seed: int = 0) -> Dataset:
    """Build a seeded reflectance dataset whose outputs depend on absorption peaks.

    Reflectance stays inside (0, 1) so absorbance and continuum removal are valid.

    """
    rng = np.random.default_rng(seed)
    centers = np.array([700.0, 1400.0, 1900.0, 2200.0])
    depths = rng.uniform(0.05, 0.3, size=(n_samples, centers.size))
    baseline = rng.uniform(0.35, 0.6, size=(n_samples, 1))
    peaks = np.exp(-(((WAVELENGTHS[None, :] - centers[:, None]) / 80) ** 2))
    X = baseline - depths @ peaks
    X = np.clip(X + rng.normal(0, 0.003, X.shape), 0.02, 0.98)
    weights = rng.normal(size=(centers.size, n_outputs))
    Y = depths @ weights + rng.normal(0, 0.01, size=(n_samples, n_outputs))
    return Dataset(
        X,
        Y,
        X_names=[int(w) for w in WAVELENGTHS],
        Y_names=[f"prop{i + 1}" for i in range(n_outputs)],
    )


@pytest.fixture
def synthetic() -> Dataset:
    return make_synthetic()


@pytest.fixture
def synthetic2d() -> Dataset:
    return make_synthetic(n_outputs=2)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")
