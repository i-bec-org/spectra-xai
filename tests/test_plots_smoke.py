# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for plots not covered elsewhere (fast, synthetic data)."""

import numpy as np

from spectraxai.dataset import Dataset
from spectraxai.explain import PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.utils import scatter_plot


def _importance(dataset: Dataset) -> np.ndarray:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    res = model.fit_and_predict(dataset, idx_trn=np.arange(60))
    return res.loc[0, "feature_importance"]


def test_scatter_plot(synthetic: Dataset) -> None:
    y = synthetic.Y.ravel()
    y_hat = y + np.random.default_rng(0).normal(0, 0.01, y.shape)
    for use_kde in (False, True):
        ax = scatter_plot(y, y_hat, use_kde=use_kde)
        assert ax.get_xlabel() == "Observed"
        assert "RMSE" in ax.get_title()


def test_bar_plots_of_importance(synthetic: Dataset) -> None:
    importance = _importance(synthetic)
    xai = PostHocAnalysis(synthetic)
    ax = xai.bar_plot_importance(importance)
    assert len(ax.patches) == synthetic.n_features
    ax = xai.circular_bar_plot_importance(importance, top=10)
    assert len(ax.patches) == 10


def test_correlogram(synthetic2d: Dataset) -> None:
    axes = PreHocAnalysis(synthetic2d).correlogram(top=3)
    assert axes.shape == (2, 3)
    assert all(len(ax.collections) == 1 for ax in axes.ravel())
