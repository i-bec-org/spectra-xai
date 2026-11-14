# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The plotting contract: draw into supplied axes, never touch global state."""

import inspect
from collections.abc import Callable
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.plotting import STYLE, style
from spectraxai.utils import scatter_plot


@pytest.fixture
def fitted(synthetic: Dataset) -> StandardModel:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    model.fit(synthetic)
    return model


def _importance(model: StandardModel) -> np.ndarray:
    return model.best_model.coef_.ravel()


def _y_ranges(dataset: Dataset) -> list[np.ndarray]:
    return [
        np.linspace(dataset.Y[:, i].min() - 1, dataset.Y[:, i].max() + 1, 3)
        for i in range(dataset.n_outputs)
    ]


def _snapshot() -> dict[str, Any]:
    return {k: v for k, v in mpl.rcParams.items() if k != "backend"}


# -- Global state ---------------------------------------------------------------


def test_constructing_analyses_does_not_change_rcparams(synthetic: Dataset) -> None:
    before = _snapshot()
    PreHocAnalysis(synthetic)
    PostHocAnalysis(synthetic)
    assert _snapshot() == before


def test_plots_restore_rcparams(synthetic: Dataset, fitted: StandardModel) -> None:
    before = _snapshot()
    PreHocAnalysis(synthetic).mean_spectrum()
    PostHocAnalysis(synthetic).bar_plot_importance(_importance(fitted))
    scatter_plot(synthetic.Y, fitted.predict(synthetic.X))
    assert _snapshot() == before


def test_created_figures_use_the_library_style(synthetic: Dataset) -> None:
    with plt.style.context(STYLE):
        expected = to_rgba(mpl.rcParams["axes.edgecolor"])
    ax = PreHocAnalysis(synthetic).mean_spectrum()
    assert to_rgba(ax.spines["left"].get_edgecolor()) == expected


def test_supplied_axes_keep_the_callers_style(synthetic: Dataset) -> None:
    with plt.style.context("classic"):
        _, ax = plt.subplots()
        expected = to_rgba(mpl.rcParams["axes.edgecolor"])
    PreHocAnalysis(synthetic).mean_spectrum(ax=ax)
    assert to_rgba(ax.spines["left"].get_edgecolor()) == expected


def test_style_context_manager_restores_rcparams() -> None:
    before = _snapshot()
    with style():
        assert mpl.rcParams["axes.grid"] is True
    assert _snapshot() == before


# -- Drawing into supplied axes ---------------------------------------------------

PlotCall = Callable[[Dataset, StandardModel, Axes], Any]

_SINGLE_AXES_PLOTS: dict[str, PlotCall] = {
    "prehoc.bar_plot_importance": lambda d, m, ax: PreHocAnalysis(
        d
    ).bar_plot_importance(FeatureRanking.CORR, ax=ax),
    "prehoc.mean_spectrum": lambda d, m, ax: PreHocAnalysis(d).mean_spectrum(ax=ax),
    "prehoc.mean_spectrum_by_range": lambda d, m, ax: PreHocAnalysis(
        d
    ).mean_spectrum_by_range(_y_ranges(d), axes=ax),
    "prehoc.correlogram": lambda d, m, ax: PreHocAnalysis(d).correlogram(
        top=1, axes=ax
    ),
    "posthoc.bar_plot_importance": lambda d, m, ax: PostHocAnalysis(
        d
    ).bar_plot_importance(_importance(m), ax=ax),
    "posthoc.bar_plot_permutation_importance": lambda d, m, ax: PostHocAnalysis(
        d, random_state=0
    ).bar_plot_permutation_importance(m.best_model, ax=ax),
    "utils.scatter_plot": lambda d, m, ax: scatter_plot(d.Y, m.predict(d.X), ax=ax),
}


@pytest.mark.parametrize("name", sorted(_SINGLE_AXES_PLOTS))
def test_plot_draws_into_supplied_axes(
    name: str, synthetic: Dataset, fitted: StandardModel
) -> None:
    fig, (other, ax) = plt.subplots(1, 2)
    layout_before = fig.subplotpars.left, fig.subplotpars.wspace
    n_figures = len(plt.get_fignums())

    result = _SINGLE_AXES_PLOTS[name](synthetic, fitted, ax)

    assert ax in np.ravel(np.asarray(result, dtype=object))
    assert ax.has_data()
    assert not other.has_data()
    assert len(plt.get_fignums()) == n_figures, "no new figure should be created"
    assert (fig.subplotpars.left, fig.subplotpars.wspace) == layout_before


def test_circular_bar_plot_draws_into_supplied_polar_axes(
    synthetic: Dataset, fitted: StandardModel
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2, projection="polar")
    n_figures = len(plt.get_fignums())
    result = PostHocAnalysis(synthetic).circular_bar_plot_importance(
        np.abs(_importance(fitted)), ax=ax, top=5
    )
    assert result is ax
    assert len(ax.patches) == 5
    assert ax.axison, "the caller's axes frame must not be hidden"
    assert len(plt.get_fignums()) == n_figures


def test_circular_bar_plot_rejects_cartesian_axes(
    synthetic: Dataset, fitted: StandardModel
) -> None:
    _, ax = plt.subplots()
    with pytest.raises(SpectraXAIValueError, match="polar"):
        PostHocAnalysis(synthetic).circular_bar_plot_importance(
            np.abs(_importance(fitted)), ax=ax
        )


@pytest.mark.parametrize("shape", [(2,), (2, 1), (1, 2)])
def test_multi_output_plots_accept_any_axes_shape(
    synthetic2d: Dataset, shape: tuple[int, ...]
) -> None:
    xai = PreHocAnalysis(synthetic2d)
    _, axes = plt.subplots(*(shape if len(shape) == 2 else (shape[0], 1)))
    axes = np.asarray(axes).reshape(shape)
    result = xai.bar_plot_importance(FeatureRanking.CORR, ax=axes)
    assert list(result) == list(axes.ravel())
    result = xai.mean_spectrum_by_range(_y_ranges(synthetic2d), axes=axes)
    assert result.shape == (2, 1)
    assert all(ax.has_data() for ax in axes.ravel())


def test_correlogram_into_supplied_grid(synthetic2d: Dataset) -> None:
    _, axes = plt.subplots(2, 3)
    result = PreHocAnalysis(synthetic2d).correlogram(top=3, axes=axes)
    assert result.shape == (2, 3)
    assert all(ax.has_data() for ax in axes.ravel())


@pytest.mark.parametrize(
    "call",
    [
        lambda xai, axes: xai.bar_plot_importance(FeatureRanking.CORR, ax=axes),
        lambda xai, axes: xai.mean_spectrum_by_range(_y_ranges(xai.dataset), axes=axes),
        lambda xai, axes: xai.correlogram(top=2, axes=axes),
    ],
)
def test_wrong_number_of_axes_raises(synthetic2d: Dataset, call: Any) -> None:
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError):
        call(PreHocAnalysis(synthetic2d), axes)


def test_correlogram_validates_top(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError):
        PreHocAnalysis(synthetic).correlogram(top=synthetic.n_features + 1)


# -- Keyword arguments are forwarded ------------------------------------------------


def test_bar_plot_forwards_kwargs(synthetic: Dataset, fitted: StandardModel) -> None:
    ax = PostHocAnalysis(synthetic).bar_plot_importance(
        _importance(fitted), color="red"
    )
    assert all(to_rgba(p.get_facecolor()) == to_rgba("red") for p in ax.patches)


def test_mean_spectrum_forwards_kwargs(synthetic: Dataset) -> None:
    ax = PreHocAnalysis(synthetic).mean_spectrum(
        excluded_bands=[(1000, 1200)], c="green", lw=3, label="mean"
    )
    assert len(ax.lines) == 2  # split around the excluded band
    assert all(line.get_linewidth() == 3 for line in ax.lines)
    assert all(to_rgba(line.get_color()) == to_rgba("green") for line in ax.lines)
    assert [line.get_label() for line in ax.lines].count("mean") == 1


def test_mean_spectrum_by_range_forwards_kwargs(synthetic: Dataset) -> None:
    axes = PreHocAnalysis(synthetic).mean_spectrum_by_range(_y_ranges(synthetic), lw=4)
    assert all(line.get_linewidth() == 4 for line in axes[0, 0].lines)


def test_correlogram_forwards_kwargs(synthetic: Dataset) -> None:
    axes = PreHocAnalysis(synthetic).correlogram(top=2, s=20)
    assert cast(PathCollection, axes[0, 0].collections[0]).get_sizes()[0] == 20


def test_circular_bar_plot_forwards_kwargs(
    synthetic: Dataset, fitted: StandardModel
) -> None:
    ax = PostHocAnalysis(synthetic).circular_bar_plot_importance(
        np.abs(_importance(fitted)), fc="black"
    )
    assert all(to_rgba(p.get_facecolor()) == to_rgba("black") for p in ax.patches)


def test_scatter_plot_forwards_kwargs_and_accepts_columns(
    synthetic: Dataset, fitted: StandardModel
) -> None:
    ax = scatter_plot(synthetic.Y, fitted.predict(synthetic.X), s=9)
    assert cast(PathCollection, ax.collections[0]).get_sizes()[0] == 9
    with pytest.raises(ValueError):
        scatter_plot(synthetic.Y, synthetic.Y[:-1])


# -- Documentation keeps the real signatures ------------------------------------------


def test_decorated_plots_keep_their_signature() -> None:
    assert "axes" in inspect.signature(PreHocAnalysis.correlogram).parameters
    assert "ax" in inspect.signature(scatter_plot).parameters
    assert PreHocAnalysis.correlogram.__doc__
