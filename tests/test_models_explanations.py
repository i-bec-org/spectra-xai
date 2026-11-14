# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 5: new models and explanation plots."""

import warnings
from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import PostHocAnalysis
from spectraxai.models import Model, StandardModel

NEW_MODELS = [
    Model.RIDGE,
    Model.LASSO,
    Model.ELASTIC_NET,
    Model.KNN,
    Model.GPR,
    Model.MLP,
]
LINEAR = {Model.RIDGE, Model.LASSO, Model.ELASTIC_NET}


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings on tiny data
        yield


def _pls(n: int = 3) -> StandardModel:
    return StandardModel(Model.PLS, init_hyperparameters={"n_components": n})


# -- New models ------------------------------------------------------------------------


@pytest.mark.parametrize("model", NEW_MODELS, ids=lambda m: m.name)
def test_new_models_fit_and_report(synthetic: Dataset, model: Model) -> None:
    sm = StandardModel(model, random_state=0, cv=3)
    res = sm.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert res.loc[0, "R2"] > 0.5
    assert res.loc[0, "model_name"] == model.name
    for key in sm._REPORTED_HYPERPARAMETERS[model]:
        assert key in res.columns
    assert sm.pipeline_ is not None and "scaler" in sm.pipeline_.named_steps
    expected = "coef" if model in LINEAR else None
    assert res.loc[0, "importance_method"] == expected


@pytest.mark.parametrize("model", NEW_MODELS, ids=lambda m: m.name)
def test_new_models_support_multiple_outputs(
    synthetic2d: Dataset, model: Model
) -> None:
    res = StandardModel(model, random_state=0, cv=3).fit_and_predict(
        synthetic2d, idx_trn=np.arange(60)
    )
    assert list(res["output"]) == synthetic2d.Y_names


@pytest.mark.parametrize("model", NEW_MODELS, ids=lambda m: m.name)
def test_new_models_clone(model: Model) -> None:
    original = StandardModel(model, random_state=1)
    copy = cast(StandardModel, clone(original))
    assert copy.model == model and copy.random_state == 1


def test_linear_importance_is_the_standardized_coefficient(synthetic: Dataset) -> None:
    sm = StandardModel(Model.RIDGE, init_hyperparameters={"alpha": 1.0}).fit(synthetic)
    importance, method = sm.feature_importance()
    assert method == "coef"
    np.testing.assert_allclose(
        cast(np.ndarray, importance), np.abs(sm.best_model.coef_).ravel()
    )


@pytest.mark.parametrize("model", [Model.MLP, Model.GPR])
def test_stochastic_new_models_are_reproducible(
    synthetic: Dataset, model: Model
) -> None:
    preds = [
        StandardModel(model, random_state=3, cv=3).fit(synthetic).predict(synthetic.X)
        for _ in range(2)
    ]
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-6)


# -- Importance over the mean spectrum -------------------------------------------------


def test_plot_importance_spectrum(synthetic: Dataset) -> None:
    model = _pls().fit(synthetic)
    fig, (other, ax) = plt.subplots(1, 2)
    n_figures = len(plt.get_fignums())
    main, twin = PostHocAnalysis(synthetic).plot_importance_spectrum(
        model, ax=ax, color="purple"
    )
    assert main is ax and twin is not ax and not other.has_data()
    assert len(plt.get_fignums()) == n_figures
    assert twin.get_shared_x_axes().joined(twin, ax)
    (line,) = twin.lines
    assert to_rgba(line.get_color()) == to_rgba("purple")
    vip = cast(np.ndarray, model.feature_importance()[0])
    np.testing.assert_allclose(cast(np.ndarray, line.get_ydata()), vip)


def test_plot_importance_spectrum_needs_an_importance(synthetic: Dataset) -> None:
    xai = PostHocAnalysis(synthetic)
    svr = StandardModel(Model.SVR, init_hyperparameters={"C": 1.0}).fit(synthetic)
    with pytest.raises(SpectraXAIValueError, match="no feature importance"):
        xai.plot_importance_spectrum(svr)
    with pytest.raises(SpectraXAIValueError):
        xai.plot_importance_spectrum(np.ones(3))
    _, twin = xai.plot_importance_spectrum(np.ones(synthetic.n_features))
    assert twin.get_ylabel() == "Importance"


# -- PLS coefficients ------------------------------------------------------------------


def test_plot_pls_coefficients(synthetic2d: Dataset) -> None:
    model = _pls().fit(synthetic2d)
    _, ax = plt.subplots()
    result = PostHocAnalysis(synthetic2d).plot_pls_coefficients(model, ax=ax, lw=3)
    assert result is ax
    lines = [line for line in ax.lines if line.get_label() in synthetic2d.Y_names]
    assert [line.get_label() for line in lines] == synthetic2d.Y_names
    assert all(line.get_linewidth() == 3 for line in lines)
    coef = np.atleast_2d(model.best_model.coef_)
    for line, row in zip(lines, coef):
        np.testing.assert_allclose(cast(np.ndarray, line.get_ydata()), row)
    assert ax.get_legend() is not None


def test_plot_pls_coefficients_accepts_plsregression(synthetic: Dataset) -> None:
    pls = PLSRegression(n_components=2).fit(synthetic.X, synthetic.Y)
    ax = PostHocAnalysis(synthetic).plot_pls_coefficients(pls)
    assert ax.get_legend() is None and ax.get_ylabel() == "PLS coefficient"


def test_plot_pls_coefficients_rejects_other_models(synthetic: Dataset) -> None:
    xai = PostHocAnalysis(synthetic)
    ridge = StandardModel(Model.RIDGE, init_hyperparameters={"alpha": 1.0})
    with pytest.raises(SpectraXAIValueError, match="PLS"):
        xai.plot_pls_coefficients(ridge.fit(synthetic))
    narrow = Dataset(synthetic.X[:, :5], synthetic.Y)
    with pytest.raises(SpectraXAIValueError, match="features"):
        xai.plot_pls_coefficients(_pls().fit(narrow))


# -- SAGE ------------------------------------------------------------------------------


def test_plot_sage_importance_from_explanation(synthetic: Dataset) -> None:
    n = synthetic.n_features
    explanation = SimpleNamespace(values=np.linspace(0, 1, n), std=np.full(n, 0.1))
    ax = PostHocAnalysis(synthetic).plot_sage_importance(explanation, color="green")
    bars = cast(list[Rectangle], ax.patches)
    np.testing.assert_allclose([b.get_height() for b in bars], explanation.values)
    assert to_rgba(bars[0].get_facecolor()) == to_rgba("green")
    assert ax.containers and ax.get_ylabel() == "SAGE value"


def test_plot_sage_importance_from_model(synthetic: Dataset) -> None:
    small = synthetic.subset(np.arange(30))
    ax = PostHocAnalysis(small).plot_sage_importance(_pls().fit(small))
    assert len(ax.patches) == small.n_features


# -- SHAP (optional extra) -------------------------------------------------------------


def test_shap_values_are_additive(synthetic: Dataset) -> None:
    pytest.importorskip("shap")
    model = _pls().fit(synthetic)
    X = synthetic.X[:15]
    explanation = PostHocAnalysis(synthetic, random_state=0).shap_importance(
        model, X=X, background=20
    )
    assert explanation.values.shape == (15, synthetic.n_features)
    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        np.ravel(model.predict(X)),
        atol=1e-6,
    )


def test_plot_shap_summary(synthetic2d: Dataset) -> None:
    pytest.importorskip("shap")
    model = _pls().fit(synthetic2d)
    xai = PostHocAnalysis(synthetic2d, random_state=0)
    explanation = xai.shap_importance(model, X=synthetic2d.X[:10], background=10)
    assert explanation.values.shape == (10, synthetic2d.n_features, 2)
    _, ax = plt.subplots()
    result = xai.plot_shap_summary(explanation, ax=ax, output=1)
    assert result is ax
    heights = [b.get_height() for b in cast(list[Rectangle], ax.patches)]
    np.testing.assert_allclose(heights, np.abs(explanation.values[:, :, 1]).mean(0))
    assert ax.get_ylabel() == "mean(|SHAP value|)"
