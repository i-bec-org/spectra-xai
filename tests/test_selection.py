# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 4: feature selection, selected-bands plot and the cLHS split."""

from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from spectraxai.dataset import Dataset, DatasetSplit, _clhs, _clhs_objective
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.selection import RFE, RFECV, SelectKBest, VIPSelector
from spectraxai.spectra import SpectralPreprocessing

PEAKS = np.array([700.0, 1400.0, 1900.0, 2200.0])  # see conftest.make_synthetic


def _pls(n: int = 3) -> StandardModel:
    return StandardModel(Model.PLS, init_hyperparameters={"n_components": n})


def _near_peaks(wavelengths: np.ndarray, within: float = 150) -> np.ndarray:
    return np.abs(wavelengths[:, None] - PEAKS[None, :]).min(axis=1) <= within


# -- scikit-learn conventions ---------------------------------------------------------


@parametrize_with_checks(
    [
        SelectKBest(k=1),
        SelectKBest(FeatureRanking.F_STATISTIC, k="all"),
        VIPSelector(n_components=1),
        RFE(Ridge(), n_features_to_select=1),
    ]
)
def test_selectors_follow_sklearn_conventions(estimator: Any, check: Any) -> None:
    check(estimator)


def test_clone_standard_model_with_init_hyperparameters() -> None:
    # Regression: cloning failed whenever the model also had a default grid
    for model in [Model.SVR, Model.RF, Model.CUBIST, Model.XGBOOST]:
        original = StandardModel(model, init_hyperparameters={"n_estimators": 5})
        copy = cast(StandardModel, clone(original))
        assert copy.init_hyperparameters == {"n_estimators": 5}
    with pytest.raises(SpectraXAIValueError):
        StandardModel(
            Model.SVR,
            init_hyperparameters={"C": 1.0},
            grid_search_hyperparameters={"C": [1.0, 2.0]},
        )


# -- Dataset.select_features -----------------------------------------------------------


def test_select_features(synthetic: Dataset) -> None:
    mask = np.zeros(synthetic.n_features, dtype=bool)
    mask[[1, 5, 9]] = True
    reduced = synthetic.select_features(mask)
    assert reduced.X_names == [synthetic.X_names[i] for i in (1, 5, 9)]
    wavelengths = cast(np.ndarray, synthetic.wavelengths)
    np.testing.assert_array_equal(reduced.wavelengths, wavelengths[[1, 5, 9]])
    np.testing.assert_array_equal(
        synthetic.select_features([9, 1]).X, synthetic.X[:, [9, 1]]
    )
    for bad in (
        mask[:-1],
        np.zeros(synthetic.n_features, bool),
        [synthetic.n_features],
    ):
        with pytest.raises(SpectraXAIValueError):
            synthetic.select_features(np.asarray(bad))


# -- Selectors -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector",
    [
        RFE(_pls(), n_features_to_select=8, step=4),
        RFE(
            StandardModel(Model.SVR, init_hyperparameters={"C": 4.0}),
            n_features_to_select=8,
            step=8,
            random_state=0,
        ),
        RFE(
            StandardModel(
                Model.RF, init_hyperparameters={"n_estimators": 30}, random_state=0
            ),
            n_features_to_select=8,
            step=8,
        ),
        VIPSelector(n_components=3),
        SelectKBest(k=8),
        SelectKBest(FeatureRanking.MI, k=8, random_state=0),
    ],
    ids=["rfe-pls", "rfe-svr", "rfe-rf", "vip", "kbest-corr", "kbest-mi"],
)
def test_selectors_find_the_absorption_bands(synthetic: Dataset, selector: Any) -> None:
    reduced = selector.fit(synthetic).transform(synthetic)
    assert isinstance(reduced, Dataset)
    assert reduced.n_features == int(selector.get_support().sum())
    wavelengths = selector.selected_wavelengths(synthetic)
    np.testing.assert_array_equal(reduced.wavelengths, wavelengths)
    # most kept bands should lie on the synthetic absorption features
    assert _near_peaks(wavelengths).mean() >= 0.75


def test_rfe_reports_where_importances_came_from(synthetic: Dataset) -> None:
    pls = RFE(_pls(), n_features_to_select=10, step=8).fit(synthetic)
    assert pls.importance_method_ == "native"
    svr = StandardModel(Model.SVR, init_hyperparameters={"C": 4.0})
    rfe = RFE(svr, n_features_to_select=10, step=8, random_state=0).fit(synthetic)
    assert rfe.importance_method_ == "permutation"
    assert rfe.n_features_ == 10
    assert rfe.ranking_.min() == 1 and (rfe.ranking_ == 1).sum() == 10
    assert svr.best_model is None, "the passed estimator must not be fitted"
    assert rfe.predict(synthetic).shape[0] == synthetic.n_samples


def test_rfecv(synthetic: Dataset) -> None:
    rfecv = RFECV(_pls(2), step=6, cv=3, min_features_to_select=2).fit(synthetic)
    assert 2 <= rfecv.n_features_ <= synthetic.n_features
    assert "mean_test_score" in rfecv.cv_results_


def test_selectors_accept_arrays_and_multiple_outputs(synthetic2d: Dataset) -> None:
    X, Y = synthetic2d.X, synthetic2d.Y
    for selector in [SelectKBest(k=5), RFE(_pls(), n_features_to_select=5, step=8)]:
        assert selector.fit(X, Y).transform(X).shape == (synthetic2d.n_samples, 5)
    vip = VIPSelector(n_components=3).fit(X, Y)
    assert vip.transform(X).shape == (synthetic2d.n_samples, vip.get_support().sum())
    with pytest.raises(SpectraXAIValueError):
        SelectKBest().fit(X)
    with pytest.raises(SpectraXAIValueError):
        SelectKBest().fit(synthetic2d, Y)


def test_selector_in_a_pipeline(synthetic: Dataset) -> None:
    pipe = make_pipeline(
        SpectralPreprocessor([SpectralPreprocessing.ABS, SpectralPreprocessing.SNV]),
        SelectKBest(k=10),
        _pls(),
    )
    y = synthetic.Y.ravel()
    assert pipe.fit(synthetic.X, y).score(synthetic.X, y) > 0.5


def test_vip_selector(synthetic: Dataset) -> None:
    selector = VIPSelector(n_components=3).fit(synthetic)
    np.testing.assert_array_equal(selector.get_support(), selector.vip_ > 1)
    assert selector.model_.best_hyperparameters == {"n_components": 3}
    strict = VIPSelector(threshold=1e9, n_components=3).fit(synthetic)
    assert strict.get_support().sum() == 1, "always keep the best feature"
    tuned = VIPSelector(cv=3).fit(synthetic)
    assert tuned.model_.best_score is not None


def test_select_k_best_validates_k(synthetic: Dataset) -> None:
    for k in (0, synthetic.n_features + 1, "some"):
        with pytest.raises(SpectraXAIValueError):
            SelectKBest(k=k).fit(synthetic)
    assert SelectKBest(k="all").fit(synthetic).get_support().all()


def test_transform_checks_the_dataset(synthetic: Dataset) -> None:
    selector = SelectKBest(k=3).fit(synthetic)
    with pytest.raises(SpectraXAIValueError):
        selector.transform(synthetic.select_features(np.arange(10)))


# -- plot_selected_bands ---------------------------------------------------------------


@pytest.mark.parametrize("analysis", [PreHocAnalysis, PostHocAnalysis])
def test_plot_selected_bands(synthetic: Dataset, analysis: Any) -> None:
    selector = SelectKBest(k=6).fit(synthetic)
    fig, (other, ax) = plt.subplots(1, 2)
    result = analysis(synthetic).plot_selected_bands(selector, ax=ax, color="red")
    assert result is ax and not other.has_data()
    mask = selector.get_support()
    runs = 1 + int(np.sum(np.diff(np.flatnonzero(mask)) > 1))
    assert len(ax.patches) == runs
    labels = [p.get_label() for p in ax.patches]
    assert labels.count("Selected") == 1
    # every selected wavelength lies inside a shaded band
    wavelengths = cast(np.ndarray, synthetic.wavelengths)[mask]
    spans = [
        (p.get_x(), p.get_x() + p.get_width())
        for p in cast(list[Rectangle], ax.patches)
    ]
    assert all(any(lo <= w <= hi for lo, hi in spans) for w in wavelengths)


def test_plot_selected_bands_accepts_masks_and_indices(synthetic: Dataset) -> None:
    xai = PreHocAnalysis(synthetic)
    ax = xai.plot_selected_bands(np.array([0, 1, 2, 10]))
    assert len(ax.patches) == 2
    with pytest.raises(SpectraXAIValueError):
        xai.plot_selected_bands(np.ones(3, dtype=bool))


# -- cLHS split ------------------------------------------------------------------------


def test_clhs_split(synthetic: Dataset) -> None:
    idx_trn, idx_tst = synthetic.train_test_split(
        DatasetSplit.CLHS, 0.75, random_state=0
    )
    assert len(idx_trn) == 60 and len(idx_tst) == 20
    np.testing.assert_array_equal(
        np.sort(np.concatenate((idx_trn, idx_tst))), np.arange(synthetic.n_samples)
    )
    again, _ = synthetic.train_test_split(DatasetSplit.CLHS, 0.75, random_state=0)
    np.testing.assert_array_equal(idx_trn, again)
    for bad in (0.0, 1.0, 0.001):
        with pytest.raises(ValueError):
            synthetic.train_test_split(DatasetSplit.CLHS, bad)
    with pytest.raises(TypeError):
        synthetic.train_test_split(DatasetSplit.CLHS, 3)


def test_clhs_covers_the_strata_better_than_random_sampling() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(300, 3)) @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 1]])
    n = 30
    selected = _clhs(data, n, 3000, np.random.default_rng(1))
    assert len(np.unique(selected)) == n

    edges = np.quantile(data, np.linspace(0, 1, n + 1), axis=0)
    strata = np.stack(
        [
            np.clip(np.searchsorted(edges[1:-1, j], data[:, j], side="right"), 0, n - 1)
            for j in range(3)
        ],
        axis=1,
    )
    corr = np.asarray(np.corrcoef(data, rowvar=False))
    clhs = _clhs_objective(data, strata, corr, selected, n)
    random = np.mean(
        [
            _clhs_objective(data, strata, corr, rng.choice(300, n, replace=False), n)
            for _ in range(20)
        ]
    )
    assert clhs < 0.5 * random
