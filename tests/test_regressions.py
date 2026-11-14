# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the bugs fixed in 1.2.0 (fast, synthetic data)."""

import warnings

import numpy as np
import pytest
from hypothesis import given, strategies as st
from sklearn.model_selection import StratifiedKFold

from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.utils import continuum_removal, estimate_epsilon, sigest

# -- Exceptions --------------------------------------------------------------


def test_exception_is_both_value_and_assertion_error(synthetic: Dataset) -> None:
    for exc in (ValueError, AssertionError, SpectraXAIValueError):
        with pytest.raises(exc):
            synthetic.train_test_split_explicit()


# -- Grid search must not leak dataset-dependent state between fits ------------


def test_pls_grid_is_recomputed_per_fit(synthetic: Dataset) -> None:
    model = StandardModel(Model.PLS)
    model.fit(synthetic, cv=3)
    assert model.grid_search_hyperparameters == {}
    assert model.param_grid_ is not None
    assert max(model.param_grid_["n_components"]) == synthetic.n_features - 1

    narrow = Dataset(synthetic.X[:, :8], synthetic.Y)
    model.fit(narrow, cv=3)  # used to fail: n_components up to 41 > 8 features
    assert max(model.param_grid_["n_components"]) == 7
    assert model.grid_search_hyperparameters == {}


def test_svr_gamma_is_recomputed_per_fit(synthetic: Dataset) -> None:
    model = StandardModel(Model.SVR, random_state=0)
    model.fit(synthetic, cv=3)
    assert model.param_grid_ is not None
    gamma_full = np.asarray(model.param_grid_["gamma"])
    assert "gamma" not in model.grid_search_hyperparameters

    model.fit(Dataset(synthetic.X[:, :5], synthetic.Y), cv=3)
    assert not np.allclose(gamma_full, model.param_grid_["gamma"])


def test_user_grid_is_not_mutated(synthetic: Dataset) -> None:
    grid = {"n_components": [1, 2]}
    model = StandardModel(Model.PLS, grid_search_hyperparameters=grid)
    model.fit(synthetic, cv=3)
    assert grid == {"n_components": [1, 2]}


# -- Bounds checks -------------------------------------------------------------


def test_index_equal_to_n_samples_is_out_of_bounds(synthetic: Dataset) -> None:
    bad = np.array([0, synthetic.n_samples])
    with pytest.raises(ValueError):
        synthetic.subset(bad)
    with pytest.raises(SpectraXAIValueError):
        synthetic.train_test_split_explicit(trn=bad)
    with pytest.raises(SpectraXAIValueError):
        synthetic.train_test_split_explicit(tst=bad)
    last = np.array([synthetic.n_samples - 1])
    assert synthetic.subset(last).n_samples == 1


# -- Reproducibility -----------------------------------------------------------


@pytest.mark.parametrize(
    "split, opt", [(DatasetSplit.RANDOM, 0.7), (DatasetSplit.CROSS_VALIDATION, 4)]
)
def test_splits_are_reproducible(
    synthetic: Dataset, split: DatasetSplit, opt: float
) -> None:
    a = synthetic.train_test_split(split, opt, random_state=42)
    b = synthetic.train_test_split(split, opt, random_state=42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("model", [Model.RF, Model.XGBOOST, Model.SVR])
def test_models_are_reproducible(synthetic: Dataset, model: Model) -> None:
    grids = {
        Model.RF: {"n_estimators": [10], "max_features": ["sqrt"]},
        Model.XGBOOST: {"n_estimators": [10], "colsample_bytree": [0.5]},
        Model.SVR: {"C": [1.0], "epsilon": [0.1]},
    }
    preds = []
    for _ in range(2):
        m = StandardModel(
            model, grid_search_hyperparameters=dict(grids[model]), random_state=7
        )
        m.fit(synthetic, cv=3)
        preds.append(m.predict(synthetic.X))
    np.testing.assert_allclose(preds[0], preds[1])


def test_svr_helpers_accept_random_state(synthetic: Dataset) -> None:
    np.testing.assert_array_equal(
        sigest(synthetic.X, random_state=1), sigest(synthetic.X, random_state=1)
    )
    y = synthetic.Y.ravel()
    assert estimate_epsilon(
        synthetic.X, y, ndegree=1, random_state=3
    ) == estimate_epsilon(synthetic.X, y, ndegree=1, random_state=3)


def test_mutual_information_is_reproducible(synthetic: Dataset) -> None:
    a = PreHocAnalysis(synthetic, random_state=0).feature_importance(FeatureRanking.MI)
    b = PreHocAnalysis(synthetic, random_state=0).feature_importance(FeatureRanking.MI)
    np.testing.assert_array_equal(a, b)


# -- Pre-hoc normalisation must not divide by zero ------------------------------


@pytest.mark.parametrize("method", [FeatureRanking.MI, FeatureRanking.F_STATISTIC])
def test_ranking_handles_uninformative_output(method: FeatureRanking) -> None:
    rng = np.random.default_rng(0)
    dataset = Dataset(rng.normal(size=(50, 6)), np.zeros(50))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns about the constant target
        importance = PreHocAnalysis(dataset, random_state=0).feature_importance(method)
    assert np.isfinite(importance).all()


# -- Post-hoc labels -------------------------------------------------------------


def test_permutation_importance_label(synthetic: Dataset) -> None:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    model.fit(synthetic)
    ax = PostHocAnalysis(synthetic, random_state=0).bar_plot_permutation_importance(
        model.best_model
    )
    assert ax.get_ylabel() == "Mean decrease in score"


# -- fit_and_predict result columns ---------------------------------------------


def test_xgboost_reports_hyperparameters_and_importance(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.XGBOOST, init_hyperparameters={"n_estimators": 10}, random_state=0
    )
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    for column in ["max_depth", "learning_rate", "n_estimators", "feature_importance"]:
        assert column in res.columns
    assert res.loc[0, "n_estimators"] == 10
    assert len(res.loc[0, "feature_importance"]) == synthetic.n_features


def test_partial_init_hyperparameters_do_not_raise(synthetic: Dataset) -> None:
    # Used to raise KeyError because gamma was not in init_hyperparameters
    model = StandardModel(Model.SVR, init_hyperparameters={"C": 1.0})
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert res.loc[0, "C"] == 1.0
    assert res.loc[0, "gamma"] == "scale"


def test_get_model_is_propagated_to_folds(synthetic: Dataset) -> None:
    idx_trn, _ = synthetic.train_test_split(
        DatasetSplit.CROSS_VALIDATION, 3, random_state=0
    )
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 2})
    res = model.fit_and_predict(synthetic, idx_trn=idx_trn, get_model=True)
    assert "model" in res.columns
    assert bool(res["model"].notna().all())
    assert sorted(res["fold"]) == [1, 2, 3]


# -- Stratified split -------------------------------------------------------------


def test_stratified_split_on_continuous_output(synthetic: Dataset) -> None:
    idx_trn, idx_tst = synthetic.train_test_split(DatasetSplit.STRATIFIED, 4)
    assert len(idx_tst) == 4
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(idx_tst))), np.arange(synthetic.n_samples)
    )
    # Each fold should span the range of the output
    y = synthetic.Y.ravel()
    for fold in idx_tst:
        assert y[fold].min() < np.quantile(y, 0.2)
        assert y[fold].max() > np.quantile(y, 0.8)


def test_stratified_split_on_class_labels_is_unchanged() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), np.repeat([0, 1, 2], 20)
    _, idx_tst = Dataset(X, y).train_test_split(DatasetSplit.STRATIFIED, 5)
    expected = [tst for _, tst in StratifiedKFold(5).split(X, y)]
    for got, exp in zip(idx_tst, expected):
        np.testing.assert_array_equal(got, exp)


def test_stratified_split_on_multi_output(synthetic2d: Dataset) -> None:
    _, idx_tst = synthetic2d.train_test_split(
        DatasetSplit.STRATIFIED, 3, random_state=0
    )
    assert len(idx_tst) == 3


# -- SpectralPreprocessing.init_class round-trips -----------------------------------

_options = st.fixed_dictionaries(
    {"window_length": st.sampled_from([5, 7, 11]), "polyorder": st.integers(1, 3)}
)
_single = st.sampled_from(list(SpectralPreprocessing))
_SG = [SpectralPreprocessing.SG0, SpectralPreprocessing.SG1, SpectralPreprocessing.SG2]
_with_options = st.tuples(
    st.sampled_from(_SG),
    _options,
)
_step = st.one_of(_single, _with_options)
_sequence = st.one_of(
    _step,
    st.lists(st.one_of(_step, st.lists(_step, min_size=1, max_size=2)), max_size=4),
)


@given(_sequence)
def test_init_class_round_trips(sequence: object) -> None:
    assert SpectralPreprocessing.init_class(str(sequence)) == sequence


def test_init_class_keeps_string_options() -> None:
    seq = [(SpectralPreprocessing.SG1, {"window_length": 7, "mode": "nearest"})]
    assert SpectralPreprocessing.init_class(str(seq)) == seq


def test_init_class_rejects_unknown_tokens() -> None:
    with pytest.raises(SpectraXAIValueError):
        SpectralPreprocessing.init_class("[foo, SNV]")


# -- Continuum removal -------------------------------------------------------------


def test_continuum_removal_uses_wavelengths_for_matrices(synthetic: Dataset) -> None:
    wvs = [float(w) for w in synthetic.X_names]
    absorbance = -np.log10(synthetic.X[:3])
    matrix = continuum_removal(absorbance, wvs)
    rows = np.array([continuum_removal(row, wvs) for row in absorbance])
    np.testing.assert_allclose(matrix, rows)


def test_preprocess_rejects_unknown_method(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError):
        synthetic.preprocess("SNV")  # type: ignore[arg-type]
