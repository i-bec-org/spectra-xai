# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3: scikit-learn interface, new pre-processing, unified results."""

import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import PostHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.spectra import Spectra, SpectralPreprocessing as SP

# -- SpectralPreprocessor ----------------------------------------------------------


@parametrize_with_checks(
    [
        SpectralPreprocessor(),
        SpectralPreprocessor(SP.SNV),
        SpectralPreprocessor([SP.MSC, SP.MEAN_CENTER]),
        SpectralPreprocessor(SP.DETREND),
    ]
)
def test_preprocessor_follows_sklearn_conventions(
    estimator: SpectralPreprocessor, check: Callable[[Any], None]
) -> None:
    check(estimator)


def test_preprocessor_matches_spectra_for_stateless_sequences(
    synthetic: Dataset,
) -> None:
    seq = [SP.ABS, (SP.SG1, {"window_length": 7, "polyorder": 2}), SP.SNV]
    expected = Spectra(synthetic.X).preprocess(seq).X
    np.testing.assert_allclose(
        SpectralPreprocessor(seq).fit_transform(synthetic.X), expected
    )
    # stateless sequences work without fit
    np.testing.assert_allclose(
        SpectralPreprocessor(seq).transform(synthetic.X), expected
    )


def test_preprocessor_learns_references_on_training_data(synthetic: Dataset) -> None:
    trn, tst = synthetic.X[:50], synthetic.X[50:]
    pre = SpectralPreprocessor([SP.ABS, SP.MSC]).fit(trn)
    reference = (-np.log10(trn)).mean(axis=0)
    np.testing.assert_allclose(cast(np.ndarray, pre.states_[1]), reference)
    np.testing.assert_allclose(
        pre.transform(tst), Spectra(-np.log10(tst)).msc(reference).X
    )
    with pytest.raises(Exception):
        SpectralPreprocessor(SP.MSC).transform(tst)  # stateful: must be fitted


# -- New pre-processing methods ------------------------------------------------------


def test_msc_recovers_reference_from_scaled_copies() -> None:
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.2, 0.8, 30)
    X = np.array([a + b * reference for a, b in [(0.1, 0.9), (-0.05, 1.2), (0, 1)]])
    np.testing.assert_allclose(Spectra(X).msc(reference).X, np.tile(reference, (3, 1)))


def test_detrend_and_mean_center() -> None:
    x = np.linspace(0, 1, 20)
    X = np.vstack([3 * x + 1, -x + 2])
    np.testing.assert_allclose(Spectra(X).detrend().X, 0, atol=1e-12)
    centered = Spectra(X).mean_center().X
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-12)


def test_stateful_steps_round_trip_through_strings() -> None:
    seq = [SP.ABS, SP.MSC, SP.DETREND, SP.MEAN_CENTER]
    assert SP.init_class(str(seq)) == seq


# -- Wavelengths and Dataset helpers -------------------------------------------------


def test_dataset_wavelengths(synthetic: Dataset) -> None:
    wvs = synthetic.wavelengths
    assert wvs is not None and wvs.dtype == float
    np.testing.assert_array_equal(wvs, np.asarray(synthetic.X_names, dtype=float))
    named = Dataset(synthetic.X, synthetic.Y)
    assert named.wavelengths is None
    explicit = Dataset(synthetic.X, synthetic.Y, wavelengths=wvs * 2)
    np.testing.assert_array_equal(explicit.wavelengths, wvs * 2)
    np.testing.assert_array_equal(explicit.subset(np.arange(3)).wavelengths, wvs * 2)
    np.testing.assert_array_equal(explicit.preprocess(SP.SNV).wavelengths, wvs * 2)
    with pytest.raises(SpectraXAIValueError):
        Dataset(synthetic.X, synthetic.Y, wavelengths=wvs[:-1])


def test_dataset_accepts_array_names(synthetic: Dataset) -> None:
    names = np.asarray(synthetic.X_names)
    assert Dataset(synthetic.X, synthetic.Y, X_names=names).n_features == names.size


def test_to_dataframe(synthetic2d: Dataset) -> None:
    df = synthetic2d.to_dataframe()
    assert isinstance(df, pandas.DataFrame)
    assert df.shape == (synthetic2d.n_samples, synthetic2d.n_features + 2)
    assert list(df.columns[-2:]) == synthetic2d.Y_names
    np.testing.assert_array_equal(df.iloc[:, :-2].to_numpy(), synthetic2d.X)


def test_continuum_removal_uses_uneven_wavelengths(synthetic: Dataset) -> None:
    absorbance = -np.log10(synthetic.X[:2])
    even = np.arange(synthetic.n_features, dtype=float) * 10
    uneven = np.cumsum(np.linspace(1, 20, synthetic.n_features))
    by_index = Spectra(absorbance).cr().X
    np.testing.assert_allclose(Spectra(absorbance, wavelengths=even).cr().X, by_index)
    assert not np.allclose(Spectra(absorbance, wavelengths=uneven).cr().X, by_index)


# -- StandardModel as an estimator ----------------------------------------------------


def _pls() -> StandardModel:
    return StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})


def test_clone_and_params_round_trip() -> None:
    for model in [
        _pls(),
        StandardModel(Model.SVR),
        StandardModel(Model.PLS, grid_search_hyperparameters={}),
        StandardModel(Model.CUSTOM, estimator=Ridge(), random_state=3, cv=4),
    ]:
        copy = cast(StandardModel, clone(model))
        assert copy.get_params().keys() == model.get_params().keys()
        assert copy.model == model.model and copy.cv == model.cv
    model = _pls().set_params(init_hyperparameters={"n_components": 2}, n_jobs=1)
    assert model.init_hyperparameters == {"n_components": 2} and model.n_jobs == 1


def test_default_grid_is_still_exposed() -> None:
    assert "C" in StandardModel(Model.SVR).grid_search_hyperparameters
    assert StandardModel(Model.PLS).grid_search_hyperparameters == {}


def test_fit_accepts_dataset_or_arrays(synthetic: Dataset) -> None:
    a = _pls().fit(synthetic)
    b = _pls().fit(synthetic.X, synthetic.Y.ravel())
    assert isinstance(a, StandardModel)
    np.testing.assert_allclose(a.predict(synthetic.X), b.predict(synthetic.X))
    np.testing.assert_allclose(a.predict(synthetic), a.predict(synthetic.X))
    with pytest.raises(SpectraXAIValueError):
        _pls().fit(synthetic.X)


def test_positional_cv_is_deprecated_but_works(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.PLS, grid_search_hyperparameters={"n_components": [1, 2]}
    )
    with pytest.warns(DeprecationWarning):
        model.fit(synthetic, 3)
    assert model.best_score is not None


def test_cross_val_score_and_pipelines(synthetic: Dataset) -> None:
    y = synthetic.Y.ravel()
    scores = cross_val_score(_pls(), synthetic.X, y, cv=3)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    pipe = make_pipeline(SpectralPreprocessor([SP.ABS, SP.MSC]), _pls())
    assert pipe.fit(synthetic.X, y).predict(synthetic.X).shape[0] == synthetic.n_samples
    search = GridSearchCV(
        pipe,
        {
            "standardmodel__init_hyperparameters": [
                {"n_components": 2},
                {"n_components": 4},
            ]
        },
        cv=3,
    )
    search.fit(synthetic.X, y)
    assert search.best_params_


def test_predict_rejects_wrong_number_of_features(synthetic: Dataset) -> None:
    model = _pls().fit(synthetic)
    with pytest.raises(SpectraXAIValueError):
        model.predict(synthetic.X[:, :5])


def test_svr_scaling_is_part_of_the_model(synthetic: Dataset) -> None:
    model = StandardModel(Model.SVR, init_hyperparameters={"C": 4.0}).fit(synthetic)
    assert model.pipeline_ is not None and "scaler" in model.pipeline_.named_steps
    np.testing.assert_allclose(
        model.predict(synthetic.X), model.pipeline_.predict(synthetic.X)
    )
    # the bare estimator expects scaled inputs, which is why tools need the model
    assert not np.allclose(
        model.best_model.predict(synthetic.X), model.predict(synthetic.X)
    )


def test_posthoc_accepts_standard_model(synthetic: Dataset) -> None:
    model = StandardModel(Model.SVR, init_hyperparameters={"C": 4.0}).fit(synthetic)
    ax = PostHocAnalysis(synthetic, random_state=0).bar_plot_permutation_importance(
        model
    )
    assert len(ax.patches) == synthetic.n_features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = PostHocAnalysis(synthetic).sage_importance(_pls().fit(synthetic))
    assert np.shape(explanation.values) == (synthetic.n_features,)


# -- Custom estimators -----------------------------------------------------------------


def test_custom_estimator(synthetic: Dataset) -> None:
    ridge = Ridge()
    model = StandardModel(
        Model.CUSTOM,
        grid_search_hyperparameters={"alpha": [0.01, 1.0]},
        estimator=ridge,
    )
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert not hasattr(ridge, "coef_"), "the passed estimator must not be fitted"
    assert res.loc[0, "model_name"] == "CUSTOM"
    assert res.loc[0, "importance_method"] == "coef"
    assert set(res.loc[0, "best_hyperparameters"]) == {"alpha"}
    with pytest.raises(SpectraXAIValueError):
        StandardModel(Model.CUSTOM)
    with pytest.raises(SpectraXAIValueError):
        StandardModel(Model.PLS, estimator=Ridge())


# -- Unified results ------------------------------------------------------------

SCHEMA = {
    "N", "RMSE", "R2", "RPIQ", "output", "pre_process", "val_score", "model_name",
    "best_hyperparameters", "feature_importance", "importance_method", "random_state",
    "training_time", "testing_time",
}  # fmt: skip


@pytest.mark.parametrize(
    "model, method, legacy",
    [
        (_pls(), "VIP", {"n_components"}),
        (StandardModel(Model.SVR, init_hyperparameters={"C": 1.0}), None, {"C", "SVs"}),
        (
            StandardModel(Model.RF, init_hyperparameters={"n_estimators": 10}),
            "impurity",
            {"max_features", "n_estimators"},
        ),
        (
            StandardModel(Model.XGBOOST, init_hyperparameters={"n_estimators": 10}),
            "gain",
            {"max_depth", "alpha"},
        ),
    ],
)
def test_results_schema(
    synthetic: Dataset, model: StandardModel, method: str | None, legacy: set[str]
) -> None:
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert SCHEMA | legacy <= set(res.columns)
    assert res.loc[0, "importance_method"] == method
    assert (res.loc[0, "feature_importance"] is None) == (method is None)


def test_permutation_importance_fills_missing_importance(synthetic2d: Dataset) -> None:
    model = StandardModel(Model.SVR, init_hyperparameters={"C": 1.0}, random_state=0)
    res = model.fit_and_predict(
        Dataset(synthetic2d.X, synthetic2d.Y[:, 0]),
        idx_trn=np.arange(60),
        return_permutation_importance=True,
    )
    assert res.loc[0, "importance_method"] == "permutation"
    assert len(res.loc[0, "feature_importance"]) == synthetic2d.n_features
    # models with their own importance keep it
    res = _pls().fit_and_predict(
        synthetic2d, idx_trn=np.arange(60), return_permutation_importance=True
    )
    assert list(res["importance_method"]) == ["VIP", "VIP"]


def test_stateless_results_match_preprocessing_everything_first(
    synthetic: Dataset,
) -> None:
    seq = [SP.ABS, SP.CR, (SP.SG1, {"window_length": 5, "polyorder": 2}), SP.SNV]
    trn, tst = np.arange(60), np.arange(60, synthetic.n_samples)
    res = _pls().fit_and_predict(synthetic, seq, idx_trn=trn)
    # the 1.1.0 route: pre-process the whole dataset, then split
    pre = synthetic.preprocess(seq)
    y_hat = _pls().fit(pre.subset(trn)).predict(pre.X[tst])
    assert res.loc[0, "RMSE"] == pytest.approx(
        np.sqrt(np.mean((pre.Y[tst, 0] - np.ravel(y_hat)) ** 2))
    )


def test_stateful_preprocessing_does_not_leak(synthetic: Dataset) -> None:
    trn, tst = np.arange(60), np.arange(60, synthetic.n_samples)
    res = _pls().fit_and_predict(synthetic, SP.MEAN_CENTER, idx_trn=trn)
    mean = synthetic.X[trn].mean(axis=0)
    y_hat = (
        _pls()
        .fit(synthetic.X[trn] - mean, synthetic.Y[trn])
        .predict(synthetic.X[tst] - mean)
    )
    assert res.loc[0, "RMSE"] == pytest.approx(
        np.sqrt(np.mean((synthetic.Y[tst, 0] - np.ravel(y_hat)) ** 2))
    )
