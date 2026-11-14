# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.models` includes classes and operations on ML models.

This module defines the class `StandardModel` which should be used to train your
models. It supports different ML algorithms and provides short-hand versions to
test multiple pre-treatment methods.

"""

import time
import warnings
from enum import Enum
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from cubist import Cubist
import xgboost as xgb

from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.utils import metrics, sigest
from spectraxai.spectra import SpectralPreprocessing, SpectralPreprocessingSequence
from spectraxai.dataset import Dataset
from spectraxai.preprocessing import SpectralPreprocessor


class Model(str, Enum):
    """A class to describe commonly used ML models for spectral processing."""

    PLS = "Partial Least Squares"
    SVR = "Support Vector Regression"
    RF = "Random Forest"
    CUBIST = "Cubist"
    XGBOOST = "XGBoost"
    CUSTOM = "Custom estimator"
    """Any scikit-learn compatible regressor, passed as ``estimator``."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value


_DEFAULT_GRIDS: dict[Model, dict[str, Any]] = {
    Model.SVR: {
        "epsilon": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5],
        "C": np.logspace(start=-2, stop=9, base=2, num=12),
    },
    Model.RF: {
        "max_features": [1.0, "sqrt", "log2"],
        "n_estimators": [50, 100, 150, 200],
    },
    Model.CUBIST: {
        "n_committees": [1, 5, 10, 20],
        "neighbors": [1, 5, 9],
    },
    Model.XGBOOST: {
        "max_depth": [3, 5, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [10, 50, 100],
        "colsample_bytree": [0.3, 0.6],
        "alpha": [0, 10],
    },
}
"""Default grid searched per model (PLS n_components and SVR gamma depend on the
dataset and are added at fit time)."""


def _is_default_grid(model: Model, grid: dict[str, Any]) -> bool:
    """Whether ``grid`` equals the default grid of ``model`` (e.g. after a clone)."""
    default = _DEFAULT_GRIDS.get(model)
    if default is None or set(grid) != set(default):
        return False
    return all(
        np.array_equal(np.asarray(grid[k], dtype=object), np.asarray(v, dtype=object))
        for k, v in default.items()
    )


class StandardModel(RegressorMixin, BaseEstimator):
    """Class with standard models for ML to apply on spectral datasets.

    `StandardModel` is a scikit-learn compatible regressor: besides
    ``fit(dataset)`` it accepts ``fit(X, y)``, and it can be used with
    `sklearn.base.clone`, pipelines, `sklearn.model_selection.cross_val_score`,
    `sklearn.inspection.permutation_importance`, SAGE, etc. For SVR the input
    scaling is part of the model, so these tools see the right inputs.

    """

    model: Model
    """The type of `Model` used."""

    init_hyperparameters: dict[str, Any]
    """A dictionary of the hyperparameters of the models identified by an expert, to
    override a grid search."""

    grid_search_hyperparameters: dict[str, Any]
    """A dictionary containing as keys the hyperparameters of the model and as values a
    list of the potential candidate values."""

    estimator: Any
    """The scikit-learn regressor to use with `Model.CUSTOM` (never modified; a
    clone is fitted)."""

    cv: int | Iterable[tuple[np.ndarray, np.ndarray]]
    """The cross-validation strategy of the grid search."""

    n_jobs: int | None
    """Number of parallel jobs of the grid search (-1 uses all processors)."""

    best_hyperparameters: dict[str, Any] | None
    """A dictionary of the best hyperparameters, either set externally or as identified
    after calling the train function."""

    training_time: float | None
    """Training time in seconds."""

    testing_time: float | None
    """Time for the prediction in seconds."""

    best_model: Any
    """The best optimized model after tuning the hyperparameters (the bare estimator,
    without the SVR input scaling; see `pipeline_`)."""

    best_score: float | None
    """The best score (R2) in the internal validation set corresponding to the best
    model. Only available after a grid search."""

    scaler: StandardScaler | None
    """The scaler fitted on the training inputs (used by SVR only)."""

    random_state: int | None
    """Seed passed to stochastic models (RF, XGBoost, Cubist) and to the SVR gamma
    estimation. None keeps the behaviour of earlier versions."""

    param_grid_: dict[str, Any] | None
    """The grid actually searched by the last call to `fit`, including any
    dataset-dependent entries (SVR gamma, PLS n_components)."""

    pipeline_: Pipeline | None
    """The fitted prediction pipeline: the SVR scaler (if any) followed by
    `best_model`. Used by `predict`."""

    def __init__(
        self,
        model: Model,
        init_hyperparameters: dict[str, Any] | None = None,
        grid_search_hyperparameters: dict[str, Any] | None = None,
        *,
        estimator: Any = None,
        cv: int | Iterable[tuple[np.ndarray, np.ndarray]] = 5,
        random_state: int | None = None,
        n_jobs: int | None = -1,
    ) -> None:
        """Initialize StandardModel class for a `Model` and its hyperparameters.

        You need to pass either a set of hyperparameters for the model, or a range
        thereof in which to search for the optimal set.

        Parameters
        ----------
        model: `Model`
            Select a model from `Model` class. Use `Model.CUSTOM` together with
            ``estimator`` to bring your own regressor.

        init_hyperparameters: `dict`, optional
            A dictionary of pre-selected hyperparameters (e.g. a best model)

        grid_search_hyperparameters: `dict`, optional
            Specify custom grid search range for the hyperparameters

        estimator: scikit-learn regressor, optional
            The regressor to use with `Model.CUSTOM`, e.g.
            ``sklearn.linear_model.Ridge()``. It is cloned, never modified.

        cv: int, or an iterable, default=5
            The cross-validation strategy of the grid search (see `fit`).

        random_state: `int`, optional
            Seed for stochastic models (RF, XGBoost, Cubist) and for the SVR gamma
            estimation, making results reproducible. Defaults to None, which keeps
            the behaviour of earlier versions (XGBoost then uses seed 2000).

        n_jobs: `int`, optional
            Number of parallel jobs of the grid search. Defaults to -1 (all
            processors).

        """
        if (
            init_hyperparameters
            and grid_search_hyperparameters
            and not _is_default_grid(model, grid_search_hyperparameters)
        ):
            raise SpectraXAIValueError(
                "You cannot specify both init_hyperparameters and "
                "grid_search_hyperparameters"
            )
        if (model == Model.CUSTOM) != (estimator is not None):
            raise SpectraXAIValueError(
                "Pass an estimator if and only if model is Model.CUSTOM"
            )
        self.model = model
        # Parameters are stored as passed (when given) so that sklearn's clone and
        # get_params/set_params work.
        self.init_hyperparameters = (
            {} if init_hyperparameters is None else init_hyperparameters
        )
        if grid_search_hyperparameters or model not in _DEFAULT_GRIDS:
            self.grid_search_hyperparameters = (
                {}
                if grid_search_hyperparameters is None
                else grid_search_hyperparameters
            )
        else:
            self.grid_search_hyperparameters = dict(_DEFAULT_GRIDS[model])
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.training_time = None
        self.testing_time = None
        self.best_model = None
        self.best_hyperparameters = None
        self.best_score = None
        self.scaler = None
        self.param_grid_ = None
        self.pipeline_ = None

    def __sklearn_is_fitted__(self) -> bool:
        return self.pipeline_ is not None

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = self.model not in (Model.SVR, Model.CUBIST)
        tags.target_tags.single_output = True
        return tags

    def _new_estimator(self, grid_search: bool) -> Any:
        """Create an unfitted estimator for `model`, seeded by `random_state`.

        Without a `random_state`, XGBoost keeps the historical seed of 2000 in grid
        search and its library default otherwise.

        """
        seeded: dict[str, Any] = (
            {} if self.random_state is None else {"random_state": self.random_state}
        )
        if self.model == Model.SVR:
            return SVR(kernel="rbf", max_iter=int(5e8))
        elif self.model == Model.PLS:
            return PLSRegression()
        elif self.model == Model.RF:
            return RandomForestRegressor(**seeded)
        elif self.model == Model.CUBIST:
            return Cubist(**seeded)
        elif self.model == Model.XGBOOST:
            if self.random_state is not None:
                return xgb.XGBRegressor(random_state=self.random_state)
            return xgb.XGBRegressor(seed=2000) if grid_search else xgb.XGBRegressor()
        elif self.model == Model.CUSTOM:
            return clone(self.estimator)
        raise SpectraXAIValueError(f"Unsupported model: {self.model!r}")

    _REPORTED_HYPERPARAMETERS: dict[Model, list[str]] = {
        Model.SVR: ["C", "epsilon", "gamma"],
        Model.PLS: ["n_components"],
        Model.RF: ["max_features", "n_estimators"],
        Model.CUBIST: ["n_committees", "neighbors"],
        Model.XGBOOST: [
            "max_depth",
            "learning_rate",
            "n_estimators",
            "colsample_bytree",
            "alpha",
        ],
        Model.CUSTOM: [],
    }
    """Hyperparameters reported as columns by `fit_and_predict`, per model."""

    def _hyperparameter(self, key: str) -> Any:
        """Return a fitted hyperparameter, falling back to the estimator's value."""
        if self.best_hyperparameters is not None and key in self.best_hyperparameters:
            return self.best_hyperparameters[key]
        return self.best_model.get_params().get(key)

    def __vip(self, model: PLSRegression) -> np.ndarray:
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array(
                [(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)]
            )
            vips[i] = np.sqrt(p * (s.ravel() @ weight) / total_s)
        return vips

    def feature_importance(self) -> tuple[np.ndarray | None, str | None]:
        """Return the fitted model's own feature importance and how it was computed.

        Returns
        -------
        `tuple`
            ``(importance, method)``: VIP scores for PLS (``"VIP"``),
            ``feature_importances_`` for RF (``"impurity"``), Cubist (``"usage"``),
            XGBoost (its ``importance_type``, ``"gain"`` by default) and custom
            estimators (``"native"``), absolute coefficients for linear custom
            estimators (``"coef"``), or ``(None, None)`` when the model has none
            (e.g. SVR; see ``return_permutation_importance`` in
            `fit_and_predict`).

        """
        if self.best_model is None:
            raise RuntimeError("You need to train the model first")
        model = self.best_model
        if self.model == Model.PLS:
            return self.__vip(model), "VIP"
        if self.model == Model.SVR:
            return None, None
        if hasattr(model, "feature_importances_"):
            method = {
                Model.RF: "impurity",
                Model.CUBIST: "usage",
                Model.XGBOOST: getattr(model, "importance_type", None) or "gain",
            }.get(self.model, "native")
            return np.asarray(model.feature_importances_), method
        coef = getattr(model, "coef_", None)
        if coef is not None:
            coef = np.abs(np.asarray(coef))
            if coef.ndim == 2:
                coef = coef.mean(axis=0 if coef.shape[1] == self.n_features_in_ else 1)
            if coef.size == self.n_features_in_:
                return coef.ravel(), "coef"
        return None, None

    @staticmethod
    def _as_dataset(X: Any, y: Any) -> Dataset:
        if isinstance(X, Dataset):
            return X
        if y is None:
            raise SpectraXAIValueError("y is required when X is not a Dataset")
        return Dataset(np.asarray(X), np.asarray(y))

    def fit(
        self,
        dataset: Dataset | Any,
        y: Any = None,
        cv: int | Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "StandardModel":
        """Trains the model on a given dataset (or on ``X, y``).

        If you didn't supply the `init_hyperparameters` option to the constructor, then
        a grid search optimization process takes place as follows:
        Using the sklearn.model_selection.GridSearchCV approach, a grid search using a
        cross-validation splitting strategy specified by cv is performed. After the
        optimal hyperparameters are defined, the model is then retrained on the whole
        dataset.

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset` or array-like
            the Dataset to train the model, or the input matrix ``X`` (then pass
            ``y`` too), as scikit-learn does

        y: array-like, optional
            The output(s), when ``dataset`` is an input matrix

        cv: int, or an iterable, optional
            Determines the cross-validation splitting strategy. Defaults to the
            ``cv`` passed to the constructor (5).
            Possible inputs for cv are:
            * integer, to specify the number of folds
            * An iterable yielding (train, test) splits as arrays of indices

        Returns
        -------
        `StandardModel`
            The fitted model (``self``).

        """
        if isinstance(dataset, Dataset) and y is not None:
            # Earlier versions had the signature fit(dataset, cv)
            if cv is not None:
                raise SpectraXAIValueError("y should not be given with a Dataset")
            warnings.warn(
                "Passing cv positionally to StandardModel.fit is deprecated; "
                "use fit(dataset, cv=...)",
                DeprecationWarning,
                stacklevel=2,
            )
            cv, y = y, None
        dataset = self._as_dataset(dataset, y)
        cv = self.cv if cv is None else cv
        if self.model in [Model.SVR, Model.CUBIST] and dataset.n_outputs > 1:
            raise SpectraXAIValueError(
                "Cannot create a multi-output model. "
                "Please train a model for each output property."
            )
        self.n_features_in_ = dataset.n_features

        # Scale the data for SVR
        if self.model == Model.SVR:
            self.scaler = StandardScaler()
            self.scaler.fit(dataset.X)
            X_train = np.asarray(self.scaler.transform(dataset.X))
        else:
            self.scaler = None
            X_train = dataset.X

        # If hyperparameters are passed, use them; otherwise run grid search
        if len(self.init_hyperparameters) != 0:
            self.best_model = self._new_estimator(grid_search=False).set_params(
                **self.init_hyperparameters
            )
            self.param_grid_ = None
            self.best_score = None
            trn_t0 = time.time()
            self.best_model.fit(
                X_train,
                np.squeeze(dataset.Y, axis=1) if dataset.n_outputs == 1 else dataset.Y,
            )
            self.best_hyperparameters = dict(self.init_hyperparameters)
            trn_t1 = time.time()
        else:
            # Work on a copy so that dataset-dependent entries computed here do not
            # leak into later calls (e.g. other folds or pre-processings).
            param_grid = dict(self.grid_search_hyperparameters)
            if self.model == Model.SVR and "gamma" not in param_grid:
                gamma = sigest(X_train, random_state=self.random_state)
                param_grid["gamma"] = np.linspace(gamma[0], gamma[2], num=5)
            elif self.model == Model.PLS and "n_components" not in param_grid:
                param_grid["n_components"] = np.arange(1, min(100, X_train.shape[1]), 1)
            self.param_grid_ = param_grid
            trn_t0 = time.time()
            grid_search = GridSearchCV(
                estimator=self._new_estimator(grid_search=True),
                param_grid=param_grid,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=0,
            )
            grid_search.fit(
                X_train,
                np.squeeze(dataset.Y, axis=1) if dataset.n_outputs == 1 else dataset.Y,
            )
            trn_t1 = time.time()
            self.best_model = grid_search.best_estimator_
            self.best_hyperparameters = grid_search.best_params_
            self.best_score = grid_search.best_score_
        self.training_time = trn_t1 - trn_t0
        steps: list[tuple[str, Any]] = [("model", self.best_model)]
        if self.scaler is not None:
            steps.insert(0, ("scaler", self.scaler))
        self.pipeline_ = Pipeline(steps)
        return self

    def predict(self, X_test: np.ndarray | Dataset | Any) -> np.ndarray:
        """Predict using the best_model from a new unknown input.

        Parameters
        ----------
        X_test: `np.ndarray` or `spectraxai.dataset.Dataset`
            The new input data to predict their output

        Returns
        -------
        `np.ndarray`
            A np.ndarray of size (n_test_samples, n_outputs) with the predictions

        """
        if self.pipeline_ is None:
            raise RuntimeError("You need to train the model first")
        X = X_test.X if isinstance(X_test, Dataset) else np.asarray(X_test)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise SpectraXAIValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.n_features_in_}"
            )
        tst_t0 = time.time()
        y_hat = self.pipeline_.predict(X)
        tst_t1 = time.time()
        self.testing_time = tst_t1 - tst_t0
        return y_hat

    def _permutation_importance(
        self, X: np.ndarray, Y: np.ndarray, output: int
    ) -> np.ndarray:
        """Mean decrease in R2 of output ``output`` when permuting each feature."""
        if self.pipeline_ is None:
            raise RuntimeError("You need to train the model first")

        def score(estimator: Any, X_: np.ndarray, y_: np.ndarray) -> float:
            y_hat = estimator.predict(X_)
            y_hat = y_hat[:, output] if y_hat.ndim == 2 else y_hat
            return float(r2_score(y_, y_hat))

        result = cast(
            Bunch,
            permutation_importance(
                self.pipeline_,
                X,
                Y[:, output],
                scoring=score,
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
        )
        return np.asarray(result.importances_mean)

    def fit_and_predict(
        self,
        dataset: Dataset,
        preprocess: SpectralPreprocessingSequence = SpectralPreprocessing.NONE,
        idx_trn: np.ndarray | None = None,
        idx_tst: np.ndarray | None = None,
        get_model: bool = False,
        *,
        return_permutation_importance: bool = False,
    ) -> pandas.DataFrame:
        """Train and test a model in given dataset and spectral pre-processing sequence.

        Pass here the whole dataset of (X, Y) and either specify
        idx_trn (training indices) or idx_tst (testing indices).

        The pre-processing is fitted on the training samples only (this matters for
        stateful steps such as `SpectralPreprocessing.MSC`).

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        preprocess: `spectraxai.spectra.SpectralPreprocessingSequence`
            Optional pre-processing sequence. Defaults to SpectralPreprocessing.NONE.

        idx_trn: `np.ndarray`
            The indices of the trn samples. Defaults to np.array([]).
            This can either be a one dimensional array (for a single split), or
            two-dimensional to indicate a cross-validation split.

        idx_tst: `np.ndarray`
            The indices of the tst samples. Defaults to np.array([]).
            This can either be a one dimensional array (for a single split), or
            two-dimensional to indicate a cross-validation split.

        get_model: `bool`
            If true, also return the generated model (column ``model``). Defaults
            to False.

        return_permutation_importance: `bool`
            For models without their own feature importance (e.g. SVR), fill
            ``feature_importance`` with the permutation importance on the test set
            (mean decrease in R2). Defaults to False.

        Returns
        -------
        `pandas.DataFrame`
            The accuracy results and assorted metadata for each output property.
            Every row has ``N``, ``RMSE``, ``R2``, ``RPIQ``, ``output``,
            ``pre_process``, ``val_score``, ``model_name``, ``best_hyperparameters``
            (dict), ``feature_importance`` (array or None), ``importance_method``,
            ``random_state``, ``training_time`` and ``testing_time``, plus the
            model's main hyperparameters as separate columns (e.g. ``C``,
            ``n_components``) and, for SVR, ``SVs``.

        """
        idx_trn = np.array([]) if idx_trn is None else idx_trn
        idx_tst = np.array([]) if idx_tst is None else idx_tst
        if idx_tst.size == 0 and idx_trn.size == 0:
            raise SpectraXAIValueError("You need to specify either tst or trn indices")
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise SpectraXAIValueError("You cannot specify both trn and tst")
        if idx_trn.size > 0 and isinstance(idx_trn[0], np.ndarray):  # Multiple-folds
            results = []
            for fold in range(idx_trn.shape[0]):
                result = self.fit_and_predict(
                    dataset,
                    preprocess,
                    idx_trn=idx_trn[fold],
                    get_model=get_model,
                    return_permutation_importance=return_permutation_importance,
                )
                result["fold"] = fold + 1
                results.append(result)
            return pandas.concat(results, ignore_index=True)
        elif idx_tst.size > 0 and isinstance(idx_tst[0], np.ndarray):  # Multiple-folds
            results = []
            for fold in range(idx_tst.shape[0]):
                result = self.fit_and_predict(
                    dataset,
                    preprocess,
                    idx_tst=idx_tst[fold],
                    get_model=get_model,
                    return_permutation_importance=return_permutation_importance,
                )
                result["fold"] = fold + 1
                results.append(result)
            return pandas.concat(results, ignore_index=True)
        elif idx_trn.size > 0 and idx_tst.size == 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(trn=idx_trn)
        elif idx_trn.size == 0 and idx_tst.size > 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(tst=idx_tst)

        preprocessor = SpectralPreprocessor(preprocess, wavelengths=dataset.wavelengths)
        X_trn = preprocessor.fit_transform(dataset.X[idx_trn])
        X_tst = preprocessor.transform(dataset.X[idx_tst])
        Y_trn, Y_tst = dataset.Y[idx_trn], dataset.Y[idx_tst]
        self.fit(Dataset(X_trn, Y_trn))
        if self.best_model is None or self.best_hyperparameters is None:
            raise RuntimeError("The model failed to fit")
        y_hat = self.predict(X_tst)
        importance, importance_method = self.feature_importance()

        results = []
        for i in range(dataset.n_outputs):
            res = metrics(Y_tst[:, i], y_hat[:, i] if y_hat.ndim == 2 else y_hat)
            res["output"] = dataset.Y_names[i]
            # Gather all results
            res["pre_process"] = str(preprocess)
            res["val_score"] = self.best_score
            res["model_name"] = str(self.model)
            res["best_hyperparameters"] = dict(self.best_hyperparameters)
            for key in self._REPORTED_HYPERPARAMETERS[self.model]:
                res[key] = self._hyperparameter(key)
            if self.model == Model.SVR:
                res["SVs"] = len(self.best_model.support_)
            if importance is None and return_permutation_importance:
                res["feature_importance"] = self._permutation_importance(
                    X_tst, Y_tst, i
                )
                res["importance_method"] = "permutation"
            else:
                res["feature_importance"] = importance
                res["importance_method"] = importance_method
            res["random_state"] = self.random_state
            res["training_time"] = self.training_time
            res["testing_time"] = self.testing_time
            if get_model:
                res["model"] = self.best_model
            results.append(res)
        return pandas.DataFrame(results)

    def fit_and_predict_multiple(
        self,
        dataset: Dataset,
        preprocesses: Sequence[SpectralPreprocessingSequence] | None = None,
        idx_trn: np.ndarray | None = None,
        idx_tst: np.ndarray | None = None,
        *,
        return_permutation_importance: bool = False,
    ) -> pandas.DataFrame:
        """Train a model using different pre-treatments and predict on the test set.

        A short-hand version to quickly test different pre-treatments methods,
        calling the `fit_and_predict` function.

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        preprocesses: `spectraxai.spectra.List[SpectralPreprocessingSequence]`
            List of different pre-processing sequences to test. Defaults to [].

        idx_trn: `np.ndarray`
            The indices of the trn samples. Defaults to np.array([]).

        idx_tst: `np.ndarray`
            The indices of the tst samples. Defaults to np.array([]).

        return_permutation_importance: `bool`
            See `fit_and_predict`. Defaults to False.

        Returns
        -------
        `pandas.DataFrame`
            Returns a dataframe with the results of the trained models.
            By default, no model is returned to keep a low memory footprint.

        """
        preprocesses = [] if preprocesses is None else preprocesses
        idx_trn = np.array([]) if idx_trn is None else idx_trn
        idx_tst = np.array([]) if idx_tst is None else idx_tst
        if len(preprocesses) == 0:
            raise SpectraXAIValueError(
                "The list of SpectralPreprocessingSequence may not be empty"
            )
        results = [
            self.fit_and_predict(
                dataset,
                preprocess,
                idx_trn,
                idx_tst,
                return_permutation_importance=return_permutation_importance,
            )
            for preprocess in preprocesses
        ]
        return pandas.concat(results, ignore_index=True)
