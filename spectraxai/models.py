# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.models` includes classes and operations on ML models.

This module defines the class `StandardModel` which should be used to train your
models. It supports different ML algorithms and provides short-hand versions to
test multiple pre-treatment methods.

"""

import time
from enum import Enum
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from cubist import Cubist
import xgboost as xgb

from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.utils import metrics, sigest
from spectraxai.spectra import SpectralPreprocessing, SpectralPreprocessingSequence
from spectraxai.dataset import Dataset


class Model(str, Enum):
    """A class to describe commonly used ML models for spectral processing."""

    PLS = "Partial Least Squares"
    SVR = "Support Vector Regression"
    RF = "Random Forest"
    CUBIST = "Cubist"
    XGBOOST = "XGBoost"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value


class StandardModel:
    """Class with standard models for ML to apply on spectral datasets."""

    model: Model
    """The type of `Model` used."""

    init_hyperparameters: dict[str, Any]
    """A dictionary of the hyperparameters of the models identified by an expert, to
    override a grid search."""

    grid_search_hyperparameters: dict[str, Any]
    """A dictionary containing as keys the hyperparameters of the model and as values a
    list of the potential candidate values."""

    best_hyperparameters: dict[str, Any] | None
    """A dictionary of the best hyperparameters, either set externally or as identified
    after calling the train function."""

    training_time: float | None
    """Training time in seconds."""

    testing_time: float | None
    """Time for the prediction in seconds."""

    best_model: Any
    """The best optimized model after tuning the hyperparameters."""

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

    def __init__(
        self,
        model: Model,
        init_hyperparameters: dict[str, Any] | None = None,
        grid_search_hyperparameters: dict[str, Any] | None = None,
        *,
        random_state: int | None = None,
    ) -> None:
        """Initialize StandardModel class for a `Model` and its hyperparameters.

        You need to pass either a set of hyperparameters for the model, or a range
        thereof in which to search for the optimal set.

        Parameters
        ----------
        model: `Model`
            Select a model from `Model` class.

        init_hyperparameters: `dict`, optional
            A dictionary of pre-selected hyperparameters (e.g. a best model)

        grid_search_hyperparameters: `dict`, optional
            Specify custom grid search range for the hyperparameters

        random_state: `int`, optional
            Seed for stochastic models (RF, XGBoost, Cubist) and for the SVR gamma
            estimation, making results reproducible. Defaults to None, which keeps
            the behaviour of earlier versions (XGBoost then uses seed 2000).

        """
        init_hyperparameters = (
            {} if init_hyperparameters is None else init_hyperparameters
        )
        grid_search_hyperparameters = (
            {} if grid_search_hyperparameters is None else grid_search_hyperparameters
        )
        if init_hyperparameters and grid_search_hyperparameters:
            raise SpectraXAIValueError(
                "You cannot specify both init_hyperparameters and "
                "grid_search_hyperparameters"
            )
        self.model = model
        self.init_hyperparameters = init_hyperparameters
        if len(grid_search_hyperparameters) != 0:
            self.grid_search_hyperparameters = grid_search_hyperparameters
        else:
            if model == Model.SVR:
                self.grid_search_hyperparameters = {
                    "epsilon": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5],
                    "C": np.logspace(start=-2, stop=9, base=2, num=12),
                }
            elif model == Model.RF:
                self.grid_search_hyperparameters = {
                    "max_features": [1.0, "sqrt", "log2"],
                    "n_estimators": [50, 100, 150, 200],
                }
            elif model == Model.PLS:
                # Not set because it depends on the dataset's numbers of features
                self.grid_search_hyperparameters = {}
            elif model == Model.CUBIST:
                self.grid_search_hyperparameters = {
                    "n_committees": [1, 5, 10, 20],
                    "neighbors": [1, 5, 9],
                }
            elif model == Model.XGBOOST:
                self.grid_search_hyperparameters = {
                    "max_depth": [3, 5, 8],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "n_estimators": [10, 50, 100],
                    "colsample_bytree": [0.3, 0.6],
                    "alpha": [0, 10],
                }
        self.random_state = random_state
        self.training_time = None
        self.testing_time = None
        self.best_model = None
        self.best_hyperparameters = None
        self.best_score = None
        self.scaler = None
        self.param_grid_ = None

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

    def fit(
        self,
        dataset: Dataset,
        cv: int | Iterable[tuple[np.ndarray, np.ndarray]] = 5,
    ) -> None:
        """Trains the model on a given dataset.

        If you didn't supply the `init_hyperparameters` option to the constructor, then
        a grid search optimization process takes place as follows:
        Using the sklearn.model_selection.GridSearchCV approach, a grid search using a
        cross-validation splitting strategy specified by cv is performed. After the
        optimal hyperparameters are defined, the model is then retrained on the whole
        dataset.

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        cv: int, or an iterable, default=5
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            * integer, to specify the number of folds
            * An iterable yielding (train, test) splits as arrays of indices

        """
        if self.model in [Model.SVR, Model.CUBIST] and dataset.n_outputs > 1:
            raise SpectraXAIValueError(
                "Cannot create a multi-output model. "
                "Please train a model for each output property."
            )

        # Scale the data for SVR
        if self.model == Model.SVR:
            self.scaler = StandardScaler()
            self.scaler.fit(dataset.X)
            X_train = np.asarray(self.scaler.transform(dataset.X))
        else:
            X_train = dataset.X

        # If hyperparameters are passed, use them; otherwise run grid search
        if len(self.init_hyperparameters) != 0:
            self.best_model = self._new_estimator(grid_search=False).set_params(
                **self.init_hyperparameters
            )
            self.param_grid_ = None
            trn_t0 = time.time()
            self.best_model.fit(X_train, np.squeeze(dataset.Y))
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
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, np.squeeze(dataset.Y))
            trn_t1 = time.time()
            self.best_model = grid_search.best_estimator_
            self.best_hyperparameters = grid_search.best_params_
            self.best_score = grid_search.best_score_
        self.training_time = trn_t1 - trn_t0

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using the best_model from a new unknown input.

        Parameters
        ----------
        X_test: `np.ndarray`
            The new input data to predict their output

        Returns
        -------
        `np.ndarray`
            A np.ndarray of size (n_test_samples, n_outputs) with the predictions

        """
        if self.best_model is None:
            raise RuntimeError("You need to train the model first")
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        tst_t0 = time.time()
        if self.model == Model.SVR:
            if self.scaler is None:
                raise RuntimeError("The SVR scaler is unavailable")
            X_test = np.asarray(self.scaler.transform(X_test))
        y_hat = self.best_model.predict(X_test)
        tst_t1 = time.time()
        self.testing_time = tst_t1 - tst_t0
        return y_hat

    def fit_and_predict(
        self,
        dataset: Dataset,
        preprocess: SpectralPreprocessingSequence = SpectralPreprocessing.NONE,
        idx_trn: np.ndarray | None = None,
        idx_tst: np.ndarray | None = None,
        get_model: bool = False,
    ) -> pandas.DataFrame:
        """Train and test a model in given dataset and spectral pre-processing sequence.

        Pass here the whole dataset of (X, Y) and either specify
        idx_trn (training indices) or idx_tst (testing indices).

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
            If true, also return the generated model. Defaults to False.

        Returns
        -------
        `pandas.DataFrame`
            The accuracy results and assorted metadata for each output property

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
                    dataset, preprocess, idx_trn=idx_trn[fold], get_model=get_model
                )
                result["fold"] = fold + 1
                results.append(result)
            return pandas.concat(results, ignore_index=True)
        elif idx_tst.size > 0 and isinstance(idx_tst[0], np.ndarray):  # Multiple-folds
            results = []
            for fold in range(idx_tst.shape[0]):
                result = self.fit_and_predict(
                    dataset, preprocess, idx_tst=idx_tst[fold], get_model=get_model
                )
                result["fold"] = fold + 1
                results.append(result)
            return pandas.concat(results, ignore_index=True)
        elif idx_trn.size > 0 and idx_tst.size == 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(trn=idx_trn)
        elif idx_trn.size == 0 and idx_tst.size > 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(tst=idx_tst)

        dataset = dataset.preprocess(preprocess)
        self.fit(Dataset(dataset.X[idx_trn], dataset.Y[idx_trn]))
        if self.best_model is None or self.best_hyperparameters is None:
            raise RuntimeError("The model failed to fit")
        y_hat = self.predict(dataset.X[idx_tst])

        results = []
        for i in range(dataset.n_outputs):
            res = metrics(
                dataset.Y[idx_tst, i], y_hat[:, i] if y_hat.ndim == 2 else y_hat
            )
            res["output"] = dataset.Y_names[i]
            # Gather all results
            res["pre_process"] = str(preprocess)
            res["val_score"] = self.best_score
            for key in self._REPORTED_HYPERPARAMETERS[self.model]:
                res[key] = self._hyperparameter(key)
            if self.model == Model.SVR:
                res["SVs"] = len(self.best_model.support_)
            elif self.model == Model.PLS:
                res["feature_importance"] = self.__vip(self.best_model)
            else:
                res["feature_importance"] = self.best_model.feature_importances_
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
            self.fit_and_predict(dataset, preprocess, idx_trn, idx_tst)
            for preprocess in preprocesses
        ]
        return pandas.concat(results, ignore_index=True)
