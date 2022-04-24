import time
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas

from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from cubist import Cubist
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from spectraxai.utils.modelAssessment import metrics
from spectraxai.utils.svrParams import sigest
from spectraxai.spectra import SpectralPreprocessingSequence
from spectraxai.dataset import Dataset


class Model(str, Enum):
    """A model class to describe commonly used ML models for spectral processing"""

    PLS = "Partial Least Squares"
    SVR = "Support Vector Regression"
    RF = "Random Forest"
    CUBIST = "Cubist"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value


class StandardModel:
    """Class with standard models for machine learning that can be applied to spectral datasets"""

    def __init__(
        self,
        model: Model,
        best_hyperparameters: Dict = {},
        grid_search_hyperparameters: Dict = {},
    ):
        """
        Parameters
        ----------

        model: `Model`
                Select a model from `Model` class.

        best_hyperparameters: `dict`
            A dictionary of pre-selected hyperparameters (e.g. a best model)

        grid_search_hyperparameters: `dict`
            Specify custom grid search range for the hyperparameters
        """
        if len(best_hyperparameters) != 0 and len(grid_search_hyperparameters) != 0:
            raise AssertionError(
                "You cannot specify both best_hyperparameters and grid_search_hyperparameters"
            )
        self.model = model
        self.best_hyperparameters = best_hyperparameters
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
                    "max_features": ["auto", "sqrt", "log2"],
                    "n_estimators": [50, 100, 150, 200],
                }
            elif model == Model.PLS:
                self.grid_search_hyperparameters = {}
            elif model == Model.CUBIST:
                self.grid_search_hyperparameters = {
                    "n_committees": [1, 5, 10, 20],
                    "neighbors": [0, 1, 5, 9],
                }

    def __vip(self, model):
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
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        preprocess: SpectralPreprocessingSequence = None,
        idx_trn: np.array = np.array([]),
        idx_tst: np.array = np.array([]),
        get_model: bool = False,
    ):
        """
        Train the model giving the X, Y, preprocess sequence, idx_trn or idx_tst.

        Returns a dict of the trained model
        """
        if preprocess is None:
            raise AssertionError("You need to specify Spectral Preprocessing Sequence")
        if idx_tst.size == 0 and idx_trn.size == 0:
            raise AssertionError(
                "You need to specify either tst or trn indices or both"
            )
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        if idx_trn.size > 0 and idx_tst.size == 0:
            X_train, X_test, y_train, y_test, _, _ = (
                Dataset(X, Y)
                .preprocess(preprocess)
                .train_test_split_explicit(trn=idx_trn)
            )
        elif idx_trn.size == 0 and idx_tst.size > 0:
            X_train, X_test, y_train, y_test, _, _ = (
                Dataset(X, Y)
                .preprocess(preprocess)
                .train_test_split_explicit(tst=idx_tst)
            )
        else:
            X_train, X_test, y_train, y_test = (
                X[idx_trn],
                X[idx_tst],
                Y[idx_trn],
                Y[idx_tst],
            )

        # Scale the data for SVR
        if self.model == Model.SVR:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        #     ymean, ystd = np.mean(y_train), np.std(y_train)
        #     y_train = (y_train - ymean) / ystd

        # If hyperparameters are passed, use them; otherwise run grid search
        if len(self.best_hyperparameters) != 0:
            if self.model == Model.SVR:
                model = SVR(kernel="rbf", max_iter=5e8)
                model = model.set_params(**self.best_hyperparameters)
            elif self.model == Model.PLS:
                model = PLSRegression()
                model = model.set_params(**self.best_hyperparameters)
            elif self.model == Model.RF:
                model = RandomForestRegressor()
                model = model.set_params(**self.best_hyperparameters)
            elif self.model == Model.CUBIST:
                model = Cubist()
                model = model.set_params(**self.best_hyperparameters)
            trn_t0 = time.time()
            model.fit(X_train, np.squeeze(y_train))
            trn_t1 = time.time()
        else:
            if self.model == Model.SVR:
                if "gamma" not in self.grid_search_hyperparameters:
                    gamma = sigest(X_train)
                    self.grid_search_hyperparameters["gamma"] = np.linspace(
                        gamma[0], gamma[2], num=5
                    )
                model = SVR(kernel="rbf", max_iter=5e8)
            elif self.model == Model.PLS:
                if "n_components" not in self.grid_search_hyperparameters:
                    self.grid_search_hyperparameters["n_components"] = np.arange(
                        1, min(100, X_train.shape[1]), 1
                    )
                model = PLSRegression()
            elif self.model == Model.RF:
                model = RandomForestRegressor()
            elif self.model == Model.CUBIST:
                model = Cubist()
            trn_t0 = time.time()
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.grid_search_hyperparameters,
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, np.squeeze(y_train))
            trn_t1 = time.time()
            # grid_search.best_params_ has the best Parameters
            model = grid_search.best_estimator_

        # Predict on test
        tst_t0 = time.time()
        y_hat = model.predict(X_test)
        tst_t1 = time.time()
        # if self.model == Model.SVR:
        #     y_hat = y_hat * ystd + ymean
        res = metrics(y_test, y_hat)

        # Gather all results
        res["pre-process"] = str(preprocess)
        if self.model == Model.SVR:
            for key in ["C", "epsilon", "gamma"]:
                res[key] = model.get_params()[key]
            res["SVs"] = len(model.support_)
        elif self.model == Model.PLS:
            res["n_components"] = model.get_params()["n_components"]
            res["VIP"] = self.__vip(model)
        elif self.model == Model.RF:
            for key in ["max_features", "n_estimators"]:
                res[key] = model.get_params()[key]
            res["feature_importance"] = model.feature_importances_
        elif self.model == Model.CUBIST:
            for key in ["n_committees", "neighbors"]:
                res[key] = model.get_params()[key]
            res["feature_importance"] = model.feature_importances_
        res["TrainingTime"] = trn_t1 - trn_t0
        res["TestingTime"] = tst_t1 - tst_t0
        if get_model:
            res["model"] = model
        return res

    def train_with_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        preprocesses: List[SpectralPreprocessingSequence] = [],
        idx_trn: np.array = np.array([]),
        idx_tst: np.array = np.array([]),
    ):
        """Train the model giving the X, Y, list of preprocesses sequence, idx_trn or idx_tst. Returns a dataframe of the trained models"""
        if len(preprocesses) == 0:
            raise AssertionError(
                "You need to specify a list of Spectral Preprocessing Sequence"
            )
        if idx_tst.size == 0 and idx_trn.size == 0:
            raise AssertionError("You need to specify either tst or trn indices")
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        common = [
            "pre-process",
            "TrainingTime",
            "TestingTime",
            "N",
            "RMSE",
            "R2",
            "RPIQ",
        ]
        if self.model == Model.PLS:
            dataFrame = pandas.DataFrame(columns=common + ["n_components", "VIP"])
        elif self.model == Model.SVR:
            dataFrame = pandas.DataFrame(
                columns=common + ["epsilon", "C", "gamma", "SVs"]
            )
        elif self.model == Model.RF:
            dataFrame = pandas.DataFrame(
                columns=common + ["max_features", "n_estimators", "feature_importance"]
            )
        elif self.model == Model.CUBIST:
            dataFrame = pandas.DataFrame(
                columns=common + ["n_committees", "neighbors", "feature_importance"]
            )
        for preprocess in preprocesses:
            dataFrame.loc[len(dataFrame)] = self.train(
                X, Y, preprocess, idx_trn, idx_tst
            )
        return dataFrame
