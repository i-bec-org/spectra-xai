from multiprocessing.sharedctypes import Value
import time
from enum import Enum
from typing import Dict, List, Any, Union

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
from spectraxai.spectra import SpectralPreprocessing, SpectralPreprocessingSequence
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
        """The type of `Model` used"""
        self.best_hyperparameters = best_hyperparameters
        """A dictionary of the best hyperparameters, either set externally or as identified after calling the train function"""
        if len(grid_search_hyperparameters) != 0:
            self.grid_search_hyperparameters = grid_search_hyperparameters
            """A dictionary containing as keys the hyperparameters of the model and as values a list of the potential candidate values"""
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
                # Not set because it depends on the dataset's numbers of features
                self.grid_search_hyperparameters = {}
            elif model == Model.CUBIST:
                self.grid_search_hyperparameters = {
                    "n_committees": [1, 5, 10, 20],
                    "neighbors": [0, 1, 5, 9],
                }
        self.training_time = None
        """Training time in seconds"""
        self.testing_time = None
        """Time for the prediction in seconds"""
        self.best_model = None
        """The best optimized model after tuning the hyperparameters"""
        self.best_score = None
        """The best score (R2 for regression and accuracy for classification) in the internal validation set corresponding to the best model"""

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

    def train(self, dataset: Dataset, cv: Union[int, List] = 5):
        """Trains the model on a given dataset.

        If you didn't supply the `best_hyperparameters` option to the constructor, then
        a grid search optimization process takes place as follows:
        Using the sklearn.model_selection.GridSearchCV approach, a grid search using a
        cross-validation splitting strategy specified by cv is performed. After the optimal
        hyperparameters are defined, the model is then retrained on the whole dataset.

        Parameters
        ----------

        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        cv : int, or an iterable, default=5
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - integer, to specify the number of folds
            - An iterable yielding (train, test) splits as arrays of indices
        """
        if self.model in [Model.SVR, Model.CUBIST] and dataset.n_outputs > 1:
            raise AssertionError(
                "Cannot create a multi-output SVR model. Please train a model for each output property."
            )

        # Scale the data for SVR
        if self.model == Model.SVR:
            self.scaler = StandardScaler()
            self.scaler.fit(dataset.X)
            X_train = self.scaler.transform(dataset.X)
        else:
            X_train = dataset.X

        # If hyperparameters are passed, use them; otherwise run grid search
        if len(self.best_hyperparameters) != 0 and self.training_time is None:
            if self.model == Model.SVR:
                self.best_model = SVR(kernel="rbf", max_iter=5e8)
                self.best_model = self.best_model.set_params(
                    **self.best_hyperparameters
                )
            elif self.model == Model.PLS:
                self.best_model = PLSRegression()
                self.best_model = self.best_model.set_params(
                    **self.best_hyperparameters
                )
            elif self.model == Model.RF:
                self.best_model = RandomForestRegressor()
                self.best_model = self.best_model.set_params(
                    **self.best_hyperparameters
                )
            elif self.model == Model.CUBIST:
                self.best_model = Cubist()
                self.best_model = self.best_model.set_params(
                    **self.best_hyperparameters
                )
            trn_t0 = time.time()
            self.best_model.fit(X_train, np.squeeze(dataset.Y))
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
        if self.best_model is None:
            raise RuntimeError("You need to train the model first")
        tst_t0 = time.time()
        if self.model == Model.SVR:
            X_test = self.scaler.transform(X_test)
        y_hat = self.best_model.predict(X_test)
        tst_t1 = time.time()
        self.testing_time = tst_t1 - tst_t0
        return y_hat

    def train_and_test(
        self,
        dataset: Dataset,
        preprocess: SpectralPreprocessingSequence = SpectralPreprocessing.NONE,
        idx_trn: np.array = np.array([]),
        idx_tst: np.array = np.array([]),
        get_model: bool = False,
    ) -> List[Dict]:
        """Trains and tests a model given a dataset and a spectral pre-processing sequence.

        Pass here the whole dataset of (X, Y) and either specify idx_trn (training indices) or idx_tst (testing indices).

        Parameters
        ----------

        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        preprocess: `spectraxai.spectra.SpectralPreprocessingSequence`
            Optional pre-processing sequence. Defaults to SpectralPreprocessing.NONE.

        idx_trn: `np.array`
            The indices of the trn samples. Defaults to np.array([]).

        idx_tst: `np.array`
            The indices of the tst samples. Defaults to np.array([]).

        get_model: `bool`
            If true, also return the generated model. Defaults to False.

        Returns
        -------
        `List[Dict]`
            A dictionary containing the accuracy results and assorted metadata for each output property
        """

        if idx_tst.size == 0 and idx_trn.size == 0:
            raise AssertionError(
                "You need to specify either tst or trn indices or both"
            )
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        elif idx_trn.size > 0 and idx_tst.size == 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(
                trn=idx_trn
            )
        elif idx_trn.size == 0 and idx_tst.size > 0:
            idx_trn, idx_tst = dataset.train_test_split_explicit(
                tst=idx_tst
            )

        dataset = dataset.preprocess(preprocess)
        self.train(Dataset(dataset.X[idx_trn], dataset.Y[idx_trn]))
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
            if self.model == Model.SVR:
                for key in ["C", "epsilon", "gamma"]:
                    res[key] = self.best_hyperparameters[key]
                res["SVs"] = len(self.best_model.support_)
            elif self.model == Model.PLS:
                res["n_components"] = self.best_hyperparameters["n_components"]
                res["VIP"] = self.__vip(self.best_model)
            elif self.model == Model.RF:
                for key in ["max_features", "n_estimators"]:
                    res[key] = self.best_hyperparameters[key]
                res["feature_importance"] = self.best_model.feature_importances_
            elif self.model == Model.CUBIST:
                for key in ["n_committees", "neighbors"]:
                    res[key] = self.best_hyperparameters[key]
                res["feature_importance"] = self.best_model.feature_importances_
            res["training_time"] = self.training_time
            res["testing_time"] = self.testing_time
            if get_model:
                res["model"] = self.best_model
            results.append(res)
        return results

    def train_and_test_with_sequence(
        self,
        dataset: Dataset,
        preprocesses: List[SpectralPreprocessingSequence] = [],
        idx_trn: np.array = np.array([]),
        idx_tst: np.array = np.array([]),
    ) -> pandas.DataFrame:
        """Train a model using different SpectralPreprocessingSequences and predict on the test set

        A short-hand version to quickly test different pre-treatments, calling the train_and_test function

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            the Dataset to train the model

        preprocesses: `spectraxai.spectra.List[SpectralPreprocessingSequence]`
            List of pre-processing sequences to test. Defaults to [].

        idx_trn: `np.array`
            The indices of the trn samples. Defaults to np.array([]).

        idx_tst: `np.array`
            The indices of the tst samples. Defaults to np.array([]).

        Returns
        -------
        `pandas.DataFrame`
            Returns a dataframe with the results of the trained models. By default, no model is returned to keep a low memory footprint.
        """
        if len(preprocesses) == 0:
            raise ValueError(
                "The list of SpectralPreprocessingSequence may not be empty"
            )
        if idx_tst.size == 0 and idx_trn.size == 0:
            raise AssertionError("You need to specify either tst or trn indices")
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        common = [
            "pre_process",
            "training_time",
            "testing_time",
            "val_score",
            "N",
            "RMSE",
            "R2",
            "RPIQ",
        ]
        if (idx_trn.size > 0 and isinstance(idx_trn[0], np.ndarray)) or (
            idx_tst.size > 0 and isinstance(idx_tst[0], np.ndarray)
        ):
            common += ["fold"]
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
            if idx_trn.size > 0 and isinstance(idx_trn[0], np.ndarray):
                for fold in range(idx_trn.shape[0]):
                    for result in self.train_and_test(
                        dataset, preprocess, idx_trn=idx_trn[fold]
                    ):
                        result["fold"] = fold + 1
                        dataFrame.loc[len(dataFrame)] = result
            elif idx_tst.size > 0 and isinstance(idx_tst[0], np.ndarray):
                for fold in range(idx_tst.shape[0]):
                    for result in self.train_and_test(
                        dataset, preprocess, idx_tst=idx_tst[fold]
                    ):
                        result["fold"] = fold + 1
                        dataFrame.loc[len(dataFrame)] = result
            else:
                for result in self.train_and_test(
                    dataset, preprocess, idx_trn, idx_tst
                ):
                    dataFrame.loc[len(dataFrame)] = result
        return dataFrame
