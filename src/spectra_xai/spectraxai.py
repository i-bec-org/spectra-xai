import ast
import pandas
import time
import numpy as np
import utils_xai.kennardStone as kennardStone
from enum import Enum
from numbers import Number
from scipy.signal import savgol_filter
from utils_xai.continuumRemoval import continuum_removal
from typing import Union, Tuple, Dict, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils_xai.modelAssessment import metrics
from utils_xai.svrParams import sigest, estimateC


class SpectralPreprocessing(str, Enum):
    """Spectral Preprocessing class"""

    NONE = "no"
    REF = "reflectance"
    ABS = "absorbance"
    SNV = "SNV"
    SG1 = "SG1"
    SG2 = "SG2"
    CR = "continuum-removal"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value

    def __get_class(string):
        return {
            "no": SpectralPreprocessing.NONE,
            "reflectance": SpectralPreprocessing.REF,
            "absorbance": SpectralPreprocessing.ABS,
            "SNV": SpectralPreprocessing.SNV,
            "SG1": SpectralPreprocessing.SG1,
            "SG2": SpectralPreprocessing.SG2,
            "continuum-removal": SpectralPreprocessing.CR,
        }.get(string, None)

    def __init_class(lst):
        for i in range(len(lst)):
            if isinstance(lst[i], tuple):
                lst[i] = (SpectralPreprocessing.__get_class(lst[i][0]), lst[i][1])
            elif isinstance(lst[i], list):
                SpectralPreprocessing.__init_class(lst[i])
            else:
                lst[i] = SpectralPreprocessing.__get_class(lst[i])
        return lst

    def init_class(string):
        lst = ast.literal_eval(
            string.replace("no", "'no'")
            .replace("reflectance", "'reflectance'")
            .replace("absorbance", "'absorbance'")
            .replace("SNV", "'SNV'")
            .replace("SG1", "'SG1'")
            .replace("SG2", "'SG2'")
            .replace("continuum-removal", "'continuum-removal'")
        )

        return SpectralPreprocessing.__init_class(lst)


# Options passed to pre-processing (e.g. window_length for SG)
SpectralPreprocessingOptions = Union[
    SpectralPreprocessing, Tuple[SpectralPreprocessing, Dict[str, int]]
]
# A sequence of spectral pre-processing (e.g. SG1 + SNV)
SpectralPreprocessingSequence = List[
    Union[SpectralPreprocessingOptions, List[SpectralPreprocessingOptions]]
]


class Spectra:
    """Spectra class which apply preprocessing to input data."""

    def __init__(self, X: np.ndarray) -> None:
        self.X = X

    def reflectance(self) -> np.ndarray:
        """Transform absorbance to reflectance"""
        return Spectra(-1 * self.X ** 10)

    def absorbance(self) -> np.ndarray:
        """Transform reflectance to absorbance"""
        return Spectra(-1 * np.log10(self.X))

    def snv(self) -> np.ndarray:
        snv = np.zeros(self.X.shape)
        mu, sd = self.X.mean(axis=-1), self.X.std(axis=-1, ddof=1)
        for i in range(np.shape(self.X)[0]):
            snv[i, :] = (self.X[i, :] - mu[i]) / sd[i]
        return Spectra(snv)

    def sg(self, **kwargs) -> np.ndarray:
        return Spectra(savgol_filter(self.X, **kwargs))

    def cr(self) -> np.ndarray:
        return Spectra(continuum_removal(self.X))

    def apply(self, method: SpectralPreprocessing, **kwargs) -> np.ndarray:
        if method == SpectralPreprocessing.REF:
            return self.reflectance()
        elif method == SpectralPreprocessing.ABS:
            return self.absorbance()
        elif method == SpectralPreprocessing.SNV:
            return self.snv()
        elif method == SpectralPreprocessing.SG1:
            return self.sg(deriv=1, **kwargs)
        elif method == SpectralPreprocessing.SG2:
            return self.sg(deriv=2, **kwargs)
        elif method == SpectralPreprocessing.CR:
            return self.cr()
        elif method == SpectralPreprocessing.NONE:
            return self
        else:
            raise RuntimeError


class Scale(str, Enum):
    STANDARD = "standard"
    MINMAX = "min-max"


class DatasetSplit(str, Enum):
    """DatasetSplit class"""

    RANDOM = "random"
    KENNARD_STONE = "Kennard-Stone"
    CLHS = "clhs"


DataSplit = Tuple[
    np.ndarray,  # X_trn
    np.ndarray,  # X_tst
    np.ndarray,  # Y_trn
    np.ndarray,  # Y_tst
    np.array,  # idx_trn
    np.array,  # idx_tst
]


class Dataset:
    """Dataset class manages the dataset"""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise AssertionError("X and Y don't have the same number of rows!")
        if X.ndim != 2:
            raise AssertionError("X should have exactly two dimensions")
        self.X = X.to_numpy() if isinstance(X, pandas.DataFrame) else X
        if isinstance(Y, pandas.DataFrame) or isinstance(Y, pandas.Series):
            Y = Y.to_numpy()
        self.Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)

    def train_test_split(self, split: DatasetSplit, trn: Number) -> DataSplit:
        """Splits dataset with method split to train and test by trn percentage. Returns X_trn, X_tst, y_trn, y_tst, idx_trn, idx_tst"""
        indices = np.arange(self.X.shape[0])
        if trn <= 0 or trn >= 1:
            raise AssertionError("trn param should be in the (0, 1) range")
        if split == DatasetSplit.RANDOM:
            return train_test_split(self.X, self.Y, indices, train_size=trn)
        elif split == DatasetSplit.KENNARD_STONE:
            return kennardStone.train_test_split(
                self.X, self.Y, indices, test_size=(1 - trn)
            )
        elif split == DatasetSplit.CLHS:
            raise NotImplementedError("clhs not implemented yet")
        else:
            raise RuntimeError("Not a valid split method!")

    def train_test_split_explicit(
        self, trn: np.array = np.array([]), tst: np.array = np.array([])
    ) -> DataSplit:
        """Splits dataset to train and test by trn or tst indices. Returns X_trn, X_tst, y_trn, y_tst, idx_trn, idx_tst"""
        if tst.size == 0 and trn.size == 0:
            raise AssertionError("You need to specify either tst or trn indices")
        if tst.size > 0 and trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        if tst.size > 0:
            trn = np.array(list(set(range(0, self.Y.shape[0])).difference(set(tst))))
        else:
            tst = np.array(list(set(range(0, self.Y.shape[0])).difference(set(trn))))
        return self.X[trn, :], self.X[tst, :], self.Y[trn, :], self.Y[tst, :], trn, tst

    def preprocess(self, method: SpectralPreprocessingSequence):
        """Preprocess dataset by method. Returns self obj"""
        self.X = self.__preprocess(self.X, method)
        return self

    def preprocess_3D(self, methods: List[SpectralPreprocessingSequence]):
        """Preprocess 3D matrix by methods in a list structure. Returns"""
        if len(methods) <= 1:
            raise AssertionError(
                "A 3D matrix must contain at least two pre-processing sequences"
            )
        X = np.empty((self.X.shape[0], self.X.shape[1], len(methods)))
        for i, method in enumerate(methods):
            thisX = np.copy(self.X)
            X[:, :, i] = self.__preprocess(thisX, method)
        return X

    def unscale_X(
        self, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        self.X = Dataset.unscale_X(self.X, method, set_params, set_attributes)
        return self

    def unscale_X(
        X: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        if len(set_attributes) == 0:
            raise AssertionError("You need to specify set_attributes")
        if X.ndim == 3:
            if method == Scale.STANDARD:
                scaler = [StandardScaler() for _ in range(X.shape[2])]
            elif method == Scale.MINMAX:
                scaler = [MinMaxScaler() for _ in range(X.shape[2])]
            for i in range(X.shape[2]):
                if len(set_params) != 0:
                    scaler[i] = scaler[i].set_params(**set_params[i])
                if method == Scale.STANDARD:
                    scaler[i].scale_ = set_attributes[i]["scale_"]
                    scaler[i].mean_ = set_attributes[i]["mean_"]
                    scaler[i].var_ = set_attributes[i]["var_"]
                    scaler[i].n_samples_seen_ = set_attributes[i]["n_samples_seen_"]
                    X[:, :, i] = scaler[i].inverse_transform(X[:, :, i])
                elif method == Scale.MINMAX:
                    scaler[i].min_ = set_attributes[i]["min_"]
                    scaler[i].scale_ = set_attributes[i]["scale_"]
                    scaler[i].data_min_ = set_attributes[i]["data_min_"]
                    scaler[i].data_max_ = set_attributes[i]["data_max_"]
                    scaler[i].data_range_ = set_attributes[i]["data_range_"]
                    X[:, :, i] = scaler[i].inverse_transform(X[:, :, i])
        else:
            if method == Scale.STANDARD:
                scaler = StandardScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                scaler.scale_ = set_attributes[0]["scale_"]
                scaler.mean_ = set_attributes[0]["mean_"]
                scaler.var_ = set_attributes[0]["var_"]
                scaler.n_samples_seen_ = set_attributes[0]["n_samples_seen_"]
                X = scaler.inverse_transform(X)
            elif method == Scale.MINMAX:
                scaler = MinMaxScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                scaler.min_ = set_attributes[0]["min_"]
                scaler.scale_ = set_attributes[0]["scale_"]
                scaler.data_min_ = set_attributes[0]["data_min_"]
                scaler.data_max_ = set_attributes[0]["data_max_"]
                scaler.data_range_ = set_attributes[0]["data_range_"]
                X = scaler.inverse_transform(X)
        return X

    def scale_X(self, method: Scale, set_params: List = [], set_attributes: List = []):
        self.X, self.get_scale_X_params = Dataset.scale_X(
            self.X, method, set_params, set_attributes
        )
        return self

    def scale_X(
        X: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        if X.ndim == 3:
            if method == Scale.STANDARD:
                scaler = [StandardScaler() for _ in range(X.shape[2])]
            elif method == Scale.MINMAX:
                scaler = [MinMaxScaler() for _ in range(X.shape[2])]
            get_params = []
            get_attributes = []
            for i in range(X.shape[2]):
                if len(set_params) != 0:
                    scaler[i] = scaler[i].set_params(**set_params[i])
                get_params.append(scaler[i].get_params())
                if method == Scale.STANDARD:
                    if len(set_attributes) != 0:
                        scaler[i].scale_ = set_attributes[i]["scale_"]
                        scaler[i].mean_ = set_attributes[i]["mean_"]
                        scaler[i].var_ = set_attributes[i]["var_"]
                        scaler[i].n_samples_seen_ = set_attributes[i]["n_samples_seen_"]
                    else:
                        scaler[i] = scaler[i].fit(X[:, :, i])
                    X[:, :, i] = scaler[i].transform(X[:, :, i])
                    get_attributes.append(
                        {
                            "scale_": scaler[i].scale_,
                            "mean_": scaler[i].mean_,
                            "var_": scaler[i].var_,
                            "n_samples_seen_": scaler[i].n_samples_seen_,
                        }
                    )
                elif method == Scale.MINMAX:
                    if len(set_attributes) != 0:
                        scaler[i].min_ = set_attributes[i]["min_"]
                        scaler[i].scale_ = set_attributes[i]["scale_"]
                        scaler[i].data_min_ = set_attributes[i]["data_min_"]
                        scaler[i].data_max_ = set_attributes[i]["data_max_"]
                        scaler[i].data_range_ = set_attributes[i]["data_range_"]
                    else:
                        scaler[i] = scaler[i].fit(X[:, :, i])
                    X[:, :, i] = scaler[i].transform(X[:, :, i])
                    get_attributes.append(
                        {
                            "min_": scaler[i].min_,
                            "scale_": scaler[i].scale_,
                            "data_min_": scaler[i].data_min_,
                            "data_max_": scaler[i].data_max_,
                            "data_range_": scaler[i].data_range,
                        }
                    )
        else:
            if method == Scale.STANDARD:
                scaler = StandardScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                if len(set_attributes) != 0:
                    scaler.scale_ = set_attributes[0]["scale_"]
                    scaler.mean_ = set_attributes[0]["mean_"]
                    scaler.var_ = set_attributes[0]["var_"]
                    scaler.n_samples_seen_ = set_attributes[0]["n_samples_seen_"]
                else:
                    scaler = scaler.fit(X)
                X = scaler.transform(X)
                get_params = [scaler.get_params()]
                get_attributes = [
                    {
                        "scale_": scaler.scale_,
                        "mean_": scaler.mean_,
                        "var_": scaler.var_,
                        "n_samples_seen_": scaler.n_samples_seen_,
                    }
                ]
            elif method == Scale.MINMAX:
                scaler = MinMaxScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                if len(set_attributes) != 0:
                    scaler.min_ = set_attributes[0]["min_"]
                    scaler.scale_ = set_attributes[0]["scale_"]
                    scaler.data_min_ = set_attributes[0]["data_min_"]
                    scaler.data_max_ = set_attributes[0]["data_max_"]
                    scaler.data_range_ = set_attributes[0]["data_range_"]
                else:
                    scaler = scaler.fit(X)
                X = scaler.transform(X)
                get_params = [scaler.get_params()]
                get_attributes = [
                    {
                        "min_": scaler.min_,
                        "scale_": scaler.scale_,
                        "data_min_": scaler.data_min_,
                        "data_max_": scaler.data_max_,
                        "data_range_": scaler.data_range_,
                    }
                ]
        return X, {"params": get_params, "attributes": get_attributes}

    def unscale_Y(
        self, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        self.Y = Dataset.unscale_Y(self.Y, method, set_params, set_attributes)
        return self

    def unscale_Y(
        Y: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        if len(set_attributes) == 0:
            raise AssertionError("You need to specify set_attributes")
        if method == Scale.STANDARD:
            scaler = StandardScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            scaler.scale_ = set_attributes[0]["scale_"]
            scaler.mean_ = set_attributes[0]["mean_"]
            scaler.var_ = set_attributes[0]["var_"]
            scaler.n_samples_seen_ = set_attributes[0]["n_samples_seen_"]
            Y = scaler.inverse_transform(Y)
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            scaler.min_ = set_attributes[0]["min_"]
            scaler.scale_ = set_attributes[0]["scale_"]
            scaler.data_min_ = set_attributes[0]["data_min_"]
            scaler.data_max_ = set_attributes[0]["data_max_"]
            scaler.data_range_ = set_attributes[0]["data_range_"]
            Y = scaler.inverse_transform(Y)
        return Y

    def scale_Y(self, method: Scale, set_params: List = [], set_attributes: List = []):
        self.Y, self.get_scale_Y_params = Dataset.scale_Y(
            self.Y, method, set_params, set_attributes
        )
        return self

    def scale_Y(
        Y: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        if method == Scale.STANDARD:
            scaler = StandardScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            if len(set_attributes) != 0:
                scaler.scale_ = set_attributes[0]["scale_"]
                scaler.mean_ = set_attributes[0]["mean_"]
                scaler.var_ = set_attributes[0]["var_"]
                scaler.n_samples_seen_ = set_attributes[0]["n_samples_seen_"]
            else:
                scaler = scaler.fit(Y)
            Y = scaler.transform(Y)
            get_attributes = [
                {
                    "scale_": scaler.scale_,
                    "mean_": scaler.mean_,
                    "var_": scaler.var_,
                    "n_samples_seen_": scaler.n_samples_seen_,
                }
            ]
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            if len(set_attributes) != 0:
                scaler.min_ = set_attributes[0]["min_"]
                scaler.scale_ = set_attributes[0]["scale_"]
                scaler.data_min_ = set_attributes[0]["data_min_"]
                scaler.data_max_ = set_attributes[0]["data_max_"]
                scaler.data_range_ = set_attributes[0]["data_range_"]
            else:
                scaler = scaler.fit(Y)
            Y = scaler.transform(Y)
            get_attributes = [
                {
                    "min_": scaler.min_,
                    "scale_": scaler.scale_,
                    "data_min_": scaler.data_min_,
                    "data_max_": scaler.data_max_,
                    "data_range_": scaler.data_range_,
                }
            ]
        return Y, {"params": [scaler.get_params()], "attributes": get_attributes}

    def __preprocess(self, X: np.ndarray, method: SpectralPreprocessingSequence):
        if isinstance(method, str):
            X = Spectra(X).apply(method).X
        elif isinstance(method, tuple):
            X = Spectra(X).apply(method[0], **method[1]).X
        elif isinstance(method, list):
            for each_method in method:
                if isinstance(each_method, str):
                    newX = Spectra(X).apply(each_method).X
                elif isinstance(each_method, tuple):
                    newX = Spectra(X).apply(each_method[0], **each_method[1]).X
            X = newX
        return X


class Model(str, Enum):
    """Model class"""

    PLS = "Partial Least Squares"
    SVR = "Support Vector Regression"
    RF = "Random Forest"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value


class StandardModel:
    """Class with standard models for machine learning"""

    def __init__(
        self,
        model: Model,
        best_hyperparameters: Dict = {},
        grid_search_hyperparameters: Dict = {},
    ):
        """
        Select a model from Model class.
        Option: Use the best_hyperparameters for the best model or
         grid_search_hyperparameters for searching the optimum hyperparameters in a custom range.
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
                self.grid_search_hyperparameters = {
                    "n_components": np.arange(1, min(100, X.shape[1]), 1)
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
    ):
        """Train the model giving the X, Y, preprocess sequense, idx_trn or idx_tst. Returns dict of the trained model"""
        if preprocess == None:
            raise AssertionError("You need to specify Spectral Preprocessing Sequence")
        if idx_tst.size == 0 and idx_trn.size == 0:
            raise AssertionError("You need to specify either tst or trn indices")
        if idx_tst.size > 0 and idx_trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        if idx_trn.size > 0:
            X_train, X_test, y_train, y_test, _, _ = (
                Dataset(X, Y)
                .preprocess(preprocess)
                .train_test_split_explicit(trn=idx_trn)
            )
        else:
            X_train, X_test, y_train, y_test, _, _ = (
                Dataset(X, Y)
                .preprocess(preprocess)
                .train_test_split_explicit(tst=idx_tst)
            )

        # Scale the data for SVR
        if self.model == Model.SVR:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            ymean, ystd = np.mean(y_train), np.std(y_train)
            y_train = (y_train - ymean) / ystd

        # Find optimal SVR RBF parameters through grid search and PLS n_components
        if len(self.best_hyperparameters) != 0:
            if self.model == Model.SVR:
                model = SVR(kernel="rbf", max_iter=5e8)
                model.set_params(**self.best_hyperparameters)
            elif self.model == Model.PLS:
                model = PLSRegression()
                model.set_params(**self.best_hyperparameters)
            elif self.model == Model.RF:
                model = RandomForestRegressor()
                model.set_params(**self.best_hyperparameters)
            trn_t0 = time.time()
            model.fit(X_train, y_train)
            trn_t1 = time.time()
        else:
            if self.model == Model.SVR:
                if not "gamma" in self.grid_search_hyperparameters:
                    gamma = sigest(X_train)
                    self.grid_search_hyperparameters["gamma"] = np.linspace(
                        gamma[0], gamma[2], num=5
                    )
                model = SVR(kernel="rbf", max_iter=5e8)
            elif self.model == Model.PLS:
                model = PLSRegression()
            elif self.model == Model.RF:
                model = RandomForestRegressor()
            trn_t0 = time.time()
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.grid_search_hyperparameters,
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            trn_t1 = time.time()
            # grid_search.best_params_ has the best Parameters
            model = grid_search.best_estimator_

        # Predict on test
        tst_t0 = time.time()
        y_hat = model.predict(X_test)
        tst_t1 = time.time()
        if self.model == Model.SVR:
            y_hat = y_hat * ystd + ymean
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
        res["TrainingTime"] = trn_t1 - trn_t0
        res["TestingTime"] = tst_t1 - tst_t0
        # self.dataFrame.loc[len(self.dataFrame)] = res
        return res

    def train_with_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        preprocesses: List[SpectralPreprocessingSequence] = [],
        idx_trn: np.array = np.array([]),
        idx_tst: np.array = np.array([]),
    ):
        """Train the model giving the X, Y, list of preprocesses sequense, idx_trn or idx_tst. Returns a dataframe of the trained models"""
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
        for preprocess in preprocesses:
            dataFrame.loc[len(dataFrame)] = self.train(
                X, Y, preprocess, idx_trn, idx_tst
            )
        return dataFrame
