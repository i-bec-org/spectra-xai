import ast
import pandas
import time
import numpy as np
from enum import Enum
from numbers import Number
from scipy.signal import savgol_filter
from typing import Union, Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from spectraxai.utils.continuumRemoval import continuum_removal
from spectraxai.utils.modelAssessment import metrics
from spectraxai.utils.svrParams import sigest, estimateC
import spectraxai.utils.kennardStone as kennardStone


class SpectralPreprocessing(str, Enum):
    """
    Spectral Preprocessing enum

    A collection of different spectral pre-processing (or pre-treatments)
    that may be applied to a spectral matrix.
    """

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

    def init_class(string: str):
        """Initialize a SpectralPreprocessing object from its string representation"""
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
    """
    Spectra class to hold a 2-D spectral matrix.

    Can accept a 1-D vector but always returns a 2-D matrix.
    """

    def __init__(self, X: np.ndarray) -> None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise AssertionError("X should a 2-D matrix")
        self.X = X

    def reflectance(self) -> np.ndarray:
        """Transform absorbance to reflectance"""
        return Spectra(-1 * self.X ** 10)

    def absorbance(self) -> np.ndarray:
        """Transform reflectance to absorbance"""
        return Spectra(-1 * np.log10(self.X))

    def snv(self) -> np.ndarray:
        """Apply the standard normal variate transform"""
        snv = np.zeros(self.X.shape)
        mu, sd = self.X.mean(axis=-1), self.X.std(axis=-1, ddof=1)
        for i in range(np.shape(self.X)[0]):
            snv[i, :] = (self.X[i, :] - mu[i]) / sd[i]
        return Spectra(snv)

    def sg(self, **kwargs) -> np.ndarray:
        """
        Apply a general Savitzky–Golay transform.

        You need to pass as kwargs the parameters of scipy.signal.savgol_filter
        """
        return Spectra(savgol_filter(self.X, **kwargs))

    def cr(self) -> np.ndarray:
        """Transform using the Continuum Removal"""
        return Spectra(continuum_removal(self.X))

    def apply(self, method: SpectralPreprocessing, **kwargs) -> np.ndarray:
        """Apply the transform specified by method"""
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
    """Scaling of an input feature (or of the output) supported by the `Dataset` class"""
    STANDARD = "standard"
    MINMAX = "min-max"
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value


class DatasetSplit(str, Enum):
    """Types of dataset split supported by the `Dataset` class"""

    RANDOM = "random"
    KENNARD_STONE = "Kennard-Stone"
    CLHS = "clhs"
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value


DataSplit = Tuple[
    np.ndarray,  # X_trn
    np.ndarray,  # X_tst
    np.ndarray,  # Y_trn
    np.ndarray,  # Y_tst
    np.ndarray,  # idx_trn
    np.ndarray,  # idx_tst
]
"""A tuple"""


class Dataset:
    """
    A general class to manage the dataset (i.e. X and Y).
    
    Use this class to pass your 2D spectral matrix and 1D or 2D output properties.
    Supports methods for pre-processing X, scaling X and Y, splitting the dataset, and more. 
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        
        Parameters
        ----------
        
        X: `numpy.ndarray`
            A 2D matrix of the spectra
        
        Y: `numpy.ndarray`
            A 1D vector or 2D matrix of the output property(ies).
            If 1D it will be implicitly converted to 2D. 
        """
        if X.shape[0] != Y.shape[0]:
            raise AssertionError("X and Y don't have the same number of rows!")
        if X.ndim != 2:
            raise AssertionError("X should have exactly two dimensions")
        self.X = X.to_numpy() if isinstance(X, pandas.DataFrame) else X
        if isinstance(Y, pandas.DataFrame) or isinstance(Y, pandas.Series):
            Y = Y.to_numpy()
        self.Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)

    def train_test_split(self, split: DatasetSplit, trn: Number) -> DataSplit:
        """
        Splits dataset with method split to train and test by trn percentage. 
        
        Parameters
        ----------
        
        split: `DatasetSplit`
                The method used to split the dataset
        
        trn: `Number`
                A float number (between 0 and 1) indicating the percetange of the training dataset
        
        Returns
        -------
        `DataSplit`
            The X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst tuple
        """
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
        """
        Splits dataset to train and test from pre-selected by the user trn or tst indices. 
        
        Returns
        -------
        `DataSplit`
            The X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst tuple
        """
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
        """
            Preprocess dataset by method.

            Parameters
            ----------

            method: `SpectralPreprocessingSequence`
                The method for the preprocess.

            Returns
            -------
            `Dataset`
                A Dataset object.
        """
        self.X = self.__preprocess(self.X, method)
        return self

    def preprocess_3D(self, methods: List[SpectralPreprocessingSequence]):
        """
        Preprocess 3D matrix by methods in a list structure.

        Parameters
        ----------

        methods: `List[SpectralPreprocessingSequence]`
            The methods for the preprocess.

        Returns
        -------
        `numpy.ndarray`
            A 3D matrix.
        """
        if len(methods) <= 1:
            raise AssertionError(
                "A 3D matrix must contain at least two pre-processing sequences"
            )
        X = np.empty((self.X.shape[0], self.X.shape[1], len(methods)))
        for i, method in enumerate(methods):
            thisX = np.copy(self.X)
            X[:, :, i] = self.__preprocess(thisX, method)
        return X

    def apply_PCA(self, set_params: Dict = {}):
        pca = PCA()
        if len(set_params) != 0:
            pca = pca.set_params(**set_params)
        return pca.fit_transform(self.X)

    def apply_unscale_X(
        self, method: Scale, set_params: List = [], set_attributes: List = [], X: np.ndarray = np.array([])
    ):
        """
            Unscale X matrix of the spectra with Scale method.

            Parameters
            ----------

            method: `Scale`
                    The method is used to scale X

            set_params: `List`
                    A list of dicts with the parameters of each Scale method.

            set_attributes: `List`
                    A list of dicts with the attributes of each Scale method.

            X: `numpy.ndarray`
                    A 2D or 3D matrix of the spectra for scaled X hat

            Returns
            -------
            `numpy.ndarray`
                The original X matrix of the spectra. If X, set_params and set_attributes have given for parameters.
            or
            `Dataset`
                A Dataset object.
        """
        if X.size > 0 and len(self.get_scale_X_props) > 0:
            return Dataset.unscale_X(X, method, self.get_scale_X_props["params"], self.get_scale_X_props["attributes"])
        elif len(self.get_scale_X_props) > 0:
            self.X = Dataset.unscale_X(self.X, method, self.get_scale_X_props["params"], self.get_scale_X_props["attributes"])
        else:
            self.X = Dataset.unscale_X(self.X, method, set_params, set_attributes)
        return self

    def unscale_X(
        X: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        """
           Static unscale for X matrix of the spectra with Scale method.

           Parameters
           ----------

           X: `numpy.ndarray`
                   A 2D or 3D scaled matrix of the spectra.

           method: `Scale`
                   The method is used to scale X

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `numpy.ndarray`
               The original X matrix of the spectra.
       """
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
                    scaler[i] = Dataset.__set_scale_attributes(method, scaler[i], set_attributes[i])
                    X[:, :, i] = scaler[i].inverse_transform(X[:, :, i])
                elif method == Scale.MINMAX:
                    scaler[i] = Dataset.__set_scale_attributes(method, scaler[i], set_attributes[i])
                    X[:, :, i] = scaler[i].inverse_transform(X[:, :, i])
        else:
            if method == Scale.STANDARD:
                scaler = StandardScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
                X = scaler.inverse_transform(X)
            elif method == Scale.MINMAX:
                scaler = MinMaxScaler()
                if len(set_params) != 0:
                    scaler = scaler.set_params(**set_params[0])
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
                X = scaler.inverse_transform(X)
        return X

    def apply_scale_X(self, method: Scale, set_params: List = [], set_attributes: List = []):
        """
           Scale X matrix of the spectra with Scale method.

           Parameters
           ----------

           method: `Scale`
                   The method used is to scale 2D or 3D X matrix of the spectra.

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `Dataset`
                A Dataset object.
       """
        self.X, self.get_scale_X_props = Dataset.scale_X(
            self.X, method, set_params, set_attributes
        )
        return self

    def scale_X(
        X: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        """
           Static scale method of X matrix of the spectra with Scale method.

           Parameters
           ----------

           X: `numpy.ndarray`
                   A 2D or 3D matrix of the spectra.

           method: `Scale`
                   The method is used to scale X

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `Dataset`
                A Dataset object.
       """
        if X.ndim == 3:
            if method == Scale.STANDARD:
                scaler = [StandardScaler() for _ in range(X.shape[2])]
            elif method == Scale.MINMAX:
                scaler = [MinMaxScaler() for _ in range(X.shape[2])]
            get_params = []
            get_attributes = []
            for i in range(X.shape[2]):
                X[:, :, i], params, attributes = Dataset.__scale_X(X[:, :, i], method, scaler[i], set_params[i] if len(set_params) else {}, set_attributes[i] if len(set_attributes) else {})
                get_params.append(params)
                get_attributes.append(attributes)
        else:
            if method == Scale.STANDARD:
                scaler = StandardScaler()
            elif method == Scale.MINMAX:
                scaler = MinMaxScaler()
            X, params, attributes = Dataset.__scale_X(X, method, scaler, set_params[0] if len(set_params) else {}, set_attributes[0] if len(set_attributes) else {})
            get_params = [params]
            get_attributes = [attributes]
        return X, {"params": get_params, "attributes": get_attributes}

    def apply_unscale_Y(
        self, method: Scale, set_params: List = [], set_attributes: List = [], Y: np.ndarray = np.array([])
    ):
        """
            Unscale a 1D vector or 2D matrix of the output property(ies) with Scale method.

            Parameters
            ----------

            method: `Scale`
                    The method is used to scale Y

            set_params: `List`
                    A list of dicts with the parameters of each Scale method.

            set_attributes: `List`
                    A list of dicts with the attributes of each Scale method.

            Y: `numpy.ndarray`
                    A 1D vector or 2D matrix of the output property(ies).

            Returns
            -------
            `numpy.ndarray`
                The original 1D or 2D matrix Y. If Y, set_params and set_attributes have given for parameters.
            or
            `Dataset`
                A Dataset object.
        """
        if Y.size > 0 and len(self.get_scale_Y_props) > 0:
            return Dataset.unscale_Y(Y, method, self.get_scale_Y_props["params"], self.get_scale_Y_props["attributes"])
        elif len(self.get_scale_Y_props) > 0:
            self.Y = Dataset.unscale_Y(self.Y, method, self.get_scale_Y_props["params"], self.get_scale_Y_props["attributes"])
        else:
            self.Y = Dataset.unscale_Y(self.Y, method, set_params, set_attributes)
        return self

    def unscale_Y(
        Y: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        """
           Static unscale method for a 1D vector or 2D matrix of the output property(ies) with Scale method.

           Parameters
           ----------

           Y: `numpy.ndarray`
                   A scaled 1D vector or 2D matrix of the output property(ies)

           method: `Scale`
                   The method is used to scale Y

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `numpy.ndarray`
               The original 1D or 2D matrix Y
       """
        if len(set_attributes) == 0:
            raise AssertionError("You need to specify set_attributes")
        if method == Scale.STANDARD:
            scaler = StandardScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
            Y = scaler.inverse_transform(Y)
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
            Y = scaler.inverse_transform(Y)
        return Y

    def apply_scale_Y(self, method: Scale, set_params: List = [], set_attributes: List = []):
        """
           Scale a 1D vector or 2D matrix of the output property(ies) with Scale method.

           Parameters
           ----------

           method: `Scale`
                   The method is used to scale Y

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `Dataset`
                A Dataset object.
       """
        self.Y, self.get_scale_Y_props = Dataset.scale_Y(
            self.Y, method, set_params, set_attributes
        )
        return self

    def scale_Y(
        Y: np.ndarray, method: Scale, set_params: List = [], set_attributes: List = []
    ):
        """
           Static scale method for a 1D vector or 2D matrix of the output property(ies) with Scale method.

           Parameters
           ----------

           Y: `numpy.ndarray`
                   A scaled 1D vector or 2D matrix of the output property(ies).

           method: `Scale`
                   The method is used to scale Y

           set_params: `List`
                   A list of dicts with the parameters of each Scale method.

           set_attributes: `List`
                   A list of dicts with the attributes of each Scale method.


           Returns
           -------
           `Dataset`
                A Dataset object.
       """
        if method == Scale.STANDARD:
            scaler = StandardScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            if len(set_attributes) != 0:
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
            else:
                scaler = scaler.fit(Y)
            Y = scaler.transform(Y)
            get_attributes = [Dataset.__get_scale_attributes(method, scaler)]
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
            if len(set_params) != 0:
                scaler = scaler.set_params(**set_params[0])
            if len(set_attributes) != 0:
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes[0])
            else:
                scaler = scaler.fit(Y)
            Y = scaler.transform(Y)
            get_attributes = [Dataset.__get_scale_attributes(method, scaler)]
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

    def __set_scale_attributes(method: Scale, scaler: Any, set_attributes: Dict):
        if method == Scale.STANDARD:
            scaler.scale_ = set_attributes["scale_"]
            scaler.mean_ = set_attributes["mean_"]
            scaler.var_ = set_attributes["var_"]
            scaler.n_samples_seen_ = set_attributes["n_samples_seen_"]
        elif method == Scale.MINMAX:
            scaler.min_ = set_attributes["min_"]
            scaler.scale_ = set_attributes["scale_"]
            scaler.data_min_ = set_attributes["data_min_"]
            scaler.data_max_ = set_attributes["data_max_"]
            scaler.data_range_ = set_attributes["data_range_"]
        return scaler

    def __get_scale_attributes(method: Scale, scaler: Any):
        if method == Scale.STANDARD:
            return {
                "scale_": scaler.scale_,
                "mean_": scaler.mean_,
                "var_": scaler.var_,
                "n_samples_seen_": scaler.n_samples_seen_
            }
        elif method == Scale.MINMAX:
            return {
                "min_": scaler.min_,
                "scale_": scaler.scale_,
                "data_min_": scaler.data_min_,
                "data_max_": scaler.data_max_,
                "data_range_": scaler.data_range_
            }

    def __scale_X(X: np.ndarray, method: Scale, scaler: Any, set_params: Dict, set_attributes: Dict):
        if len(set_params) != 0:
            scaler = scaler.set_params(**set_params)
        get_params = scaler.get_params()
        if method == Scale.STANDARD:
            if len(set_attributes) != 0:
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes)
            else:
                scaler = scaler.fit(X)
            X = scaler.transform(X)
            get_attributes = Dataset.__get_scale_attributes(method, scaler)
        elif method == Scale.MINMAX:
            if len(set_attributes) != 0:
                scaler = Dataset.__set_scale_attributes(method, scaler, set_attributes)
            else:
                scaler = scaler.fit(X)
            X = scaler.transform(X)
            get_attributes = Dataset.__get_scale_attributes(method, scaler)
        return X, get_params, get_attributes


class Model(str, Enum):
    """A model class to describe commonly used ML models for spectral processing """

    PLS = "Partial Least Squares"
    SVR = "Support Vector Regression"
    RF = "Random Forest"

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

        # # Scale the data for SVR
        # if self.model == Model.SVR:
        #     scaler = StandardScaler()
        #     scaler.fit(X_train)
        #     X_train = scaler.transform(X_train)
        #     X_test = scaler.transform(X_test)
        #     ymean, ystd = np.mean(y_train), np.std(y_train)
        #     y_train = (y_train - ymean) / ystd

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
                if not "n_components" in self.grid_search_hyperparameters:
                    self.grid_search_hyperparameters["n_components"] = np.arange(
                        1, min(100, X.shape[1]), 1
                    )
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
        res["TrainingTime"] = trn_t1 - trn_t0
        res["TestingTime"] = tst_t1 - tst_t0
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
