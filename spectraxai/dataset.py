import pandas
import numpy as np
from enum import Enum
from numbers import Number
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import spectraxai.utils.kennardStone as kennardStone
from spectraxai.spectra import Spectra, SpectralPreprocessingSequence


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
    CROSS_VALIDATION = "cross-validation"
    STRATIFIED = "stratified"
    
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

    def train_test_split(self, split: DatasetSplit, opt: Number) -> DataSplit:
        """
        Splits dataset with method split to train and test by trn percentage. 
        
        Parameters
        ----------
        
        split: `DatasetSplit`
                The method used to split the dataset
        
        opt: `Number`
                A float number (between 0 and 1) indicating the percetange of the training dataset for Random and Kennard Stone split.
                A physical number for Cross Validation split.
        
        Returns
        -------
        `DataSplit`
            The X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst tuple
        """
        indices = np.arange(self.X.shape[0])
        if split != DatasetSplit.STRATIFIED:
            if split != DatasetSplit.CROSS_VALIDATION:
                if opt <= 0 or opt >= 1:
                    raise AssertionError("opt param should be in the (0, 1) range")
            elif split == DatasetSplit.CROSS_VALIDATION and opt <= 1:
                raise AssertionError("opt param shoud be positive")
        elif split == DatasetSplit.CROSS_VALIDATION and opt <= 1:
            raise AssertionError("opt param shoud be positive")
        if split == DatasetSplit.RANDOM:
            return train_test_split(self.X, self.Y, indices, train_size=opt)
        elif split == DatasetSplit.KENNARD_STONE:
            return kennardStone.train_test_split(
                self.X, self.Y, indices, test_size=(1 - opt)
            )
        elif split == DatasetSplit.CLHS:
            raise NotImplementedError("clhs not implemented yet")
        elif split == DatasetSplit.CROSS_VALIDATION:
            kf = KFold(opt, shuffle=True)
            X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst = [], [], [], [], [], []
            for trn_index, tst_index in kf.split(self.X):
                X_trn.append(self.X[trn_index, :])
                X_tst.append(self.X[tst_index, :])
                Y_trn.append(self.Y[trn_index, :])
                Y_tst.append(self.Y[tst_index, :])
                idx_trn.append(trn_index)
                idx_tst.append(tst_index)
            return X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst
        elif split == DatasetSplit.STRATIFIED:
            skf = StratifiedKFold(n_splits=opt)
            X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst = [], [], [], [], [], []
            for trn_index, tst_index in skf.split(self.X):
                X_trn.append(self.X[trn_index, :])
                X_tst.append(self.X[tst_index, :])
                Y_trn.append(self.Y[trn_index, :])
                Y_tst.append(self.Y[tst_index, :])
                idx_trn.append(trn_index)
                idx_tst.append(tst_index)
            return X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst
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
                scaler = [[StandardScaler() for _ in range(X.shape[1])] for _ in range(X.shape[2])]
            elif method == Scale.MINMAX:
                scaler = [[MinMaxScaler() for _ in range(X.shape[1])] for _ in range(X.shape[2])]
            for i in range(X.shape[2]):
                X[:, :, i] = Dataset.__unscale_X_parser(X, method, scaler[i], set_params[i], set_attributes[i])
        else:
            if method == Scale.STANDARD:
                scaler = [StandardScaler() for _ in range(X.shape[1])]
            elif method == Scale.MINMAX:
                scaler = [MinMaxScaler() for _ in range(X.shape[1])]
            X = Dataset.__unscale_X_parser(X, method, scaler, set_params[0], set_attributes[0])
        return X

    def __unscale_X_parser(X: np.ndarray, method: Scale, scaler: Any, set_params: Dict, set_attributes: Dict):
        for i in range(X.shape[1]):
            if len(set_params) != 0:
                scaler[i] = scaler[i].set_params(**set_params[i])
            scaler[i] = Dataset.__set_scale_attributes(method, scaler[i], set_attributes[i])
            print(scaler[i], set_attributes[i])
            X[:, i] = scaler[i].inverse_transform(X[:, i].reshape(-1, 1)).flatten()
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
                scaler = [[StandardScaler() for _ in range(X.shape[1])] for _ in range(X.shape[2])]
            elif method == Scale.MINMAX:
                scaler = [[MinMaxScaler() for _ in range(X.shape[1])] for _ in range(X.shape[2])]
            get_params = []
            get_attributes = []
            for i in range(X.shape[2]):
                X[:, :, i], params, attributes = Dataset.__scale_X_parser(X[:, :, i], method, scaler[i], set_params[i] if len(set_params) else [], set_attributes[i] if len(set_attributes) else [])
                get_params.append(params)
                get_attributes.append(attributes)
        else:
            if method == Scale.STANDARD:
                scaler = [StandardScaler() for _ in range(X.shape[1])]
            elif method == Scale.MINMAX:
                scaler = [MinMaxScaler() for _ in range(X.shape[1])]
            X, params, attributes = Dataset.__scale_X_parser(X, method, scaler, set_params[0] if len(set_params) else [], set_attributes[0] if len(set_attributes) else [])
            get_params = [params]
            get_attributes = [attributes]
        return X, {"params": get_params, "attributes": get_attributes}

    def __scale_X_parser(X: np.ndarray, method: Scale, scaler: Any, set_params: Dict, set_attributes: Dict):
        get_params = []
        get_attributes = []
        for i in range(X.shape[1]):
            temp_X, params, attributes = Dataset.__scale_X(X[:, i].reshape(-1, 1), method, scaler[i], set_params[i] if len(set_params) else {}, set_attributes[i] if len(set_attributes) else {})
            X[:, i] = temp_X.flatten()
            get_params.append(params)
            get_attributes.append(attributes)
        return X, get_params, get_attributes

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
            scaler = [StandardScaler() for _ in range(Y.shape[1])]
        elif method == Scale.MINMAX:
            scaler = [MinMaxScaler() for _ in range(Y.shape[1])]
        Y = Dataset.__unscale_Y_parser(Y, method, scaler, set_params, set_attributes)
        return Y

    def __unscale_Y_parser(Y: np.ndarray, method: Scale, scaler: Any, set_params: Dict, set_attributes: Dict):
        for i in range(Y.shape[1]):
            if len(set_params) != 0:
                scaler[i] = scaler[i].set_params(**set_params[i])
            scaler[i] = Dataset.__set_scale_attributes(method, scaler[i], set_attributes[i])
            Y[:, i] = scaler[i].inverse_transform(Y[:, i].reshape(-1, 1)).flatten()
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
            scaler = [StandardScaler() for _ in range(Y.shape[1])]
        elif method == Scale.MINMAX:
            scaler = [MinMaxScaler() for _ in range(Y.shape[1])]
        Y, get_params, get_attributes = Dataset.__scale_Y_parser(Y, method, scaler, set_params, set_attributes)
        return Y, {"params": get_params, "attributes": get_attributes}

    def __scale_Y_parser(Y: np.ndarray, method: Scale, scaler: Any, set_params: Dict, set_attributes: Dict):
        get_params = []
        get_attributes = []
        for i in range(Y.shape[1]):
            if len(set_params) != 0:
                scaler[i] = scaler[i].set_params(**set_params[i])
            if len(set_attributes) != 0:
                scaler[i] = Dataset.__set_scale_attributes(method, scaler[i], set_attributes[i])
            else:
                scaler[i] = scaler[i].fit(Y[:, i].reshape(-1, 1))
            Y[:, i] = scaler[i].transform(Y[:, i].reshape(-1, 1)).flatten()
            get_params.append(scaler[i].get_params())
            get_attributes.append(Dataset.__get_scale_attributes(method, scaler[i]))
        return Y, get_params, get_attributes

    def __preprocess(self, X: np.ndarray, method: SpectralPreprocessingSequence):
        if isinstance(method, str):
            X = Spectra(X).apply(method).X
        elif isinstance(method, tuple):
            X = Spectra(X).apply(method[0], **method[1]).X
        elif isinstance(method, list):
            for each_method in method:
                if isinstance(each_method, str):
                    X = Spectra(X).apply(each_method).X
                elif isinstance(each_method, tuple):
                    X = Spectra(X).apply(each_method[0], **each_method[1]).X
                elif isinstance(each_method, list):
                    X = self.__preprocess(X, each_method)
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
