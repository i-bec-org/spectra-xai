"""The module `spectraxai.dataset` includes classes and operations on datasets.

This module defines the class `Dataset` which ought to be used to handle all
necessary dataset transformations including scaling, spectral pre-processing,
and splitting. The classes defined in `spectraxai.models` and
`spectraxai.explain` generally expect as inputs a `Dataset` object to operate
on.

"""

from enum import Enum
from numbers import Number
from typing import Tuple, Dict, List, Any

import pandas
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y
import kennard_stone

from spectraxai.spectra import Spectra, SpectralPreprocessingSequence


class Scale(str, Enum):
    """Scaling of an input feature (or of the output).

    Uses the `sklearn.preprocessing` library under the hood.

    """

    STANDARD = "standard"
    """Standard scaling, i.e. removing the mean and scaling to unit variance"""
    MINMAX = "min-max"
    """Scale and translate so that the range is between zero and one"""

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value

    @staticmethod
    def apply(
        X: np.ndarray, method: "Scale", set_params: Dict = {}, set_attributes: Dict = {}
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Apply a method to scale a given array.

        Parameters
        ----------
        X: `numpy.ndarray`
            A 2D matrix to be scaled

        method: `Scale`
            The method to use for scaling

        set_params: dict, optional
            A dictionary of parameters defined in the corresponding
            `sklearn.preprocessing` class

        set_attributes: dict, optional
            A dictionary of the attributes defined in the corresponding
            `sklearn.preprocessing` class


        Returns
        -------
        X: `np.ndarray`
            The scaled matrix

        scale_params: dict
            A dictionary of parameters defined in the corresponding
            `sklearn.preprocessing` class. Returned only if set_params
            was not passed.

        scale_attributes: dict
            A dictionary of the attributes defined in the corresponding
            `sklearn.preprocessing` class. Returned only if set_attributes
            was not passed.

        """
        if method == Scale.STANDARD:
            scaler = StandardScaler()
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
        if len(set_params) != 0:
            scaler = scaler.set_params(**set_params)
        if len(set_attributes) != 0:
            scaler = Scale.__set_scale_attributes(method, scaler, set_attributes)
        else:
            scaler = scaler.fit(X)
        X = scaler.transform(X)
        if len(set_params) == 0 and len(set_attributes) == 0:
            return X, scaler.get_params(), Scale.__get_scale_attributes(method, scaler)
        elif len(set_params) != 0 and len(set_attributes) != 0:
            return X
        elif len(set_attributes) != 0:
            return X, scaler.get_params()
        elif len(set_params) != 0:
            return X, Scale.__get_scale_attributes(method, scaler)

    def inverse(
        X: np.ndarray, method: "Scale", set_params: Dict = {}, set_attributes: Dict = {}
    ) -> np.ndarray:
        """Apply a method to un-scale a given array.

        Parameters
        ----------
        X: `numpy.ndarray`
            A 2D matrix to be un-scaled

        method: `Scale`
            The method used to scale the matrix

        set_params: dict
            A dictionary of parameters defined in the corresponding
            `sklearn.preprocessing` class

        set_attributes: dict
            A dictionary of the attributes defined in the corresponding
            `sklearn.preprocessing` class


        Returns
        -------
        `np.ndarray`
            The un-scaled matrix

        """
        if len(set_attributes) == 0:
            raise AssertionError("You need to specify set_attributes")
        if method == Scale.STANDARD:
            scaler = StandardScaler()
        elif method == Scale.MINMAX:
            scaler = MinMaxScaler()
        if len(set_params) != 0:
            scaler = scaler.set_params(**set_params)
        scaler = Scale.__set_scale_attributes(method, scaler, set_attributes)
        X = scaler.inverse_transform(X)
        return X

    @staticmethod
    def __set_scale_attributes(method: "Scale", scaler: Any, set_attributes: Dict):
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

    @staticmethod
    def __get_scale_attributes(method: "Scale", scaler: Any):
        if method == Scale.STANDARD:
            return {
                "scale_": scaler.scale_,
                "mean_": scaler.mean_,
                "var_": scaler.var_,
                "n_samples_seen_": scaler.n_samples_seen_,
            }
        elif method == Scale.MINMAX:
            return {
                "min_": scaler.min_,
                "scale_": scaler.scale_,
                "data_min_": scaler.data_min_,
                "data_max_": scaler.data_max_,
                "data_range_": scaler.data_range_,
            }


class DatasetSplit(str, Enum):
    """Types of dataset split supported by the `Dataset` class."""

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
    np.ndarray,
    np.ndarray,
]
"""A tuple representing a split for the `Dataset` into training and testing
indices (in this order)"""


class Dataset:
    """A general class to manage the dataset (i.e. input X and output Y).

    Use this class to pass your 2D spectral matrix and 1D or 2D output
    properties. Supports methods for pre-processing X, scaling X and Y,
    splitting the dataset, and more.

    """

    n_samples: int
    """Number of samples in the dataset"""
    n_features: int
    """Number of features in the dataset's input"""
    n_outputs: int
    """Number of outputs in the dataset's output"""
    X: np.ndarray
    """The spectral data of size (`n_samples`, `n_features`)"""
    Y: np.ndarray
    """The output properties of size (`n_samples`, `n_outputs`)"""
    X_names: List[str]
    """The names of the input features of size (`n_features`)"""
    Y_names: List[str]
    """The names of the output properties of size (`n_outputs`)"""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_names: List[str] = [],
        Y_names: List[str] = [],
    ):
        """Create a new Dataset from input and output arrays.

        Parameters
        ----------
        X: `numpy.ndarray` or `pandas.DataFrame`
            A 2D matrix of the spectra of size (`n_samples`, `n_features`)

        Y: `numpy.ndarray` or `pandas.DataFrame` or `pandas.Series`
            A 1D vector (`n_samples`,) or 2D matrix (`n_samples`, `n_outputs`)
            of the output property(ies). If 1D it will be implicitly converted
            to 2D.

        X_names: `list[str]`, optional
            A list of length `n_features` containing the names of the input
            features. If missing, these will be autogenerated as X1, X2, etc.

        Y_names: `list[str]`, optional
            A list of length `n_outputs` containing the names of the output
            properties. When using a single output property pass a list of
            size 1. If missing, these will be autogenerated as Y1, Y2, etc.

        """
        if X.shape[0] != Y.shape[0]:
            raise AssertionError("X and Y don't have the same number of rows!")
        if X.ndim != 2:
            raise AssertionError("X should have exactly two dimensions")
        self.X = X.to_numpy() if isinstance(X, pandas.DataFrame) else X
        if isinstance(Y, pandas.DataFrame) or isinstance(Y, pandas.Series):
            Y = Y.to_numpy()
        self.Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)
        check_X_y(X, Y, multi_output=True)
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_outputs = self.Y.shape[1]
        if X_names and len(X_names) != self.n_features:
            raise AssertionError("X_names should have the same n_features as X")
        if Y_names and len(Y_names) != self.n_outputs:
            raise AssertionError("Y_names should have the same n_outputs as Y")
        self.X_names = (
            X_names if X_names else ["X" + str(i + 1) for i in range(self.n_features)]
        )
        self.Y_names = (
            Y_names if Y_names else ["Y" + str(i + 1) for i in range(self.n_outputs)]
        )

    def _repair_split_array(self, arr):
        arr = np.array(arr, dtype=object)
        if arr[0].dtype == np.dtype("O"):
            arr = arr.astype(np.int64)
        return arr

    def train_test_split(self, split: DatasetSplit, opt: Number) -> DataSplit:
        """Split dataset with passed split method to train and test.

        Parameters
        ----------
        split: `DatasetSplit`
            The method used to split the dataset

        opt: `Number`
            A float number (between 0 and 1) indicating the percentage of the
            training dataset for Random and Kennard–Stone split. A natural number
            for Cross Validation and Stratified split.

        Returns
        -------
        `DataSplit`
            The idx_trn, idx_tst tuple. Use e.g. X[idx_trn], Y[idx_trn] to get the
            training dataset.

        """
        indices = np.arange(self.X.shape[0])
        if split == DatasetSplit.RANDOM or split == DatasetSplit.KENNARD_STONE:
            msg = (
                "opt param should be a float in the (0, 1) range for "
                "RANDOM and KENNARD_STONE splits"
            )
            if not isinstance(opt, float):
                raise TypeError(msg)
            if opt <= 0 or opt >= 1:
                raise ValueError(msg)
        elif split == DatasetSplit.CROSS_VALIDATION or split == DatasetSplit.STRATIFIED:
            msg = (
                "opt param should be a positive int greater than 1 for "
                "CROSS_VALIDATION and STRATIFIED splits"
            )
            if not isinstance(opt, int):
                raise TypeError(msg)
            if opt <= 1:
                raise ValueError(msg)
        if split == DatasetSplit.RANDOM:
            return train_test_split(indices, train_size=opt)
        elif split == DatasetSplit.KENNARD_STONE:
            _, _, idx_trn, idx_tst = kennard_stone.train_test_split(
                self.X, indices, test_size=(1 - opt)
            )
            return idx_trn, idx_tst
        elif split == DatasetSplit.CLHS:
            raise NotImplementedError("clhs not implemented yet")
        elif split == DatasetSplit.CROSS_VALIDATION:
            kf = KFold(opt, shuffle=True)
            idx_trn, idx_tst = [], []
            for trn_index, tst_index in kf.split(self.X):
                idx_trn.append(np.array(trn_index))
                idx_tst.append(np.array(tst_index))
            return self._repair_split_array(idx_trn), self._repair_split_array(idx_tst)
        elif split == DatasetSplit.STRATIFIED:
            skf = StratifiedKFold(n_splits=opt)
            idx_trn, idx_tst = [], []
            for trn_index, tst_index in skf.split(self.X, self.Y):
                idx_trn.append(np.array(trn_index))
                idx_tst.append(np.array(tst_index))
            return self._repair_split_array(idx_trn), self._repair_split_array(idx_tst)
        else:
            raise RuntimeError("Not a valid split method!")

    def train_test_split_explicit(
        self, trn: np.ndarray = np.array([]), tst: np.ndarray = np.array([])
    ) -> DataSplit:
        """Split dataset to train and test from pre-selected by the user indices.

        This is useful in cases where one has the set containing the training
        indices of a dataset and wants to get the complement of the training set to
        obtain the test set.

        Parameters
        ----------
        trn: `np.ndarray`, optional
            Contains the indices of the training samples

        tst: `np.ndarray`, optional
            Contains the indices of the testing samples

        Returns
        -------
        `DataSplit`
            The idx_trn, idx_tst tuple

        """
        if tst.size == 0 and trn.size == 0:
            raise AssertionError("You need to specify either tst or trn indices")
        if tst.size > 0 and trn.size > 0:
            raise AssertionError("You cannot specify both trn and tst")
        if tst.size > 0 and isinstance(tst[0], np.ndarray):
            trn = [self.train_test_split_explicit(tst=fold)[0] for fold in tst]
        elif tst.size > 0:
            if not np.logical_and(tst >= 0, tst <= self.n_samples).all():
                raise AssertionError("Passed indices contain out of bound values")
            trn = list(set(range(0, self.n_samples)).difference(set(tst)))
        elif trn.size > 0 and isinstance(trn[0], np.ndarray):
            tst = [self.train_test_split_explicit(trn=fold)[1] for fold in trn]
        else:
            if not np.logical_and(trn >= 0, trn <= self.n_samples).all():
                raise AssertionError("Passed indices contain out of bound values")
            tst = list(set(range(0, self.n_samples)).difference(set(trn)))
        trn, tst = np.array(trn), np.array(tst)
        if isinstance(trn[0], np.ndarray) and trn[0].dtype == "O":
            trn = trn.astype(np.int64)
        if isinstance(tst[0], np.ndarray) and tst[0].dtype == "O":
            tst = tst.astype(np.int64)
        return trn, tst

    def subset(self, idx: np.ndarray) -> "Dataset":
        """Subset the dataset using passed indices.

        Parameters
        ----------
        idx: np.ndarray
            The indices to subset by

        Returns
        -------
        `Dataset`
            A new Dataset populated by the passed indices

        """
        if idx.size == 0:
            raise ValueError("Passed indices for subsetting cannot be empty")
        if idx.size > 0 and isinstance(idx[0], np.ndarray):
            raise ValueError("Expected indices array to contain only one dimension")
        if not np.logical_and(idx >= 0, idx <= self.n_samples).all():
            raise ValueError("Indices contain out of bound values")
        return Dataset(self.X[idx], self.Y[idx], self.X_names, self.Y_names)

    def preprocess(self, method: SpectralPreprocessingSequence) -> "Dataset":
        """Preprocess dataset by method.

        Parameters
        ----------
        method: `spectraxai.spectra.SpectralPreprocessingSequence`
            The method for the preprocess.

        Returns
        -------
        `Dataset`
            A new Dataset object, where X contains the preprocessed spectra.

        """
        return Dataset(
            Spectra(self.X).preprocess(method).X, self.Y, self.X_names, self.Y_names
        )
