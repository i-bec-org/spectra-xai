# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.dataset` includes classes and operations on datasets.

This module defines the class `Dataset` which ought to be used to handle all
necessary dataset transformations including scaling, spectral pre-processing,
and splitting. The classes defined in `spectraxai.models` and
`spectraxai.explain` generally expect as inputs a `Dataset` object to operate
on.

"""

from enum import Enum
from collections.abc import Sequence
from typing import Any, overload

import pandas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y
import kennard_stone

from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.spectra import Spectra, SpectralPreprocessingSequence

ScaleResult = (
    np.ndarray
    | tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
)
"""Possible results from :meth:`Scale.apply`.

The result contains the transformed array alone when both scaler parameters and
attributes are supplied, a two-item tuple when only one is supplied, or a three-
item tuple containing the transformed array, parameters, and attributes when
neither is supplied.

"""


class Scale(str, Enum):
    """Scaling of an input feature (or of the output).

    Uses the `sklearn.preprocessing` library under the hood.

    """

    STANDARD = "standard"
    """Standard scaling, i.e. removing the mean and scaling to unit variance."""

    MINMAX = "min-max"
    """Scale and translate so that the range is between zero and one."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value

    @staticmethod
    @overload
    def apply(
        X: np.ndarray, method: "Scale"
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]: ...

    @staticmethod
    @overload
    def apply(
        X: np.ndarray,
        method: "Scale",
        set_params: dict[str, Any],
        set_attributes: dict[str, Any],
    ) -> ScaleResult: ...

    @staticmethod
    @overload
    def apply(
        X: np.ndarray,
        method: "Scale",
        set_params: dict[str, Any],
    ) -> (
        tuple[np.ndarray, dict[str, Any]]
        | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
    ): ...

    @staticmethod
    @overload
    def apply(
        X: np.ndarray,
        method: "Scale",
        set_params: dict[str, Any] | None,
        set_attributes: dict[str, Any],
    ) -> (
        tuple[np.ndarray, dict[str, Any]]
        | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
    ): ...

    @staticmethod
    def apply(
        X: np.ndarray,
        method: "Scale",
        set_params: dict[str, Any] | None = None,
        set_attributes: dict[str, Any] | None = None,
    ) -> ScaleResult:
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
        set_params = {} if set_params is None else set_params
        set_attributes = {} if set_attributes is None else set_attributes
        scaler = Scale.__new_scaler(method)
        if len(set_params) != 0:
            scaler = scaler.set_params(**set_params)
        if len(set_attributes) != 0:
            scaler = Scale.__set_scale_attributes(method, scaler, set_attributes)
        else:
            scaler = scaler.fit(X)
        X = np.asarray(scaler.transform(X))
        if len(set_params) == 0 and len(set_attributes) == 0:
            return X, scaler.get_params(), Scale.__get_scale_attributes(method, scaler)
        elif len(set_params) != 0 and len(set_attributes) != 0:
            return X
        elif len(set_attributes) != 0:
            return X, scaler.get_params()
        elif len(set_params) != 0:
            return X, Scale.__get_scale_attributes(method, scaler)
        raise RuntimeError("Invalid scaler configuration")

    @staticmethod
    def inverse(
        X: np.ndarray,
        method: "Scale",
        set_params: dict[str, Any] | None = None,
        set_attributes: dict[str, Any] | None = None,
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
        set_params = {} if set_params is None else set_params
        if not set_attributes:
            raise SpectraXAIValueError("You need to specify set_attributes")
        scaler = Scale.__new_scaler(method)
        if len(set_params) != 0:
            scaler = scaler.set_params(**set_params)
        scaler = Scale.__set_scale_attributes(method, scaler, set_attributes)
        X = scaler.inverse_transform(X)
        return X

    @staticmethod
    def __new_scaler(method: "Scale") -> StandardScaler | MinMaxScaler:
        if method == Scale.STANDARD:
            return StandardScaler()
        elif method == Scale.MINMAX:
            return MinMaxScaler()
        raise SpectraXAIValueError(f"Unknown scaling method: {method!r}")

    @staticmethod
    def __set_scale_attributes(
        method: "Scale", scaler: Any, set_attributes: dict[str, Any]
    ) -> Any:
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
    def __get_scale_attributes(method: "Scale", scaler: Any) -> dict[str, Any]:
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value


def _clhs_objective(
    data: np.ndarray, strata: np.ndarray, corr: np.ndarray, idx: np.ndarray, n: int
) -> float:
    """cLHS objective: strata coverage error plus correlation error."""
    counts = np.stack(
        [np.bincount(strata[idx, j], minlength=n) for j in range(data.shape[1])],
        axis=1,
    )
    objective = float(np.abs(counts - 1).sum())
    if data.shape[1] > 1:
        objective += float(np.abs(np.corrcoef(data[idx], rowvar=False) - corr).sum())
    return objective


def _clhs(
    data: np.ndarray, n: int, iterations: int, rng: np.random.Generator
) -> np.ndarray:
    """Conditioned Latin hypercube sampling (Minasny & McBratney, 2006).

    Selects ``n`` rows of ``data`` (samples x continuous variables) so that each
    variable's ``n`` quantile strata hold about one selected sample each and the
    correlations between variables are preserved, using simulated annealing with
    random swaps between selected and unselected samples.

    Returns
    -------
    `np.ndarray`
        The sorted indices of the selected samples.

    """
    n_samples, n_vars = data.shape
    edges = np.quantile(data, np.linspace(0, 1, n + 1), axis=0)
    strata = np.stack(
        [
            np.clip(np.searchsorted(edges[1:-1, j], data[:, j], side="right"), 0, n - 1)
            for j in range(n_vars)
        ],
        axis=1,
    )
    corr = np.corrcoef(data, rowvar=False) if n_vars > 1 else np.ones((1, 1))
    order = rng.permutation(n_samples)
    selected, unselected = order[:n].copy(), order[n:].copy()
    objective = _clhs_objective(data, strata, corr, selected, n)
    best_objective, best = objective, selected.copy()
    temperature, cooling = 1.0, 0.95
    for iteration in range(iterations):
        i, j = rng.integers(n), rng.integers(n_samples - n)
        selected[i], unselected[j] = unselected[j], selected[i]
        candidate = _clhs_objective(data, strata, corr, selected, n)
        delta = candidate - objective
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            objective = candidate
            if objective < best_objective:
                best_objective, best = objective, selected.copy()
        else:  # undo the swap
            selected[i], unselected[j] = unselected[j], selected[i]
        if (iteration + 1) % 10 == 0:
            temperature *= cooling
    return np.sort(best)


_UNITS = {
    "nm": "nm",
    "um": "um",
    "µm": "um",
    "micron": "um",
    "cm-1": "cm-1",
    "cm^-1": "cm-1",
    "cm⁻¹": "cm-1",
    "1/cm": "cm-1",
}


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    key = str(unit).strip().lower().replace(" ", "")
    if key not in _UNITS:
        raise SpectraXAIValueError(
            f"Unknown spectral unit {unit!r}; use 'nm', 'um' or 'cm-1'"
        )
    return _UNITS[key]


DataSplit = tuple[
    np.ndarray,
    np.ndarray,
]
"""A tuple representing a split for the `Dataset` into training and testing indices (in
this order)"""


class Dataset:
    """A general class to manage the dataset (i.e. input X and output Y).

    Use this class to pass your 2D spectral matrix and 1D or 2D output
    properties. Supports methods for pre-processing X, scaling X and Y,
    splitting the dataset, and more.

    """

    n_samples: int
    """Number of samples in the dataset."""

    n_features: int
    """Number of features in the dataset's input."""

    n_outputs: int
    """Number of outputs in the dataset's output."""

    X: np.ndarray
    """The spectral data of size (`n_samples`, `n_features`)"""

    Y: np.ndarray
    """The output properties of size (`n_samples`, `n_outputs`)"""

    X_names: list[str | int | float]
    """The names of the input features of size (`n_features`)"""

    Y_names: list[str]
    """The names of the output properties of size (`n_outputs`)"""

    spectral_unit: str | None
    """The unit of `wavelengths`: ``"nm"``, ``"um"``, ``"cm-1"`` or None."""

    def __init__(
        self,
        X: np.ndarray | pandas.DataFrame,
        Y: np.ndarray | pandas.DataFrame | pandas.Series,
        X_names: Sequence[str | int | float] | np.ndarray | None = None,
        Y_names: Sequence[str] | np.ndarray | None = None,
        *,
        wavelengths: Sequence[float] | np.ndarray | None = None,
        spectral_unit: str | None = None,
    ) -> None:
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

        wavelengths: sequence of `float`, optional
            The wavelength of each input feature, of length `n_features`. If
            missing, `wavelengths` is derived from `X_names` when these are numeric.
            For mid-infrared data these are wavenumbers (see ``spectral_unit``).

        spectral_unit: `str`, optional
            The unit of `wavelengths`: ``"nm"``, ``"um"`` or ``"cm-1"`` (also
            accepted: ``"µm"``, ``"cm^-1"``, ``"1/cm"``). With ``"cm-1"``
            (wavenumbers, e.g. MIR spectra) plots label the axis "Wavenumber
            (cm⁻¹)" and draw it from high to low, as is customary, whatever the
            order of the columns. Defaults to None (unknown; axes are labelled
            "Wavelength").

        """
        if isinstance(X, pandas.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pandas.DataFrame | pandas.Series):
            Y = Y.to_numpy()
        if X.shape[0] != Y.shape[0]:
            raise SpectraXAIValueError("X and Y don't have the same number of rows!")
        if X.ndim != 2:
            raise SpectraXAIValueError("X should have exactly two dimensions")
        self.X = X
        self.Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)
        check_X_y(self.X, self.Y, multi_output=True)
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_outputs = self.Y.shape[1]
        x_names = [] if X_names is None else list(X_names)
        y_names = [] if Y_names is None else list(Y_names)
        if x_names and len(x_names) != self.n_features:
            raise SpectraXAIValueError("X_names should have the same n_features as X")
        if y_names and len(y_names) != self.n_outputs:
            raise SpectraXAIValueError("Y_names should have the same n_outputs as Y")
        self.X_names = x_names or ["X" + str(i + 1) for i in range(self.n_features)]
        self.Y_names = y_names or ["Y" + str(i + 1) for i in range(self.n_outputs)]
        self.spectral_unit = _normalize_unit(spectral_unit)
        self._wavelengths = None
        if wavelengths is not None:
            wvs = np.asarray(wavelengths, dtype=float).ravel()
            if wvs.size != self.n_features or not np.isfinite(wvs).all():
                raise SpectraXAIValueError(
                    "wavelengths should contain n_features finite values"
                )
            self._wavelengths = wvs

    @property
    def wavelengths(self) -> np.ndarray | None:
        """The wavelength of each input feature, or None if unknown.

        Either the `wavelengths` passed to the constructor, or `X_names` converted to
        floats when they are all numeric (e.g. ``[350, 370, ...]`` or
        ``["350", "370", ...]``).

        """
        if self._wavelengths is not None:
            return self._wavelengths.copy()
        try:
            wvs = np.asarray(self.X_names, dtype=float)
        except (TypeError, ValueError):
            return None
        return wvs if np.isfinite(wvs).all() else None

    @property
    def is_wavenumber(self) -> bool:
        """Whether the spectral axis is in wavenumbers (``spectral_unit="cm-1"``)."""
        return self.spectral_unit == "cm-1"

    @property
    def spectral_label(self) -> str:
        """Axis label of the spectral dimension, e.g. ``"Wavenumber (cm$^{-1}$)"``."""
        return {
            None: "Wavelength",
            "nm": "Wavelength (nm)",
            "um": "Wavelength (µm)",
            "cm-1": "Wavenumber (cm$^{-1}$)",
        }[self.spectral_unit]

    def _like(self, X: np.ndarray, Y: np.ndarray) -> "Dataset":
        """Return a new Dataset with the same names, wavelengths and unit."""
        return Dataset(
            X,
            Y,
            self.X_names,
            self.Y_names,
            wavelengths=self._wavelengths,
            spectral_unit=self.spectral_unit,
        )

    def to_dataframe(self) -> pandas.DataFrame:
        """Return the dataset as a single `pandas.DataFrame`.

        Columns are the input features (named by `X_names`, as strings) followed by
        the outputs (named by `Y_names`), e.g. to pass to tools such as pycaret.

        """
        columns = [str(name) for name in self.X_names] + list(self.Y_names)
        if len(set(columns)) != len(columns):
            raise SpectraXAIValueError("X_names and Y_names should be unique")
        return pandas.DataFrame(np.hstack((self.X, self.Y)), columns=columns)

    def _repair_split_array(self, arr: Sequence[np.ndarray]) -> np.ndarray:
        result = np.array(arr, dtype=object)
        if result[0].dtype == np.dtype("O"):
            result = result.astype(np.int64)
        return result

    def train_test_split(
        self,
        split: DatasetSplit,
        opt: int | float,
        *,
        random_state: int | None = None,
        n_bins: int = 10,
        clhs_iterations: int = 5000,
        clhs_components: int = 5,
    ) -> DataSplit:
        """Split dataset with passed split method to train and test.

        Parameters
        ----------
        split: `DatasetSplit`
            The method used to split the dataset

        opt: `Number`
            A float number (between 0 and 1) indicating the percentage of the
            training dataset for Random, Kennard–Stone and cLHS split. A natural
            number for Cross Validation and Stratified split.

        random_state: `int`, optional
            Seed for the RANDOM, CLHS, CROSS_VALIDATION and STRATIFIED splits.
            Defaults to None (non-reproducible, as in earlier versions).

        n_bins: `int`, optional
            For the STRATIFIED split of a continuous output: the number of quantile
            bins of the (first) output used as strata. Class-like outputs (binary
            or multiclass) are stratified on their labels directly. Defaults to 10.

        clhs_iterations: `int`, optional
            For the CLHS split: the number of simulated-annealing iterations.
            Defaults to 5000.

        clhs_components: `int`, optional
            For the CLHS split: the number of principal components of the
            (standardized) spectra that the sample should cover. Defaults to 5.

        Returns
        -------
        `DataSplit`
            The idx_trn, idx_tst tuple. Use e.g. X[idx_trn], Y[idx_trn] to get the
            training dataset.

        """
        indices = np.arange(self.X.shape[0])
        if split in (
            DatasetSplit.RANDOM,
            DatasetSplit.KENNARD_STONE,
            DatasetSplit.CLHS,
        ):
            msg = (
                "opt param should be a float in the (0, 1) range for "
                "RANDOM, KENNARD_STONE and CLHS splits"
            )
            if not isinstance(opt, float):
                raise TypeError(msg)
            if opt <= 0 or opt >= 1:
                raise SpectraXAIValueError(msg)
        elif split == DatasetSplit.CROSS_VALIDATION or split == DatasetSplit.STRATIFIED:
            msg = (
                "opt param should be a positive int greater than 1 for "
                "CROSS_VALIDATION and STRATIFIED splits"
            )
            if not isinstance(opt, int):
                raise TypeError(msg)
            if opt <= 1:
                raise SpectraXAIValueError(msg)
        if split == DatasetSplit.RANDOM:
            idx_trn, idx_tst = train_test_split(
                indices, train_size=opt, random_state=random_state
            )
            return np.asarray(idx_trn), np.asarray(idx_tst)
        elif split == DatasetSplit.KENNARD_STONE:
            _, _, idx_trn, idx_tst = kennard_stone.train_test_split(
                self.X, indices, test_size=(1 - opt)
            )
            return idx_trn, idx_tst
        elif split == DatasetSplit.CLHS:
            n_select = int(round(opt * self.n_samples))
            if not 1 <= n_select < self.n_samples:
                raise SpectraXAIValueError("opt leaves an empty training or test set")
            n_components = max(
                1, min(clhs_components, self.n_features, self.n_samples - 1)
            )
            scores = PCA(
                n_components=n_components, random_state=random_state
            ).fit_transform(StandardScaler().fit_transform(self.X))
            idx_trn = _clhs(
                scores, n_select, clhs_iterations, np.random.default_rng(random_state)
            )
            idx_tst = np.setdiff1d(indices, idx_trn)
            return idx_trn, idx_tst
        elif split == DatasetSplit.CROSS_VALIDATION:
            assert isinstance(opt, int)
            kf = KFold(opt, shuffle=True, random_state=random_state)
            idx_trn, idx_tst = [], []
            for trn_index, tst_index in kf.split(self.X):
                idx_trn.append(np.array(trn_index))
                idx_tst.append(np.array(tst_index))
            return self._repair_split_array(idx_trn), self._repair_split_array(idx_tst)
        elif split == DatasetSplit.STRATIFIED:
            assert isinstance(opt, int)
            if random_state is None:
                skf = StratifiedKFold(n_splits=opt)
            else:
                skf = StratifiedKFold(
                    n_splits=opt, shuffle=True, random_state=random_state
                )
            idx_trn, idx_tst = [], []
            for trn_index, tst_index in skf.split(self.X, self._strata(n_bins)):
                idx_trn.append(np.array(trn_index))
                idx_tst.append(np.array(tst_index))
            return self._repair_split_array(idx_trn), self._repair_split_array(idx_tst)

    def _strata(self, n_bins: int) -> np.ndarray:
        """Labels to stratify on: class labels, or quantile bins of output 1."""
        y = self.Y[:, 0] if self.n_outputs > 1 else self.Y.ravel()
        if type_of_target(y) in ("binary", "multiclass"):
            return y
        if n_bins < 2:
            raise SpectraXAIValueError("n_bins should be at least 2")
        return np.asarray(
            pandas.qcut(y, q=n_bins, labels=False, duplicates="drop"), dtype=int
        )

    def train_test_split_explicit(
        self, trn: np.ndarray | None = None, tst: np.ndarray | None = None
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
        trn = np.array([]) if trn is None else trn
        tst = np.array([]) if tst is None else tst
        if tst.size == 0 and trn.size == 0:
            raise SpectraXAIValueError("You need to specify either tst or trn indices")
        if tst.size > 0 and trn.size > 0:
            raise SpectraXAIValueError("You cannot specify both trn and tst")
        train_indices: np.ndarray | list[np.ndarray] = trn
        test_indices: np.ndarray | list[np.ndarray] = tst
        if tst.size > 0 and isinstance(tst[0], np.ndarray):
            train_indices = [
                self.train_test_split_explicit(tst=fold)[0] for fold in tst
            ]
        elif tst.size > 0:
            if not np.logical_and(tst >= 0, tst < self.n_samples).all():
                raise SpectraXAIValueError("Passed indices contain out of bound values")
            train_indices = np.array(
                list(set(range(0, self.n_samples)).difference(set(tst)))
            )
        elif trn.size > 0 and isinstance(trn[0], np.ndarray):
            test_indices = [self.train_test_split_explicit(trn=fold)[1] for fold in trn]
        else:
            if not np.logical_and(trn >= 0, trn < self.n_samples).all():
                raise SpectraXAIValueError("Passed indices contain out of bound values")
            test_indices = np.array(
                list(set(range(0, self.n_samples)).difference(set(trn)))
            )
        trn, tst = np.array(train_indices), np.array(test_indices)
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
            raise SpectraXAIValueError("Passed indices for subsetting cannot be empty")
        if idx.size > 0 and isinstance(idx[0], np.ndarray):
            raise SpectraXAIValueError(
                "Expected indices array to contain only one dimension"
            )
        if not np.logical_and(idx >= 0, idx < self.n_samples).all():
            raise SpectraXAIValueError("Indices contain out of bound values")
        return self._like(self.X[idx], self.Y[idx])

    def select_features(self, features: np.ndarray | Sequence[int]) -> "Dataset":
        """Keep only some input features (columns), e.g. after feature selection.

        Parameters
        ----------
        features: `np.ndarray`
            Either a boolean mask of length `n_features` or the integer indices of
            the features to keep (in the given order).

        Returns
        -------
        `Dataset`
            A new Dataset with the selected columns of X and the matching
            `X_names` and `wavelengths`.

        """
        features = np.asarray(features)
        if features.dtype == bool:
            if features.shape != (self.n_features,):
                raise SpectraXAIValueError(
                    "A boolean mask should have one value per input feature"
                )
            idx = np.flatnonzero(features)
        else:
            idx = features.astype(int).ravel()
            if not np.logical_and(idx >= 0, idx < self.n_features).all():
                raise SpectraXAIValueError(
                    "Feature indices contain out of bound values"
                )
        if idx.size == 0:
            raise SpectraXAIValueError("At least one feature should be selected")
        wavelengths = self.wavelengths
        return Dataset(
            self.X[:, idx],
            self.Y,
            [self.X_names[i] for i in idx],
            self.Y_names,
            wavelengths=None if wavelengths is None else wavelengths[idx],
            spectral_unit=self.spectral_unit,
        )

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
        spectra = Spectra(self.X, wavelengths=self.wavelengths)
        return self._like(spectra.preprocess(method).X, self.Y)
