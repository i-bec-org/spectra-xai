# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.spectra` supports operations on spectra.

This module defines the class `Spectra` which can be used to hold spectral
datasets (only the features, e.g. reflectance values) and perform common
operations on them like pre-processing.

"""

import ast
import numpy as np
from enum import Enum
from scipy.signal import savgol_filter
from typing import Union, Tuple, Dict, List

from spectraxai.utils import continuum_removal


class SpectralPreprocessing(str, Enum):
    """Spectral Preprocessing enum.

    A collection of different spectral pre-processing (or pre- treatments) that
    may be applied to a spectral matrix.

    """

    NONE = "no"
    REF = "reflectance"
    ABS = "absorbance"
    SNV = "SNV"
    SG0 = "SG0"
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
            "SG0": SpectralPreprocessing.SG0,
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
        """Initialize from its string representation."""
        lst = ast.literal_eval(
            string.replace("no", "'no'")
            .replace("reflectance", "'reflectance'")
            .replace("absorbance", "'absorbance'")
            .replace("SNV", "'SNV'")
            .replace("SG0", "'SG0'")
            .replace("SG1", "'SG1'")
            .replace("SG2", "'SG2'")
            .replace("continuum-removal", "'continuum-removal'")
        )

        return SpectralPreprocessing.__init_class(lst)


SpectralPreprocessingOptions = Union[
    SpectralPreprocessing, Tuple[SpectralPreprocessing, Dict[str, int]]
]
"""Either a single SpectralPreprocessing or a tuple specifying a SpectralPreprocessing
and its assorted options (e.g. window_length for SG).

Examples:
- SpectralPreprocessing.CR => continuum-removal
- (SpectralPreprocessing.SG2, {"window_length": 7, "polyorder": 3}) =>
    SG2 with window_length of 7 and polyorder of 3

"""
SpectralPreprocessingSequence = Union[
    SpectralPreprocessingOptions, List[SpectralPreprocessingOptions]
]
"""A sequence of one or more spectral pre-treatments applied together (e.g. SG1 + SNV)

Examples:
- SpectralPreprocessing.NONE => no pre-treatment
- [SpectralPreprocessing.ABS, SpectralPreprocessing.CR] => ABS + CR
- (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}) =>
    SG1 with window_length of 7 and polyorder of 3
- [(SpectralPreprocessing.SG1, {"window_length": 7}), SpectralPreprocessing.SNV] =>
    SG1 + SNV

"""


class Spectra:
    """Spectra class to hold a 2-D spectral matrix of shape (samples, wavelengths).

    Can accept a 1-D vector as input but always returns a 2-D matrix.

    """

    X: np.ndarray
    """A 2-D matrix representing the spectra."""

    def __init__(self, X: np.ndarray) -> None:
        """X is a np 2D array containing the (samples, wavelengths) matrix."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy array")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise AssertionError("X should be a 2-D matrix")
        self.X = X

    def reflectance(self) -> "Spectra":
        """Transform absorbance to reflectance."""
        return Spectra(1 / 10**self.X)

    def absorbance(self) -> "Spectra":
        """Transform reflectance to absorbance."""
        return Spectra(-1 * np.log10(self.X))

    def snv(self) -> "Spectra":
        """Apply the standard normal variate transform."""
        snv = np.zeros(self.X.shape)
        mu, sd = self.X.mean(axis=-1), self.X.std(axis=-1, ddof=1)
        for i in range(np.shape(self.X)[0]):
            snv[i, :] = (self.X[i, :] - mu[i]) / sd[i]
        return Spectra(snv)

    def sg(self, **kwargs) -> "Spectra":
        """Apply a general Savitzky–Golay transform.

        You need to pass as kwargs the parameters of
        `scipy.signal.savgol_filter`

        """
        return Spectra(savgol_filter(self.X, **kwargs))

    def cr(self) -> "Spectra":
        """Transform absorbance spectra using the Continuum Removal."""
        return Spectra(continuum_removal(np.squeeze(self.X)))

    def __preprocess(self, method: SpectralPreprocessing, **kwargs) -> "Spectra":
        """Apply a single transform specified by method."""
        if method == SpectralPreprocessing.REF:
            return self.reflectance()
        elif method == SpectralPreprocessing.ABS:
            return self.absorbance()
        elif method == SpectralPreprocessing.SNV:
            return self.snv()
        elif method == SpectralPreprocessing.SG0:
            return self.sg(deriv=0, **kwargs)
        elif method == SpectralPreprocessing.SG1:
            return self.sg(deriv=1, **kwargs)
        elif method == SpectralPreprocessing.SG2:
            return self.sg(deriv=2, **kwargs)
        elif method == SpectralPreprocessing.CR:
            return self.cr()
        elif method == SpectralPreprocessing.NONE:
            return self
        else:
            raise RuntimeError("Invalid method specified")

    def preprocess(self, method: SpectralPreprocessingSequence) -> "Spectra":
        """Apply a pre-processing sequence (i.e., one or more) specified by method.

        Parameters
        ----------
        method: `spectraxai.spectra.SpectralPreprocessingSequence`
            The method for the preprocess.

        Returns
        -------
        `Spectra`
            A new Spectra object, where X contains the preprocessed spectra.

        """
        if isinstance(method, str):
            return self.__preprocess(method)
        elif isinstance(method, tuple):
            return self.__preprocess(method[0], **method[1])
        elif isinstance(method, list):
            for each_method in method:
                if isinstance(each_method, str):
                    self = self.__preprocess(each_method)
                elif isinstance(each_method, tuple):
                    self = self.__preprocess(each_method[0], **each_method[1])
                elif isinstance(each_method, list):
                    self = self.preprocess(each_method)
            return self
