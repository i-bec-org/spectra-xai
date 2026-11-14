# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.spectra` supports operations on spectra.

This module defines the class `Spectra` which can be used to hold spectral
datasets (only the features, e.g. reflectance values) and perform common
operations on them like pre-processing.

"""

import ast
import re
from typing import Any, TypeGuard, cast
import numpy as np
from enum import Enum
from scipy.signal import savgol_filter

from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.utils import continuum_removal


def _is_numpy_array(value: object) -> TypeGuard[np.ndarray]:
    return isinstance(value, np.ndarray)


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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value

    @staticmethod
    def _lookup(token: str) -> "SpectralPreprocessing":
        """Find a member by its name (e.g. ``ABS``) or value (``absorbance``)."""
        for member in SpectralPreprocessing:
            if token in (member.name, member.value):
                return member
        raise SpectraXAIValueError(f"Unknown spectral pre-processing: {token!r}")

    @staticmethod
    def _from_literal(node: Any) -> Any:
        if isinstance(node, str):
            return SpectralPreprocessing._lookup(node)
        if isinstance(node, list):
            return [SpectralPreprocessing._from_literal(item) for item in node]
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            return (SpectralPreprocessing._lookup(node[0]), node[1])
        raise SpectraXAIValueError(f"Cannot parse spectral pre-processing: {node!r}")

    @staticmethod
    def init_class(string: str) -> Any:
        """Initialize from its string representation, i.e. the inverse of `str`.

        Accepts anything `str` produces for a `SpectralPreprocessingSequence`, such as
        the ``pre_process`` column returned by
        `spectraxai.models.StandardModel.fit_and_predict`: a single member
        (``"ABS"``), a member with options (``"(SG1, {'window_length': 7})"``) or a
        (nested) list of those (``"[absorbance, SNV]"``). Members may be written by
        name or by value.

        Returns
        -------
        `SpectralPreprocessingSequence`
            The parsed member, tuple or list (lists are returned as lists).

        """
        tokens = sorted(
            {token for m in SpectralPreprocessing for token in (m.name, m.value)},
            key=len,
            reverse=True,
        )
        # Quote bare tokens only; tokens inside quotes (e.g. dict keys or string
        # option values) are left untouched.
        pattern = re.compile(
            r"""('[^']*'|"[^"]*")|(?<![\w-])("""
            + "|".join(re.escape(t) for t in tokens)
            + r""")(?![\w-])"""
        )
        quoted = pattern.sub(
            lambda m: m.group(1) if m.group(1) else repr(m.group(2)), string
        )
        try:
            literal = ast.literal_eval(quoted)
        except (ValueError, SyntaxError) as error:
            raise SpectraXAIValueError(
                f"Cannot parse spectral pre-processing: {string!r}"
            ) from error
        return SpectralPreprocessing._from_literal(literal)


SpectralPreprocessingOptions = (
    SpectralPreprocessing | tuple[SpectralPreprocessing, dict[str, int]]
)
"""Either a single SpectralPreprocessing or a tuple specifying a SpectralPreprocessing
and its assorted options (e.g. window_length for SG).

Examples:
- SpectralPreprocessing.CR => continuum-removal
- (SpectralPreprocessing.SG2, {"window_length": 7, "polyorder": 3}) =>
    SG2 with window_length of 7 and polyorder of 3

"""

SpectralPreprocessingSequence = (
    SpectralPreprocessingOptions | list[SpectralPreprocessingOptions]
)
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
        if not _is_numpy_array(X):
            raise TypeError("X should be a numpy array")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise SpectraXAIValueError("X should be a 2-D matrix")
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

    def sg(self, **kwargs: Any) -> "Spectra":
        """Apply a general Savitzky–Golay transform.

        You need to pass as kwargs the parameters of
        `scipy.signal.savgol_filter`

        """
        return Spectra(cast(np.ndarray, savgol_filter(self.X, **kwargs)))

    def cr(self) -> "Spectra":
        """Transform absorbance spectra using the Continuum Removal."""
        return Spectra(cast(np.ndarray, continuum_removal(self.X)))

    def __preprocess(self, method: SpectralPreprocessing, **kwargs: Any) -> "Spectra":
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
        raise SpectraXAIValueError(f"Unknown spectral pre-processing: {method!r}")

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
        if isinstance(method, SpectralPreprocessing):
            return self.__preprocess(method)
        elif isinstance(method, tuple):
            return self.__preprocess(method[0], **method[1])
        elif isinstance(method, list):
            for each_method in method:
                if isinstance(each_method, SpectralPreprocessing):
                    self = self.__preprocess(each_method)
                elif isinstance(each_method, tuple):
                    self = self.__preprocess(each_method[0], **each_method[1])
                elif isinstance(each_method, list):
                    self = self.preprocess(each_method)
                else:
                    raise SpectraXAIValueError(
                        f"Unknown spectral pre-processing: {each_method!r}"
                    )
            return self
        raise SpectraXAIValueError(f"Unknown spectral pre-processing: {method!r}")
