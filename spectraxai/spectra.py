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
from scipy.signal import detrend, savgol_filter

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
    MSC = "MSC"
    """Multiplicative scatter correction against a reference (mean) spectrum."""
    DETREND = "detrend"
    """Remove a linear trend from each spectrum (`scipy.signal.detrend`)."""
    MEAN_CENTER = "mean-center"
    """Subtract the mean spectrum (column-wise centering)."""

    @property
    def is_stateful(self) -> bool:
        """Whether the transform learns statistics from the (training) samples.

        `MSC` and `MEAN_CENTER` depend on the mean spectrum: `Spectra.preprocess`
        computes it on the spectra it is given, while
        `spectraxai.preprocessing.SpectralPreprocessor` learns it at ``fit`` so
        test samples never influence it.

        """
        return self in (SpectralPreprocessing.MSC, SpectralPreprocessing.MEAN_CENTER)

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


def _steps(
    method: SpectralPreprocessingSequence,
) -> list[tuple[SpectralPreprocessing, dict[str, Any]]]:
    """Flatten a (nested) pre-processing sequence into ``(method, options)`` steps."""
    if isinstance(method, SpectralPreprocessing):
        return [(method, {})]
    if (
        isinstance(method, tuple)
        and len(method) == 2
        and isinstance(method[0], SpectralPreprocessing)
        and isinstance(method[1], dict)
    ):
        return [(method[0], dict(method[1]))]
    if isinstance(method, list):
        return [step for each in method for step in _steps(each)]
    raise SpectraXAIValueError(f"Unknown spectral pre-processing: {method!r}")


class Spectra:
    """Spectra class to hold a 2-D spectral matrix of shape (samples, wavelengths).

    Can accept a 1-D vector as input but always returns a 2-D matrix.

    """

    X: np.ndarray
    """A 2-D matrix representing the spectra."""

    wavelengths: np.ndarray | None
    """The wavelength of each column of `X`, if known (used by continuum removal)."""

    def __init__(
        self, X: np.ndarray, *, wavelengths: np.ndarray | list[float] | None = None
    ) -> None:
        """X is a np 2D array containing the (samples, wavelengths) matrix.

        Optionally pass the `wavelengths` of the columns of X.

        """
        if not _is_numpy_array(X):
            raise TypeError("X should be a numpy array")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise SpectraXAIValueError("X should be a 2-D matrix")
        self.X = X
        self.wavelengths = None
        if wavelengths is not None:
            wvs = np.asarray(wavelengths, dtype=float).ravel()
            if wvs.size != X.shape[1]:
                raise SpectraXAIValueError(
                    "wavelengths should have one value per column of X"
                )
            self.wavelengths = wvs

    def _new(self, X: np.ndarray) -> "Spectra":
        return Spectra(X, wavelengths=self.wavelengths)

    def reflectance(self) -> "Spectra":
        """Transform absorbance to reflectance."""
        return self._new(1 / 10**self.X)

    def absorbance(self) -> "Spectra":
        """Transform reflectance to absorbance."""
        return self._new(-1 * np.log10(self.X))

    def snv(self) -> "Spectra":
        """Apply the standard normal variate transform."""
        snv = np.zeros(self.X.shape)
        mu, sd = self.X.mean(axis=-1), self.X.std(axis=-1, ddof=1)
        for i in range(np.shape(self.X)[0]):
            snv[i, :] = (self.X[i, :] - mu[i]) / sd[i]
        return self._new(snv)

    def sg(self, **kwargs: Any) -> "Spectra":
        """Apply a general Savitzky–Golay transform.

        You need to pass as kwargs the parameters of
        `scipy.signal.savgol_filter`

        """
        return self._new(cast(np.ndarray, savgol_filter(self.X, **kwargs)))

    def cr(self, wavelengths: np.ndarray | list[float] | None = None) -> "Spectra":
        """Transform absorbance spectra using the Continuum Removal.

        The convex hull is computed against `wavelengths` (or this object's
        `wavelengths`, or else the column indices). For evenly spaced wavelengths
        all three give identical results.

        """
        wvs = self.wavelengths if wavelengths is None else wavelengths
        wvs_list = None if wvs is None else [float(w) for w in np.ravel(wvs)]
        return self._new(cast(np.ndarray, continuum_removal(self.X, wvs_list)))

    def msc(self, reference: np.ndarray | None = None) -> "Spectra":
        """Apply multiplicative scatter correction.

        Each spectrum ``x`` is regressed on the reference ``r`` (``x ≈ a + b r``) and
        replaced by ``(x - a) / b``.

        Parameters
        ----------
        reference: `np.ndarray`, optional
            The reference spectrum. Defaults to the mean of these spectra.

        """
        ref = self.X.mean(axis=0) if reference is None else np.ravel(reference)
        if ref.size != self.X.shape[1]:
            raise SpectraXAIValueError("reference should have one value per column")
        design = np.column_stack((np.ones_like(ref), ref))
        (a, b), *_ = np.linalg.lstsq(design, self.X.T, rcond=None)
        return self._new((self.X - a[:, None]) / b[:, None])

    def detrend(self, **kwargs: Any) -> "Spectra":
        """Remove a linear trend from each spectrum.

        Extra kwargs are passed to `scipy.signal.detrend` (e.g. ``bp``).

        """
        return self._new(cast(np.ndarray, detrend(self.X, axis=-1, **kwargs)))

    def mean_center(self, mean: np.ndarray | None = None) -> "Spectra":
        """Subtract the mean spectrum (or the given ``mean``) from every spectrum."""
        mu = self.X.mean(axis=0) if mean is None else np.ravel(mean)
        if mu.size != self.X.shape[1]:
            raise SpectraXAIValueError("mean should have one value per column")
        return self._new(self.X - mu)

    def _apply(
        self,
        method: SpectralPreprocessing,
        options: dict[str, Any],
        state: np.ndarray | None = None,
    ) -> "Spectra":
        """Apply a single transform; ``state`` is the learnt reference, if any."""
        if method == SpectralPreprocessing.REF:
            return self.reflectance()
        elif method == SpectralPreprocessing.ABS:
            return self.absorbance()
        elif method == SpectralPreprocessing.SNV:
            return self.snv()
        elif method == SpectralPreprocessing.SG0:
            return self.sg(deriv=0, **options)
        elif method == SpectralPreprocessing.SG1:
            return self.sg(deriv=1, **options)
        elif method == SpectralPreprocessing.SG2:
            return self.sg(deriv=2, **options)
        elif method == SpectralPreprocessing.CR:
            return self.cr(**options)
        elif method == SpectralPreprocessing.MSC:
            return self.msc(reference=state)
        elif method == SpectralPreprocessing.DETREND:
            return self.detrend(**options)
        elif method == SpectralPreprocessing.MEAN_CENTER:
            return self.mean_center(mean=state)
        elif method == SpectralPreprocessing.NONE:
            return self
        raise SpectraXAIValueError(f"Unknown spectral pre-processing: {method!r}")

    def preprocess(self, method: SpectralPreprocessingSequence) -> "Spectra":
        """Apply a pre-processing sequence (i.e., one or more) specified by method.

        Stateful steps (`SpectralPreprocessing.MSC`,
        `SpectralPreprocessing.MEAN_CENTER`) use the mean of these spectra. To learn
        it on training data only, use
        `spectraxai.preprocessing.SpectralPreprocessor`.

        Parameters
        ----------
        method: `spectraxai.spectra.SpectralPreprocessingSequence`
            The method for the preprocess.

        Returns
        -------
        `Spectra`
            A new Spectra object, where X contains the preprocessed spectra.

        """
        spectra = self
        for step, options in _steps(method):
            spectra = spectra._apply(step, options)
        return spectra
