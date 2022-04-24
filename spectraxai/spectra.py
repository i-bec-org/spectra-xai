import ast
import numpy as np
from enum import Enum
from scipy.signal import savgol_filter
from typing import Union, Tuple, Dict, List

from spectraxai.utils.continuumRemoval import continuum_removal


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
        """Initialize a SpectralPreprocessing object from its string representation"""
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
""" Options passed to pre-processing (e.g. window_length for SG) """
SpectralPreprocessingSequence = List[
    Union[SpectralPreprocessingOptions, List[SpectralPreprocessingOptions]]
]
""" A sequence of one or more spectral pre-treatments (e.g. SG1 + SNV) """


class Spectra:
    """
    Spectra class to hold a 2-D spectral matrix.

    Can accept a 1-D vector but always returns a 2-D matrix.
    """

    def __init__(self, X: np.ndarray) -> None:
        """X is a np 2D array containing the (sample, wavelengths) matrix"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise AssertionError("X should a 2-D matrix")
        self.X = X

    def reflectance(self) -> np.ndarray:
        """Transform absorbance to reflectance"""
        return Spectra(-1 * self.X**10)

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
            raise RuntimeError
