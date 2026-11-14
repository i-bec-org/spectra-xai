# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.preprocessing` provides scikit-learn transformers.

`SpectralPreprocessor` applies a
`spectraxai.spectra.SpectralPreprocessingSequence` inside scikit-learn pipelines
(and tools built on them, such as `GridSearchCV`, `cross_val_score` or pycaret):

```python
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.spectra import SpectralPreprocessing

pipe = make_pipeline(
    SpectralPreprocessor(
        [SpectralPreprocessing.ABS, SpectralPreprocessing.MSC]
    ),
    PLSRegression(n_components=10),
)
```

Stateful steps (MSC, mean-centering) learn their reference spectrum from the
training data at ``fit``, so test samples never influence it.

"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from spectraxai.spectra import (
    SpectralPreprocessing,
    SpectralPreprocessingSequence,
    Spectra,
    _steps,
)


class SpectralPreprocessor(TransformerMixin, BaseEstimator):
    """Apply a spectral pre-processing sequence as a scikit-learn transformer.

    Parameters
    ----------
    method: `spectraxai.spectra.SpectralPreprocessingSequence`
        The pre-processing to apply, e.g. ``SpectralPreprocessing.SNV`` or
        ``[SpectralPreprocessing.ABS, (SpectralPreprocessing.SG1,
        {"window_length": 7, "polyorder": 3})]``. Defaults to None, i.e. no
        pre-processing.

    wavelengths: array-like of `float`, optional
        The wavelength of each input column, used by continuum removal. Defaults
        to the column indices (identical results for evenly spaced wavelengths).

    Attributes
    ----------
    states_: list
        The learnt reference spectrum of each step (None for stateless steps).

    n_features_in_: int
        Number of input features seen during ``fit``.

    """

    def __init__(
        self,
        method: SpectralPreprocessingSequence | None = None,
        wavelengths: Any = None,
    ) -> None:
        self.method = method
        self.wavelengths = wavelengths

    def _steps(self) -> list[tuple[SpectralPreprocessing, dict[str, Any]]]:
        return _steps(
            SpectralPreprocessing.NONE if self.method is None else self.method
        )

    def _spectra(self, X: np.ndarray) -> Spectra:
        wvs = None if self.wavelengths is None else np.asarray(self.wavelengths)
        return Spectra(X, wavelengths=wvs)

    def _is_stateful(self) -> bool:
        return any(step.is_stateful for step, _ in self._steps())

    def fit(self, X: Any, y: Any = None) -> "SpectralPreprocessor":
        """Learn the reference spectra of stateful steps from ``X``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The (training) spectra.

        y: ignored

        Returns
        -------
        `SpectralPreprocessor`
            The fitted transformer.

        """
        spectra = self._spectra(
            np.asarray(validate_data(self, X, dtype="numeric", reset=True), dtype=float)
        )
        self.states_: list[np.ndarray | None] = []
        for step, options in self._steps():
            state = spectra.X.mean(axis=0) if step.is_stateful else None
            self.states_.append(state)
            spectra = spectra._apply(step, options, state)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply the pre-processing, using the references learnt at ``fit``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The spectra to transform.

        Returns
        -------
        `np.ndarray`
            The pre-processed spectra.

        """
        steps = self._steps()
        if hasattr(self, "states_"):
            X = np.asarray(
                validate_data(self, X, dtype="numeric", reset=False), dtype=float
            )
            states = self.states_
        elif self._is_stateful():
            check_is_fitted(self, "states_")
            raise AssertionError("unreachable")  # pragma: no cover
        else:  # stateless sequences may be used without fitting
            X = np.asarray(check_array(X, dtype="numeric"), dtype=float)
            states = [None] * len(steps)
        spectra = self._spectra(X)
        for (step, options), state in zip(steps, states):
            spectra = spectra._apply(step, options, state)
        return spectra.X

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.requires_fit = self._is_stateful()
        tags.input_tags.allow_nan = False
        return tags
