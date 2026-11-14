# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.selection` selects informative wavelengths.

It provides scikit-learn compatible feature selectors:

- `RFE` and `RFECV`: recursive feature elimination (with cross-validation to pick
  the number of features) for any model, including `spectraxai.models.StandardModel`.
  Importances come from VIP for PLS, the model's own importance for RF, XGBoost
  and Cubist, coefficients for linear models, and permutation importance on the
  training data otherwise (e.g. SVR).
- `VIPSelector`: keep the wavelengths whose PLS VIP score exceeds a threshold
  (1 by default, a common rule of thumb).
- `SelectKBest`: keep the `k` wavelengths ranked highest by a pre-hoc
  `spectraxai.explain.FeatureRanking` (correlation, mutual information,
  F-statistic).

Every selector works inside pipelines (``fit(X, y)`` / ``transform(X)``) and also
directly on a `spectraxai.dataset.Dataset`, keeping `X_names` and `wavelengths`
in sync:

```python
from spectraxai.models import Model, StandardModel
from spectraxai.selection import RFECV

selector = RFECV(StandardModel(Model.PLS), step=5, cv=5).fit(dataset)
reduced = selector.transform(dataset)  # a Dataset with the kept wavelengths
```

"""

from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import RFE as _SklearnRFE
from sklearn.feature_selection import RFECV as _SklearnRFECV
from sklearn.feature_selection import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PreHocAnalysis
from spectraxai.models import Model, StandardModel


def _native_importance(estimator: Any, n_features: int) -> np.ndarray | None:
    """A fitted model's own importance (VIP, impurity, |coef|...), if it has one."""
    if isinstance(estimator, StandardModel):
        return estimator.feature_importance()[0]
    importance = getattr(estimator, "feature_importances_", None)
    if importance is not None:
        return np.asarray(importance, dtype=float)
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        coef = np.abs(np.asarray(coef, dtype=float))
        if coef.ndim == 2:
            coef = coef.mean(axis=0 if coef.shape[1] == n_features else 1)
        if coef.size == n_features:
            return coef.ravel()
    return None


class _ImportanceEstimator(RegressorMixin, BaseEstimator):
    """Fit a regressor and expose ``feature_importances_`` for scikit-learn's RFE.

    Uses the model's own importance when it has one, and the permutation
    importance (mean decrease in R2) on the training data otherwise.

    """

    def __init__(
        self, estimator: Any, n_repeats: int = 5, random_state: int | None = None
    ) -> None:
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "_ImportanceEstimator":
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        self.estimator_ = cast(Any, clone(self.estimator)).fit(X, y)
        importance = _native_importance(self.estimator_, X.shape[1])
        self.importance_method_ = "native"
        if importance is None:
            result = cast(
                Bunch,
                permutation_importance(
                    self.estimator_,
                    X,
                    y,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                ),
            )
            importance = np.asarray(result.importances_mean)
            self.importance_method_ = "permutation"
        self.feature_importances_ = importance
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


def _split(X: Any, y: Any) -> tuple[Any, Any, Dataset | None]:
    """Accept ``fit(dataset)`` as well as ``fit(X, y)``."""
    if isinstance(X, Dataset):
        if y is not None:
            raise SpectraXAIValueError("y should not be given with a Dataset")
        Y = X.Y.ravel() if X.n_outputs == 1 else X.Y
        return X.X, Y, X
    if y is None:
        raise SpectraXAIValueError("y is required when X is not a Dataset")
    return X, y, None


class _SpectralSelector(SelectorMixin, BaseEstimator):
    """Shared behaviour: fit/transform on arrays or on a `Dataset`."""

    support_: np.ndarray
    n_features_in_: int

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X: Any, y: Any = None) -> Any:
        """Select the features.

        Parameters
        ----------
        X: `spectraxai.dataset.Dataset` or array-like of shape (n_samples, n_features)
            The dataset, or the input matrix (then pass ``y`` too).

        y: array-like, optional
            The output(s), when ``X`` is an input matrix.

        Returns
        -------
        The fitted selector (``self``).

        """
        X, y, _ = _split(X, y)
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        self._fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        """Keep the selected features of an input matrix or a `Dataset`.

        Returns
        -------
        `np.ndarray` or `spectraxai.dataset.Dataset`
            The reduced input matrix, or a new Dataset (same type as ``X``).

        """
        if isinstance(X, Dataset):
            check_is_fitted(self)
            if X.n_features != self.n_features_in_:
                raise SpectraXAIValueError(
                    f"The dataset has {X.n_features} features, but the selector "
                    f"was fitted with {self.n_features_in_}"
                )
            return X.select_features(self._get_support_mask())
        return super().transform(X)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit to the data, then keep the selected features of it."""
        return self.fit(X, y).transform(X)

    def selected_wavelengths(self, dataset: Dataset) -> np.ndarray:
        """The wavelengths (or names, if unknown) of the selected features."""
        mask = self._get_support_mask()
        if dataset.wavelengths is not None:
            return dataset.wavelengths[mask]
        return np.asarray(dataset.X_names, dtype=object)[mask]

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_


class RFE(_SpectralSelector):
    """Recursive feature elimination for spectral models.

    Repeatedly fits ``estimator`` and drops the ``step`` least important
    wavelengths until ``n_features_to_select`` remain. Wraps
    `sklearn.feature_selection.RFE`, adding importances for models without their
    own (permutation importance on the training data).

    Parameters
    ----------
    estimator: regressor
        A `spectraxai.models.StandardModel` or any scikit-learn regressor. It is
        cloned, never modified.

    n_features_to_select: `int` or `float`, optional
        Number (int) or fraction (float in (0, 1)) of features to keep. Defaults to
        None, i.e. half of them.

    step: `int` or `float`, optional
        Number (int) or fraction (float in (0, 1)) of features to remove at each
        iteration. Defaults to 1. Larger steps are much faster for full spectra.

    n_repeats: `int`, optional
        Repeats of the permutation importance, when needed. Defaults to 5.

    random_state: `int`, optional
        Seed for the permutation importance. Defaults to None.

    Attributes
    ----------
    support_: `np.ndarray`
        Boolean mask of the selected features.

    ranking_: `np.ndarray`
        The elimination ranking (1 for selected features, higher for features
        removed earlier).

    n_features_: `int`
        Number of selected features.

    importance_method_: `str`
        ``"native"`` or ``"permutation"``: where the importances came from.

    """

    selector_: Any
    """The fitted `sklearn.feature_selection.RFE` (or ``RFECV``)."""

    def __init__(
        self,
        estimator: Any,
        n_features_to_select: int | float | None = None,
        step: int | float = 1,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _wrapped(self) -> _ImportanceEstimator:
        return _ImportanceEstimator(
            self.estimator, n_repeats=self.n_repeats, random_state=self.random_state
        )

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.selector_ = _SklearnRFE(
            self._wrapped(),
            n_features_to_select=self.n_features_to_select,
            step=cast(Any, self.step),
        ).fit(X, y)
        self._copy_results()

    def _copy_results(self) -> None:
        self.support_ = self.selector_.support_
        self.ranking_ = self.selector_.ranking_
        self.n_features_ = int(self.selector_.n_features_)
        self.importance_method_ = self.selector_.estimator_.importance_method_

    @property
    def estimator_(self) -> Any:
        """The estimator fitted on the selected features."""
        check_is_fitted(self, "selector_")
        return self.selector_.estimator_.estimator_

    def predict(self, X: Any) -> np.ndarray:
        """Predict with `estimator_` from the full input (selection applied)."""
        return self.estimator_.predict(
            self.transform(X.X if isinstance(X, Dataset) else X)
        )


class RFECV(RFE):
    """Recursive feature elimination with cross-validation of the number of features.

    Like `RFE`, but the number of features is the one with the best
    cross-validated score. Wraps `sklearn.feature_selection.RFECV`.

    Parameters
    ----------
    estimator: regressor
        A `spectraxai.models.StandardModel` or any scikit-learn regressor.

    step: `int` or `float`, optional
        Number (int) or fraction (float in (0, 1)) of features to remove at each
        iteration. Defaults to 1.

    cv: `int` or cross-validation generator, optional
        The cross-validation strategy. Defaults to 5 folds.

    scoring: `str` or callable, optional
        The score to maximise. Defaults to the estimator's (R2).

    min_features_to_select: `int`, optional
        The minimum number of features to keep. Defaults to 1.

    n_repeats, random_state:
        As in `RFE`.

    n_jobs: `int`, optional
        Parallel jobs across folds. Defaults to None (1).

    Attributes
    ----------
    cv_results_: `dict`
        The cross-validation scores per number of features (see
        `sklearn.feature_selection.RFECV`).

    """

    def __init__(
        self,
        estimator: Any,
        step: int | float = 1,
        cv: Any = 5,
        scoring: Any = None,
        min_features_to_select: int = 1,
        n_repeats: int = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            estimator, step=step, n_repeats=n_repeats, random_state=random_state
        )
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.n_jobs = n_jobs

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.selector_ = _SklearnRFECV(
            self._wrapped(),
            step=cast(Any, self.step),
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features_to_select,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        self._copy_results()
        self.cv_results_ = self.selector_.cv_results_


class VIPSelector(_SpectralSelector):
    """Keep the wavelengths whose PLS VIP (variable importance in projection) is high.

    Parameters
    ----------
    threshold: `float`, optional
        Keep features with VIP above this value. Defaults to 1, the usual rule of
        thumb (the mean squared VIP is 1 by construction).

    n_components: `int`, optional
        Number of PLS components. Defaults to None, in which case it is tuned by a
        grid search with ``cv`` folds (see `spectraxai.models.StandardModel`).

    cv: `int`, optional
        Folds of the grid search over ``n_components``. Defaults to 5.

    Attributes
    ----------
    vip_: `np.ndarray`
        The VIP score of every feature.

    support_: `np.ndarray`
        Boolean mask of the selected features.

    model_: `spectraxai.models.StandardModel`
        The fitted PLS model.

    """

    def __init__(
        self, threshold: float = 1.0, n_components: int | None = None, cv: int = 5
    ) -> None:
        self.threshold = threshold
        self.n_components = n_components
        self.cv = cv

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.n_components is None:
            model = StandardModel(Model.PLS, cv=self.cv)
        else:
            n_components = min(self.n_components, X.shape[1], X.shape[0])
            model = StandardModel(
                Model.PLS, init_hyperparameters={"n_components": n_components}
            )
        self.model_ = model.fit(X, y)
        self.vip_ = cast(np.ndarray, model.feature_importance()[0])
        self.support_ = self.vip_ > self.threshold
        if not self.support_.any():
            # Always keep at least the most important feature
            self.support_[np.argmax(self.vip_)] = True


class SelectKBest(_SpectralSelector):
    """Keep the ``k`` wavelengths ranked highest by a pre-hoc feature ranking.

    Parameters
    ----------
    method: `spectraxai.explain.FeatureRanking`, optional
        The ranking: correlation (absolute value), mutual information or
        F-statistic. Defaults to None, i.e. `FeatureRanking.CORR`.

    k: `int` or ``"all"``, optional
        Number of features to keep. Defaults to 10.

    random_state: `int`, optional
        Seed for mutual information. Defaults to None.

    Attributes
    ----------
    scores_: `np.ndarray`
        The score of every feature (for several outputs, the maximum across them).

    support_: `np.ndarray`
        Boolean mask of the selected features.

    """

    def __init__(
        self,
        method: FeatureRanking | None = None,
        k: int | str = 10,
        random_state: int | None = None,
    ) -> None:
        self.method = method
        self.k = k
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_features = X.shape[1]
        k = n_features if self.k == "all" else self.k
        if not isinstance(k, int) or not 1 <= k <= n_features:
            raise SpectraXAIValueError(
                'k should be "all" or an int between 1 and the number of features'
            )
        ranking = PreHocAnalysis(
            Dataset(X, y), random_state=self.random_state
        ).feature_importance(
            FeatureRanking.CORR if self.method is None else self.method
        )
        self.scores_ = np.nan_to_num(np.abs(ranking)).max(axis=0)
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[np.argsort(-self.scores_, kind="stable")[:k]] = True
