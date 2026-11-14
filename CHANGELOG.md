<!--
SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to this project are documented here. See
[ROADMAP.md](ROADMAP.md) for the plan towards 1.2.0.

## [Unreleased] — 1.2.0

### Added
- `spectraxai.exceptions.SpectraXAIValueError`, raised for invalid arguments. It
  subclasses both `ValueError` and `AssertionError`, so existing
  `except AssertionError` / `except ValueError` handlers keep working.
- `random_state` keyword argument (default `None`, i.e. the previous behaviour) on
  `StandardModel`, `Dataset.train_test_split`, `PreHocAnalysis`,
  `PostHocAnalysis`, `utils.sigest` and `utils.estimate_epsilon`.
- `StandardModel.param_grid_`: the grid actually searched by the last `fit`.
- `Dataset.train_test_split(..., n_bins=10)` for stratifying continuous outputs.
- `SpectralPreprocessing.init_class` now parses everything `str()` produces for a
  pre-processing sequence (single members, members with options, nested lists,
  names or values), e.g. the `pre_process` column of `fit_and_predict`.

- **Plotting contract** (`spectraxai.plotting`). Every plot accepts axes to draw
  into, returns what it drew on, forwards extra keyword arguments to its main
  matplotlib call, and never changes global matplotlib state:
  - `PreHocAnalysis.correlogram(..., axes=)` and
    `PreHocAnalysis.mean_spectrum_by_range(..., axes=)` are new (keyword-only).
  - `**kwargs` on `PreHocAnalysis.bar_plot_importance`, `correlogram`,
    `mean_spectrum`, `mean_spectrum_by_range`, `PostHocAnalysis.bar_plot_importance`,
    `circular_bar_plot_importance`, `bar_plot_permutation_importance` and
    `utils.scatter_plot` (e.g. `color=`, `lw=`, `s=`, `label=`).
  - `spectraxai.plotting.style()` context manager to apply the library's look to
    your own figures.
  - Multi-output plots accept axes in any shape (e.g. from `plt.subplots(1, 2)`).

- **scikit-learn interface.** `StandardModel` is now a scikit-learn regressor
  (`BaseEstimator`, `RegressorMixin`): it works with `clone`, pipelines,
  `GridSearchCV`, `cross_val_score`, `permutation_importance` and SAGE.
  - `fit` also accepts `fit(X, y)` and returns `self`; `fit(dataset)` and
    `fit(dataset, cv=...)` work as before.
  - New constructor keywords: `estimator=`, `cv=` (default 5), `n_jobs=` (default
    -1).
  - `Model.CUSTOM` + `estimator=` trains any scikit-learn regressor (cloned, never
    modified), with or without a grid search.
  - New `pipeline_` (SVR scaler + `best_model`), used by `predict`; `predict`
    also accepts a `Dataset` and checks the number of features.
  - New `StandardModel.feature_importance()` returning `(importance, method)`.
  - `PostHocAnalysis.bar_plot_permutation_importance` and `sage_importance`
    accept a `StandardModel` directly. For SVR this is the correct way: passing
    `best_model` skips the input scaling.
- `spectraxai.preprocessing.SpectralPreprocessor`, a scikit-learn transformer for
  any `SpectralPreprocessingSequence`.
- New pre-processing: `SpectralPreprocessing.MSC`, `DETREND` and `MEAN_CENTER`
  (and `Spectra.msc`, `detrend`, `mean_center`). The stateful ones learn their
  reference from the training samples in `fit_and_predict` and in
  `SpectralPreprocessor`, and from the given spectra in `Spectra.preprocess`.
- `Dataset.wavelengths` (from numeric `X_names`, or the new `wavelengths=`
  constructor argument), used by continuum removal and the spectral plots;
  `Spectra(X, wavelengths=)` and `Spectra.cr(wavelengths=)`.
- `Dataset.to_dataframe()`.
- **Unified results.** Every `fit_and_predict` row now also has `model_name`,
  `best_hyperparameters` (dict), `feature_importance` (array or None, now also
  for SVR), `importance_method` (`"VIP"`, `"impurity"`, `"usage"`, `"gain"`,
  `"coef"`, `"native"`, `"permutation"` or None) and `random_state`. All existing
  columns are kept. `return_permutation_importance=True` fills in the test-set
  permutation importance for models without one.

### Changed (models)
- `fit_and_predict` fits the pre-processing on the training samples and applies
  it to the test samples. For stateless pre-processing the results are identical
  to 1.1.0.
- Continuum removal in `Dataset.preprocess` / `fit_and_predict` uses the
  dataset's wavelengths when known. For evenly spaced wavelengths (e.g. the
  bundled SSL) results are identical; for unevenly spaced ones they are now
  correct.
- Passing `cv` positionally to `StandardModel.fit(dataset, cv)` is deprecated
  (`DeprecationWarning`); use `cv=`.
- Requires scikit-learn >= 1.6.

### Changed (plots)
- Constructing `PreHocAnalysis` / `PostHocAnalysis` no longer calls
  `plt.style.use("seaborn-v0_8-whitegrid")` globally. Figures the library creates
  are still drawn in that style; to keep it for your own figures use
  `with spectraxai.plotting.style():` (or call `plt.style.use` yourself).
- Plots drawn into axes you supply keep your style; `correlogram` only calls
  `tight_layout` on figures it created.
- `circular_bar_plot_importance` no longer hides the frame of whatever axes was
  current; it raises if the supplied `ax` is not polar.
- `utils.scatter_plot` accepts `(n, 1)` columns (e.g. PLS predictions) and is
  styled like the other plots when it creates the figure.

### Fixed
These fixes may change numeric results compared to 1.1.0.
- **Grid search state leaked between fits.** `StandardModel.fit` stored the
  dataset-dependent SVR `gamma` grid and PLS `n_components` grid in
  `grid_search_hyperparameters`, so every later fit (other folds, other
  pre-processings in `fit_and_predict_multiple`, other datasets) reused the grid
  computed for the first one. Grids are now computed per fit.
- Index bounds checks in `Dataset.subset` and `Dataset.train_test_split_explicit`
  accepted `n_samples` as a valid index.
- `DatasetSplit.STRATIFIED` failed on continuous outputs; it now stratifies on
  quantile bins of the (first) output. Class-like outputs are stratified on their
  labels exactly as before.
- `fit_and_predict` raised `KeyError` for SVR/RF/Cubist when `init_hyperparameters`
  did not contain every reported hyperparameter; missing values now fall back to
  the estimator's own.
- `fit_and_predict` now reports hyperparameters and `feature_importance` for
  XGBoost.
- `fit_and_predict(..., get_model=True)` ignored `get_model` for cross-validation
  folds.
- Mutual-information and F-statistic rankings divided by zero (NaN) for
  uninformative outputs.
- `PostHocAnalysis.bar_plot_permutation_importance` labelled the y axis "Mean
  decrease in impurity"; it is now "Mean decrease in score".
- `continuum_removal` ignored `wvs` for 2-D inputs.
- `Spectra.preprocess` silently returned `None` for unknown methods; it now raises.

### Changed
- Packaging moved from `setup.py` to `pyproject.toml` (same dependencies).
- Tests run with pytest, with fast synthetic fixtures; the SSL-based model tests
  are marked `slow`. CI also runs flake8, black, pyright and the examples.
