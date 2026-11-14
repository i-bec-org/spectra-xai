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
