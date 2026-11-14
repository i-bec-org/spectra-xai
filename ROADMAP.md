<!--
SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# SpectraXAI 1.2.0 roadmap

Goal: make `spectraxai` more solid — consistent, sklearn-compatible interfaces,
robust and composable plots, real test coverage, feature selection, more models —
**without breaking any code that works on v1.1.0**.

## Ground rules

| # | Decision |
|---|----------|
| D1 | Release as **1.2.0**. Every public name and signature from v1.1.0 keeps working. New parameters are keyword-only with defaults. Anything to retire emits `DeprecationWarning` and stays until 2.0. |
| D2 | Bug fixes that change numeric results are allowed and listed in `CHANGELOG.md`. Every source of randomness gets `random_state=None` (default preserves current behaviour). |
| D3 | Validation errors raise `SpectraXAIValueError(ValueError, AssertionError)` so existing `except AssertionError` still catches them. |
| D4 | Scope stays **regression only**; fix docstrings that mention classification. Keep interfaces open for classification later. |
| D5 | Move packaging from `setup.py` to `pyproject.toml`, same dependencies (xgboost, cubist, sage remain required). New heavy deps are optional extras only. |
| D6 | Keep Azure Pipelines. Keep committing `docs/` and regenerate it at the end of each phase. |
| D7 | Deliver as six phased PRs, each releasable on its own, in the order below. |

## Phase 1 — Test infrastructure and bug fixes

**Testing**
- Switch the runner to pytest (existing `unittest` classes keep running).
- Add a small, seeded synthetic spectral fixture (`tests/conftest.py`) for fast unit tests; keep `load_GR_SSL` for a few integration tests.
- Force the `Agg` backend; close figures after each test.
- Plot tests assert on drawn artists (lines, bars, NaN gaps) and that a passed `ax` is the one drawn on. No image-comparison tests.
- CI: add pyright (basic) and flake8 jobs; coverage floor ≈ 85 %.
- CI: add an `nbmake` job running `examples/*.ipynb` and `examples/*.py` at coarse subsampling.

**Bugs to fix (one regression test each)**
- `StandardModel.fit` writes SVR `gamma` and PLS `n_components` into `self.grid_search_hyperparameters`, so later fits (folds, `fit_and_predict_multiple`) reuse the grid of the first dataset/pre-processing. Compute per-fit grids locally.
- Off-by-one bounds checks (`idx <= n_samples` → `<`) in `Dataset.subset` and `Dataset.train_test_split_explicit`.
- Unseeded randomness: RF, XGBoost, `KFold(shuffle=True)`, `sigest`, `estimate_epsilon`, random split → `random_state`.
- `PreHocAnalysis._mi` / `_f_statistic` divide by `max` without guarding zero.
- `bar_plot_permutation_importance` y-label says "Mean decrease in impurity"; should be "Mean decrease in score".
- XGBoost rows in `fit_and_predict` lack hyperparameters and `feature_importance`.
- `DatasetSplit.STRATIFIED` runs `StratifiedKFold` on continuous Y. Stratify on quantile bins of Y (`n_bins=10` default).
- `SpectralPreprocessing.init_class` uses fragile string replacement. Replace with a proper parser that round-trips everything `str(preprocess)` produces (property-based test); same signature and output.
- Replace validation `AssertionError`s with `SpectraXAIValueError` (D3).

**Done when:** all tests pass on 3.13–3.15, lint + type-check jobs green, CHANGELOG lists every behavioural fix.

## Phase 2 — Plot contract

Every plotting function:
- accepts `ax=` (or `axes=` for multi-panel) and returns what it drew on;
- never mutates global state — no `plt.style.use`, `plt.axis`, or `tight_layout` on a figure it didn't create;
- forwards `**kwargs` to the main matplotlib call;
- keeps today's default `figsize`/`dpi` only when it creates the figure itself.

Styling moves from `_Explain.__init__` into an opt-in `spectraxai.plotting.style()` context manager, applied automatically only when the library creates the figure.

Functions to change: `PreHocAnalysis.correlogram` (add `axes=`), `PreHocAnalysis.mean_spectrum_by_range` (add `axes=`), `PostHocAnalysis.circular_bar_plot_importance` (drop `plt.axis("off")` on the current axes; require/create a polar `ax`), `utils.scatter_plot` (no `plt.figure()`), plus audit of `bar_plot_*` and `mean_spectrum`.

**Done when:** every plot is tested drawing into a user-supplied axes inside a multi-panel figure, and `plt.rcParams` is unchanged after constructing `PreHocAnalysis`/`PostHocAnalysis`.

## Phase 3 — sklearn interface

**`StandardModel`**
- Becomes an sklearn estimator (`BaseEstimator`, `RegressorMixin`). `fit` accepts a `Dataset` *or* `(X, y)` (detected).
- `get_params` covers `model`, `init_hyperparameters`, `grid_search_hyperparameters`, `estimator`, `cv`, `random_state`, `n_jobs` (default `-1`, as today).
- `best_model` stays the bare estimator; new `pipeline_` holds scaler + estimator; `predict` goes through `pipeline_`.
- `Model.CUSTOM` + `estimator=` lets users bring any sklearn regressor; `model` stays the first positional argument.
- `PostHocAnalysis` methods accept a `StandardModel` directly — fixes SVR receiving unscaled X in permutation importance and SAGE.

**Pre-processing and data**
- New `SpectralPreprocessor` sklearn transformer built on `SpectralPreprocessingSequence`.
- New `SpectralPreprocessing` members: `MSC`, `DETREND`, `MEAN_CENTER`. MSC learns its reference at `fit`; `fit_and_predict` uses the transformer so the reference never sees test data.
- `Dataset.wavelengths` property (floats from `X_names`, or `None`) and optional `wavelengths=` constructor argument; `X_names` unchanged.
- `continuum_removal` / `Spectra.cr` accept `wavelengths=`.
- `Dataset.to_dataframe()`.

**Unified results schema** — every `fit_and_predict` row has `model_name`, `best_hyperparameters` (dict), `feature_importance` (native or `None`), `importance_method` (`"VIP"`, `"impurity"`, `"coef"`, `None`), `random_state`. Existing flat columns (`C`, `n_components`, `SVs`, …) and the `model` column from `get_model=True` stay. New `return_permutation_importance=False` computes test-set importance for models without native importance.

**Done when:** `sklearn.utils.estimator_checks` passes for the relevant checks, `cross_val_score(StandardModel(...), X, y)` works, and all v1.1.0 usage in the tests and examples runs unchanged.

## Phase 4 — Feature selection (+ CLHS)

New module `spectraxai.selection`:
- `RFE` / `RFECV` wrapping sklearn's.
- VIP filtering for PLS (default threshold 1).
- `SelectKBest` driven by the existing `FeatureRanking` methods (correlation, mutual information, F-statistic).

Each selector is an sklearn transformer **and** offers `selector.fit(dataset).transform(dataset) -> Dataset`, keeping `X_names`/`wavelengths` in sync. Importance getter by model: VIP for PLS, `feature_importances_` for RF/XGBoost/Cubist, coefficients for linear models, permutation importance fallback for SVR and others.

New plot: `plot_selected_bands(ax=...)` shading kept wavelengths over the mean spectrum.

Implement `DatasetSplit.CLHS` (conditioned Latin hypercube sampling, pure numpy annealing, seeded).

Later (out of scope): iPLS, CARS, SPA, GA.

## Phase 5 — Models and explanations

- New `Model` members (sklearn only, no new deps): `RIDGE`, `LASSO`, `ELASTIC_NET`, `KNN`, `GPR`, `MLP`, each with a default grid and importance where available.
- New explanation plots (all following the Phase 2 contract): importance overlaid on the mean spectrum, `plot_pls_coefficients`, and a SAGE plot helper.
- Optional extra `spectraxai[shap]`: SHAP summary across wavelengths.

## Phase 6 — pycaret, docs, examples

- No pycaret dependency for now (stable 3.x doesn't support Python ≥ 3.12 well; 4.0 with 3.13 support is still alpha as of May 2026).
- Example notebook showing `SpectralPreprocessor` and selectors used via `pycaret.regression.setup(custom_pipeline=...)` and `Dataset.to_dataframe()`.
- When pycaret 4.0 is stable: add `spectraxai[pycaret]` extra and an optional, non-blocking CI job.
- Update notebooks and README to the new APIs (`ax=`, `random_state`, selectors); regenerate `docs/`.

## Out of scope for 1.2.0

Classification models, deep learning / 1D-CNNs, LightGBM/CatBoost, iPLS/CARS/SPA/GA, image-comparison plot tests.
