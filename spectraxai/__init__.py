# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
r'''
# Welcome to the SpectraXAI library.

A library to easily develop XAI models for spectral datasets.

## Features

 - Easy setup with pip
 - Documented using pdoc following the numpydoc style
 - scikit-learn compatible models, transformers and feature selectors

## Modules

 - `spectraxai.dataset`: the `Dataset` (spectra, outputs, wavelengths), scaling and
   splitting (random, Kennard–Stone, cLHS, cross-validation, stratified)
 - `spectraxai.spectra`: spectral pre-processing (absorbance, SNV, MSC,
   Savitzky–Golay, continuum removal, ...)
 - `spectraxai.preprocessing`: pre-processing as a scikit-learn transformer
 - `spectraxai.models`: `StandardModel`, training and tuning PLS, SVR, RF, Cubist,
   XGBoost, Ridge, Lasso, Elastic Net, kNN, Gaussian processes, MLP or any
   scikit-learn regressor
 - `spectraxai.selection`: wavelength selection (RFE/RFECV, VIP, pre-hoc rankings)
 - `spectraxai.explain`: pre- and post-hoc explanations and their plots
 - `spectraxai.plotting`: the conventions shared by all plots
 - `spectraxai.data`: the bundled example datasets (vis–NIR and MIR soil spectra)

# Quickstart

As an example, we quickly develop a PLS model for a dataset and
plot the feature importance.
```python
"""
A small `spectraxai` example.
"""

# Load the dataset

from spectraxai.data import load_GR_SSL
dataset = load_GR_SSL()

# Apply SG1 on the spectra and train & test a PLS model

from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.dataset import DatasetSplit

idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
pls = StandardModel(Model.PLS, random_state=0)

results = pls.fit_and_predict(
    dataset,
    preprocess=(SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}),
    idx_trn=idx_trn
)
results[["RMSE", "R2", "RPIQ", "best_hyperparameters"]]

# Plot the feature importance (VIP) over the mean spectrum

from spectraxai.explain import PostHocAnalysis

xai = PostHocAnalysis(dataset.preprocess(
    (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3})
))
ax, importance_ax = xai.plot_importance_spectrum(results.iloc[0]["feature_importance"])
```

Every plot accepts a matplotlib ``ax`` (or ``axes``), so it can be drawn into your
own figures; see `spectraxai.plotting`.

.. include:: ../CHANGELOG.md
'''
