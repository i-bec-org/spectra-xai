# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quickstart: train and test a PLS model, then look at what it learned.

Loads the bundled vis-NIR soil spectral library, picks a representative training
set with Kennard-Stone, fits a PLS model on absorbance spectra (tuning the number
of components by cross-validation) and plots the VIP importance over the mean
spectrum.
"""

import matplotlib.pyplot as plt

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.explain import PostHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing

dataset = load_GR_SSL()  # soil organic matter (OM) from 350-2500 nm reflectance
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)

pls = StandardModel(Model.PLS, random_state=0)
results = pls.fit_and_predict(dataset, SpectralPreprocessing.ABS, idx_trn=idx_trn)
print(results[["N", "RMSE", "R2", "RPIQ", "best_hyperparameters"]].to_string())

# The fitted model is trained on absorbance, so explain it on absorbance spectra
absorbance = dataset.preprocess(SpectralPreprocessing.ABS)
ax, importance_ax = PostHocAnalysis(absorbance).plot_importance_spectrum(pls)
ax.set_ylabel("Absorbance")
importance_ax.set_ylabel("VIP")
plt.show()
