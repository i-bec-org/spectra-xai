# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Select informative wavelengths and check how much accuracy they keep.

Three selectors are fitted on the training set only (to avoid leaking the test
set into the selection): VIP filtering, recursive feature elimination with
cross-validation, and the most correlated wavelengths. Each reduced dataset is
then modelled with PLS and the bands kept by each method are plotted side by side.
"""

import matplotlib.pyplot as plt

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.explain import PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.plotting import style
from spectraxai.selection import RFECV, SelectKBest, VIPSelector
from spectraxai.spectra import SpectralPreprocessing

dataset = load_GR_SSL().preprocess(SpectralPreprocessing.ABS)
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
training = dataset.subset(idx_trn)

pls_8 = StandardModel(Model.PLS, init_hyperparameters={"n_components": 8})
selectors = {
    "VIP > 1": VIPSelector(n_components=10),
    "RFECV": RFECV(pls_8, step=5, cv=5, min_features_to_select=10),
    "20 most correlated": SelectKBest(k=20),
}

pls = StandardModel(Model.PLS, random_state=0)
full = pls.fit_and_predict(dataset, idx_trn=idx_trn)
print(f"all {dataset.n_features} wavelengths: RMSE={full.loc[0, 'RMSE']:.3f}")

xai = PreHocAnalysis(dataset)
with style():
    fig, axes = plt.subplots(len(selectors), 1, figsize=(10, 9), sharex=True)
    for ax, (name, selector) in zip(axes, selectors.items()):
        reduced = selector.fit(training).transform(dataset)
        res = pls.fit_and_predict(reduced, idx_trn=idx_trn)
        print(
            f"{name}: {reduced.n_features} wavelengths, RMSE={res.loc[0, 'RMSE']:.3f}"
        )
        xai.plot_selected_bands(selector, ax=ax)
        ax.set_title(f"{name} ({reduced.n_features} wavelengths)")
    fig.tight_layout()
plt.show()
