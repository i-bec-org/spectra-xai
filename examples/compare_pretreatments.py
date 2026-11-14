# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare spectral pre-treatments for PLS with 5-fold cross-validation.

Every pre-treatment (absorbance, Savitzky-Golay derivatives, SNV, MSC and their
combinations) is evaluated on the same folds, and the results are summarised per
pre-treatment as the mean and standard deviation across folds. Stateful steps
such as MSC learn their reference spectrum on the training folds only.
"""

import pandas as pd

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing as SP

dataset = load_GR_SSL()
idx_trn, _ = dataset.train_test_split(DatasetSplit.CROSS_VALIDATION, 5, random_state=0)

sg1 = (SP.SG1, {"window_length": 31, "polyorder": 3})
sg2 = (SP.SG2, {"window_length": 51, "polyorder": 3})
pretreatments = [
    SP.NONE,
    SP.ABS,
    [SP.ABS, sg1],
    [SP.ABS, SP.SNV],
    [SP.ABS, SP.MSC],
    [SP.ABS, sg1, SP.SNV],
    [SP.ABS, sg2],
    [SP.NONE, sg1],
]

pls = StandardModel(Model.PLS, random_state=0)
results = pls.fit_and_predict_multiple(dataset, pretreatments, idx_trn=idx_trn)

summary = pd.DataFrame(
    results.groupby("pre_process", sort=False)[["RMSE", "R2", "RPIQ"]]
    .agg(["mean", "std"])
    .round(3)
)
summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
summary = summary.sort_values(by="RMSE mean")
print(summary.to_string())
