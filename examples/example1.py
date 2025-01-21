# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib import resources

import pandas

from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel


DATA_FOLDER = "spectraxai.data"

with resources.path(DATA_FOLDER, "SSL_GR.csv") as p:
    df = pandas.read_csv(p)
dataset = Dataset(df.loc[:, "350":"2500":20], df.OM)
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.CROSS_VALIDATION, 5)

methods = [
    SpectralPreprocessing.NONE,
    SpectralPreprocessing.ABS,
    [
        SpectralPreprocessing.ABS,
        (SpectralPreprocessing.SG1, {"window_length": 31, "polyorder": 3}),
    ],
    [SpectralPreprocessing.ABS, SpectralPreprocessing.SNV],
    [
        SpectralPreprocessing.ABS,
        (SpectralPreprocessing.SG1, {"window_length": 51, "polyorder": 3}),
        SpectralPreprocessing.SNV,
    ],
    [
        SpectralPreprocessing.ABS,
        (SpectralPreprocessing.SG2, {"window_length": 51, "polyorder": 3}),
    ],
    [
        SpectralPreprocessing.NONE,
        (SpectralPreprocessing.SG1, {"window_length": 51, "polyorder": 3}),
    ],
    [SpectralPreprocessing.NONE, SpectralPreprocessing.SNV],
]

res = StandardModel(Model.PLS).fit_and_predict_multiple(dataset, methods, idx_trn)
print(res[["pre_process", "fold", "RMSE", "R2", "RPIQ"]])
