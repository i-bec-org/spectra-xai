# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from spectraxai.dataset import DatasetSplit
from spectraxai.data import load_GR_SSL
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel

dataset = load_GR_SSL()

idx_trn, _ = dataset.train_test_split(DatasetSplit.RANDOM, 0.8)
res = StandardModel(Model.PLS).fit_and_predict(
    dataset, SpectralPreprocessing.ABS, idx_trn=idx_trn
)
print(res)
