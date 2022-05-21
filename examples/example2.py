from importlib import resources

import pandas

from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel


DATA_FOLDER = "spectraxai.data"

with resources.path(DATA_FOLDER, "SSL_GR.csv") as p:
    df = pandas.read_csv(p)
dataset = Dataset(df.loc[:, "350":"2500":20], df.OM)

idx_trn, _ = dataset.train_test_split(DatasetSplit.RANDOM, 0.8)
res = StandardModel(Model.PLS).fit_and_predict(
    dataset, SpectralPreprocessing.ABS, idx_trn=idx_trn
)
print(res)
