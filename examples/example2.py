from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel
import pandas
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
dataset = Dataset(df.loc[:, "350":"2500":20], df.OM)

idx_trn, _ = dataset.train_test_split(DatasetSplit.RANDOM, 0.8)
res = StandardModel(Model.PLS).train_and_test(dataset, SpectralPreprocessing.ABS, idx_trn=idx_trn)
print(res)
