from spectraxai.spectraxai import DatasetSplit
import pandas
from spectraxai import Dataset, SpectralPreprocessing, StandardModel, Model
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
X, Y = df.loc[:, "350":"2500":20], df.OM

_, _, _, _, idx_trn, _ = Dataset(X, Y).train_test_split(DatasetSplit.RANDOM, 0.8)

pls = StandardModel(Model.PLS)
res = pls.train(X.to_numpy(), Y.to_numpy(), SpectralPreprocessing.NONE, idx_trn)
print(res)
