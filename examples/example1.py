from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel
import pandas
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
X, Y = df.loc[:, "350":"2500":20], df.OM

X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst = Dataset(X, Y).train_test_split(DatasetSplit.CROSS_VALIDATION, 5)

methods = [
    SpectralPreprocessing.NONE,
    SpectralPreprocessing.ABS,
    [SpectralPreprocessing.ABS, (SpectralPreprocessing.SG1, {"window_length": 31, "polyorder": 3})],
    [SpectralPreprocessing.ABS, SpectralPreprocessing.SNV],
    [SpectralPreprocessing.ABS, (SpectralPreprocessing.SG1, {"window_length": 51, "polyorder": 3}), SpectralPreprocessing.SNV],
    [SpectralPreprocessing.ABS, (SpectralPreprocessing.SG2, {"window_length": 51, "polyorder": 3})],
    [SpectralPreprocessing.NONE, (SpectralPreprocessing.SG1, {"window_length": 51, "polyorder": 3})],
    [SpectralPreprocessing.NONE, SpectralPreprocessing.SNV]
]

for i in range(len(idx_trn)):
    print("PLS - {0}/{1}".format(i+1, len(idx_trn)))
    for j in range(len(methods)):
        pls = StandardModel(Model.PLS)
        # or, e.g. svr = StandardModel(Model.SVR)
        res = pls.train(X.to_numpy(), Y.to_numpy(), methods[j], idx_trn[i])
        print("{0} - {1}".format(j+1, res))
