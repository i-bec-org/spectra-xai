from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.models import Model, StandardModel
import pandas
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
dataset = Dataset(df.loc[:, "350":"2500":20], df.OM)

X_trn, X_tst, Y_trn, Y_tst, idx_trn, idx_tst = dataset.train_test_split(
    DatasetSplit.CROSS_VALIDATION, 5
)

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

for method in methods:
    for i, fold in enumerate(idx_trn):
        res = StandardModel(Model.PLS).train(dataset, method, fold)
        print(
            "{0}, fold {1}: RMSE {2:.2f}".format(
                res[0]["pre-process"], i + 1, res[0]["RMSE"]
            )
        )
