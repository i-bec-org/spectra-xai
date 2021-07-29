import unittest
import os
import pandas
import numpy as np
from .context import spectraxai, DATA_FOLDER
from spectraxai import Model, StandardModel, SpectralPreprocessing


class TestStandardModelClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.models = [
            StandardModel(Model.PLS),
            StandardModel(Model.SVR),
            StandardModel(Model.RF),
            StandardModel(Model.PLS, best_hyperparameters={"n_components": 2}),
            StandardModel(
                Model.SVR,
                best_hyperparameters={"C": 16.0, "epsilon": 0.5, "gamma": 0.001},
            ),
            StandardModel(
                Model.RF,
                best_hyperparameters={"max_features": "log2", "n_estimators": 50},
            ),
            StandardModel(
                Model.PLS, grid_search_hyperparameters={"n_components": [2, 10, 20]}
            ),
            StandardModel(
                Model.SVR,
                grid_search_hyperparameters={
                    "C": [2, 10, 15, 20],
                    "epsilon": [0.1, 0.2, 0.6, 0.8],
                    "gamma": [0.001, 0.05, 0.1, 0.5],
                },
            ),
            StandardModel(
                Model.RF,
                grid_search_hyperparameters={
                    "max_features": ["log2", "auto"],
                    "n_estimators": [50, 100],
                },
            ),
        ]
        self.methods = [
            SpectralPreprocessing.ABS,
            (SpectralPreprocessing.SG1, {"window_length": 21, "polyorder": 3}),
            [
                (SpectralPreprocessing.SG1, {"window_length": 21, "polyorder": 3}),
                SpectralPreprocessing.SNV,
            ],
        ]
        df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
        self.X, self.Y = df.loc[:, "350":"2500":20], df.OM
        self.idx_trn, self.idx_tst = np.arange(200), np.arange(200, self.X.shape[0])

    def test_wrong_params(self):
        self.assertRaises(
            AssertionError,
            StandardModel,
            Model.PLS,
            {"n_components": 2},
            {"n_components": [2, 10, 20]},
        )
        self.assertRaises(
            AssertionError,
            StandardModel,
            Model.SVR,
            {"C": 16.0, "epsilon": 0.5, "gamma": 0.001},
            {
                "C": [2, 10, 15, 20],
                "epsilon": [0.1, 0.2, 0.6, 0.8],
                "gamma": [0.001, 0.05, 0.1, 0.5],
            },
        )
        self.assertRaises(
            AssertionError,
            StandardModel,
            Model.RF,
            {"max_features": "log2", "n_estimators": 50},
            {"max_features": ["log2", "auto"], "n_estimators": [50, 100]},
        )
        for model in self.models:
            self.assertRaises(AssertionError, model.train, self.X, self.Y)
            self.assertRaises(
                AssertionError, model.train, self.X, self.Y, SpectralPreprocessing.NONE
            )
            self.assertRaises(AssertionError, model.train, self.X, self.Y, idx_trn=self.idx_trn)
            self.assertRaises(AssertionError, model.train, self.X, self.Y, idx_tst=self.idx_tst)
            self.assertRaises(
                AssertionError, model.train, self.X, self.Y, idx_trn=self.idx_trn, idx_tst=self.idx_tst
            )
            # model.train(self.X, self.Y, SpectralPreprocessing.NONE, idx_trn=self.idx_trn)
            # model.train(self.X, self.Y, SpectralPreprocessing.NONE, idx_tst=self.idx_tst)
            self.assertRaises(
                AssertionError,
                model.train,
                self.X,
                self.Y,
                SpectralPreprocessing.NONE,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )

            self.assertRaises(AssertionError, model.train_with_sequence, self.X, self.Y)
            self.assertRaises(
                AssertionError, model.train_with_sequence, self.X, self.Y, self.methods
            )
            self.assertRaises(
                AssertionError, model.train_with_sequence, self.X, self.Y, idx_trn=self.idx_trn
            )
            self.assertRaises(
                AssertionError, model.train_with_sequence, self.X, self.Y, idx_tst=self.idx_tst
            )
            self.assertRaises(
                AssertionError, model.train, self.X, self.Y, idx_trn=self.idx_trn, idx_tst=self.idx_tst
            )
            model.train_with_sequence(self.X, self.Y, self.methods, idx_trn=self.idx_trn)
            # model.train_with_sequence(self.X, self.Y, self.methods, idx_tst=self.idx_tst)
            self.assertRaises(
                AssertionError,
                model.train_with_sequence,
                self.X,
                self.Y,
                self.methods,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
