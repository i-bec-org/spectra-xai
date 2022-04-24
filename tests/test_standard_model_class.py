import unittest
import os
import pandas
import numpy as np
from .context import spectraxai, DATA_FOLDER
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.dataset import Dataset


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
                Model.CUBIST,
                best_hyperparameters={
                    "n_committees": 5,
                    "neighbors": 1,
                    "composite": True,
                },
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
            StandardModel(
                Model.CUBIST,
                grid_search_hyperparameters={
                    "neighbors": [1, 5],
                    "n_committees": [1, 5, 10],
                    "composite": [True],
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
        X = df.loc[:, "350":"2500":20]
        self.dataset1D = Dataset(X, df.OM)
        self.dataset2D = Dataset(X, df.loc[:, ["OM", "Sand_Fraction"]])
        self.idx_trn, self.idx_tst = np.arange(200), np.arange(200, X.shape[0])

    def test_wrong_params_model_constructor(self):
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
        self.assertRaises(
            AssertionError,
            StandardModel,
            Model.CUBIST,
            {"neighbors": 1, "n_committees": 5},
            {"neighbors": [1, 5], "n_committees": [1, 5, 10]},
        )

    def test_wrong_params_model_train(self):
        for model in self.models:
            # Neither idx_trn nor idx_tst
            self.assertRaises(AssertionError, model.train, self.dataset1D)
            self.assertRaises(
                AssertionError, model.train, self.dataset1D, SpectralPreprocessing.NONE
            )
            # Both idx_trn and idx_tst
            self.assertRaises(
                AssertionError,
                model.train,
                self.dataset1D,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            self.assertRaises(
                AssertionError,
                model.train,
                self.dataset1D,
                SpectralPreprocessing.NONE,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            # No list of sequences given
            self.assertRaises(AssertionError, model.train_with_sequence, self.dataset1D)
            self.assertRaises(
                AssertionError,
                model.train_with_sequence,
                self.dataset1D,
                idx_trn=self.idx_trn,
            )
            self.assertRaises(
                AssertionError,
                model.train_with_sequence,
                self.dataset1D,
                idx_tst=self.idx_tst,
            )
            # Neither idx_trn nor idx_tst passed
            self.assertRaises(
                AssertionError, model.train_with_sequence, self.dataset1D, self.methods
            )
            # Both idx_trn and idx_tst passed
            self.assertRaises(
                AssertionError,
                model.train_with_sequence,
                self.dataset1D,
                self.methods,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            # This should work
            model.train_with_sequence(
                self.dataset1D, self.methods, idx_trn=self.idx_trn
            )

    def test_multi_output(self):
        for model in self.models:
            if model.model in [Model.SVR, Model.CUBIST]:
                self.assertRaises(
                    AssertionError,
                    model.train,
                    self.dataset2D,
                    idx_trn=self.idx_trn,
                )
                self.assertRaises(
                    AssertionError,
                    model.train_with_sequence,
                    self.dataset2D,
                    self.methods,
                    idx_trn=self.idx_trn,
                )
            else:
                self.assertEqual(
                    len(model.train(self.dataset2D, idx_trn=self.idx_trn)), 2
                )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
