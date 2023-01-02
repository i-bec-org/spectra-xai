import unittest
import numpy as np
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.dataset import DatasetSplit
from spectraxai.data import load_GR_SSL


class TestStandardModelClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.models = [
            StandardModel(Model.PLS),
            StandardModel(Model.SVR),
            StandardModel(Model.RF),
            StandardModel(Model.CUBIST),
            StandardModel(Model.PLS, init_hyperparameters={"n_components": 2}),
            StandardModel(
                Model.SVR,
                init_hyperparameters={"C": 16.0, "epsilon": 0.5, "gamma": 0.001},
            ),
            StandardModel(
                Model.RF,
                init_hyperparameters={"max_features": "log2", "n_estimators": 50},
            ),
            StandardModel(
                Model.CUBIST,
                init_hyperparameters={
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
                    "max_features": ["log2", 1.0],
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
        self.dataset1D = load_GR_SSL()
        self.dataset2D = load_GR_SSL(properties=["OM", "Sand_Fraction"])
        self.idx_trn, self.idx_tst = (
            np.arange(200),
            np.arange(200, self.dataset1D.n_samples),
        )

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
            {"max_features": ["log2", 1.0], "n_estimators": [50, 100]},
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
            self.assertRaises(AssertionError, model.fit_and_predict, self.dataset1D)
            self.assertRaises(
                AssertionError,
                model.fit_and_predict,
                self.dataset1D,
                SpectralPreprocessing.NONE,
            )
            # Both idx_trn and idx_tst
            self.assertRaises(
                AssertionError,
                model.fit_and_predict,
                self.dataset1D,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            self.assertRaises(
                AssertionError,
                model.fit_and_predict,
                self.dataset1D,
                SpectralPreprocessing.NONE,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            # No list of sequences given
            self.assertRaises(
                ValueError, model.fit_and_predict_multiple, self.dataset1D
            )
            self.assertRaises(
                ValueError,
                model.fit_and_predict_multiple,
                self.dataset1D,
                idx_trn=self.idx_trn,
            )
            self.assertRaises(
                ValueError,
                model.fit_and_predict_multiple,
                self.dataset1D,
                idx_tst=self.idx_tst,
            )
            # Neither idx_trn nor idx_tst passed
            self.assertRaises(
                AssertionError,
                model.fit_and_predict_multiple,
                self.dataset1D,
                self.methods,
            )
            # Both idx_trn and idx_tst passed
            self.assertRaises(
                AssertionError,
                model.fit_and_predict_multiple,
                self.dataset1D,
                self.methods,
                idx_trn=self.idx_trn,
                idx_tst=self.idx_tst,
            )
            # This should work
            model.fit_and_predict_multiple(
                self.dataset1D, self.methods, idx_trn=self.idx_trn
            )

    def test_multi_output(self):
        for model in self.models:
            if model.model in [Model.SVR, Model.CUBIST]:
                self.assertRaises(
                    AssertionError,
                    model.fit_and_predict,
                    self.dataset2D,
                    idx_trn=self.idx_trn,
                )
                self.assertRaises(
                    AssertionError,
                    model.fit_and_predict_multiple,
                    self.dataset2D,
                    self.methods,
                    idx_trn=self.idx_trn,
                )
            else:
                self.assertEqual(
                    len(model.fit_and_predict(self.dataset2D, idx_trn=self.idx_trn)), 2
                )

    def test_fit_and_predict_multiple(self):
        models = [
            StandardModel(
                Model.PLS, grid_search_hyperparameters={"n_components": [10, 20]}
            ),
            StandardModel(
                Model.RF,
                grid_search_hyperparameters={
                    "max_features": ["log2"],
                    "n_estimators": [50, 100],
                },
            ),
        ]
        for model in models:
            idx_trn, idx_tst = self.dataset1D.train_test_split(
                DatasetSplit.CROSS_VALIDATION, 3
            )
            for trn, tst in [(self.idx_trn, self.idx_tst), (idx_trn, idx_tst)]:
                model.fit_and_predict_multiple(
                    self.dataset1D, self.methods, idx_trn=trn
                )
                model.fit_and_predict_multiple(
                    self.dataset1D, self.methods, idx_tst=tst
                )
                model.fit_and_predict_multiple(
                    self.dataset2D, self.methods, idx_trn=trn
                )
                model.fit_and_predict_multiple(
                    self.dataset2D, self.methods, idx_tst=tst
                )

    def test_leave_one_out(self):
        idx_trn, idx_tst = self.dataset1D.train_test_split(
            DatasetSplit.CROSS_VALIDATION, self.dataset1D.n_samples
        )
        model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 2})
        for i in range(self.dataset1D.n_samples):
            model.fit(dataset=self.dataset1D.subset(idx_trn[i]))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
