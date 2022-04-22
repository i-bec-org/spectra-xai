import unittest
from .context import spectraxai
from spectraxai.dataset import Dataset, DatasetSplit, Scale
from spectraxai.spectra import SpectralPreprocessing
import numpy as np


class TestDatasetClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nsamples = 100
        self.nfeatures = 20
        X = (
            np.arange(self.nsamples * self.nfeatures).reshape(
                (self.nsamples, self.nfeatures)
            )
            + 1
        )
        y = np.array(list(range(self.nsamples)))
        Y = np.arange(self.nsamples * 2).reshape((self.nsamples, 2))
        self.datasetY1dim = Dataset(X, y)
        self.datasetY2dim = Dataset(X, Y)
        self.split_size = 0.6

    def _assert_X_size(self, X_trn, X_tst):
        self.assertTrue(X_trn.shape[1] == X_tst.shape[1] == self.nfeatures)
        self.assertTrue(X_trn.shape[0] == int(self.split_size * self.nsamples))
        self.assertTrue(X_tst.shape[0] == int((1 - self.split_size) * self.nsamples))

    def _assert_Y_size(self, y_trn, y_tst):
        self.assertTrue(y_tst.ndim == y_trn.ndim == 2)
        self.assertTrue(y_trn.shape[0] == int(self.split_size * self.nsamples))
        self.assertTrue(y_tst.shape[0] == int((1 - self.split_size) * self.nsamples))

    def _assert_indices(self, idx_trn, idx_tst):
        self.assertTrue(idx_trn.ndim == idx_tst.ndim == 1)
        self.assertTrue(idx_trn.shape[0] == int(self.split_size * self.nsamples))
        self.assertTrue(idx_tst.shape[0] == int((1 - self.split_size) * self.nsamples))
        self.assertFalse(bool(set(idx_trn).intersection(idx_tst)))

    def test_wrong_params(self):
        # test size not in (0, 1)
        self.assertRaises(
            AssertionError, self.datasetY1dim.train_test_split, DatasetSplit.RANDOM, 2.5
        )
        self.assertRaises(
            AssertionError,
            self.datasetY1dim.train_test_split,
            DatasetSplit.RANDOM,
            -2.5,
        )
        # Pass both trn and tst to explicit split
        self.assertRaises(
            AssertionError,
            self.datasetY1dim.train_test_split_explicit,
            trn=np.array([0, 1]),
            tst=np.array([2, 3]),
        )
        # Don't pass either trn or tst to explicit split
        self.assertRaises(AssertionError, self.datasetY1dim.train_test_split_explicit)

    def test_random_split(self):
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            X_trn, X_tst, y_trn, y_tst, idx_trn, idx_tst = dataset.train_test_split(
                DatasetSplit.RANDOM, self.split_size
            )
            self._assert_X_size(X_trn, X_tst)
            self._assert_Y_size(y_trn, y_tst)
            self._assert_indices(idx_trn, idx_tst)

    def test_ks_split(self):
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            X_trn, X_tst, y_trn, y_tst, idx_trn, idx_tst = dataset.train_test_split(
                DatasetSplit.KENNARD_STONE, self.split_size
            )
            self._assert_X_size(X_trn, X_tst)
            self._assert_Y_size(y_trn, y_tst)
            self._assert_indices(idx_trn, idx_tst)

    def test_explicit_split(self):
        trn = np.random.choice(
            np.arange(self.nsamples),
            int(self.nsamples * self.split_size),
            replace=False,
        )
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            (
                X_trn,
                X_tst,
                y_trn,
                y_tst,
                idx_trn,
                idx_tst,
            ) = dataset.train_test_split_explicit(trn=trn)
            self._assert_X_size(X_trn, X_tst)
            self._assert_Y_size(y_trn, y_tst)
            self._assert_indices(idx_trn, idx_tst)

    def test_get_matrix_3D(self):
        methods = [
            SpectralPreprocessing.ABS,
            (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}),
            [SpectralPreprocessing.ABS, SpectralPreprocessing.SNV],
            [
                (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}),
                SpectralPreprocessing.SNV,
            ],
        ]
        X = self.datasetY1dim.preprocess_3D(methods)
        self.assertTrue(X.shape[0] == self.nsamples)
        self.assertTrue(X.shape[1] == self.nfeatures)
        self.assertTrue(X.shape[2] == len(methods))

    def test_scale_X(self):
        X = self.datasetY2dim.X
        self.assertTrue(np.all(np.abs(self.datasetY2dim.apply_scale_X(Scale.STANDARD).apply_unscale_X(Scale.STANDARD).X - X) < 1e-6))
        self.assertTrue(np.all(np.abs(self.datasetY2dim.apply_scale_X(Scale.MINMAX).apply_unscale_X(Scale.MINMAX).X - X) < 1e-6))

        methods = [
            SpectralPreprocessing.ABS,
            (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}),
            [SpectralPreprocessing.ABS, SpectralPreprocessing.SNV],
            [
                (SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}),
                SpectralPreprocessing.SNV,
            ],
        ]
        X = self.datasetY1dim.preprocess_3D(methods)
        X_scaled, props = Dataset.scale_X(X, Scale.STANDARD)
        self.assertTrue(np.all(np.abs(Dataset.unscale_X(X_scaled, Scale.STANDARD, props["params"], props["attributes"]) - X) < 1e-6))
        X_scaled, props = Dataset.scale_X(X, Scale.MINMAX)
        self.assertTrue(np.all(np.abs(Dataset.unscale_X(X_scaled, Scale.MINMAX, props["params"], props["attributes"]) - X) < 1e-6))

    def test_scale_Y(self):
        y = self.datasetY1dim.Y
        self.assertTrue(np.all(np.abs(
            self.datasetY1dim.apply_scale_Y(Scale.STANDARD).apply_unscale_Y(Scale.STANDARD).Y - y) < 1e-6))
        self.assertTrue(np.all(np.abs(
            self.datasetY1dim.apply_scale_Y(Scale.MINMAX).apply_unscale_Y(Scale.MINMAX).Y - y) < 1e-6))

        Y = self.datasetY2dim.Y
        self.assertTrue(np.all(np.abs(
            self.datasetY2dim.apply_scale_Y(Scale.STANDARD).apply_unscale_Y(Scale.STANDARD).Y - Y) < 1e-6))
        self.assertTrue(np.all(np.abs(
            self.datasetY2dim.apply_scale_Y(Scale.MINMAX).apply_unscale_Y(Scale.MINMAX).Y - Y) < 1e-6))

        X_trn, X_tst, y_trn, y_tst, idx_trn, idx_tst = self.datasetY2dim.train_test_split(DatasetSplit.KENNARD_STONE, self.split_size)

        y_trn_scaled = Dataset(X_trn, y_trn).apply_scale_Y(Scale.STANDARD)
        y_tst_scaled = Dataset(X_tst, y_tst).apply_scale_Y(Scale.STANDARD, y_trn_scaled.get_scale_Y_props["params"], y_trn_scaled.get_scale_Y_props["attributes"])
        self.assertTrue(np.all(np.abs(y_trn_scaled.apply_unscale_Y(Y=y_tst_scaled.Y, method=Scale.STANDARD) - y_tst) < 1e-6))

        y_trn_scaled = Dataset(X_trn, y_trn).apply_scale_Y(Scale.MINMAX)
        y_tst_scaled = Dataset(X_tst, y_tst).apply_scale_Y(Scale.MINMAX, y_trn_scaled.get_scale_Y_props["params"], y_trn_scaled.get_scale_Y_props["attributes"])
        self.assertTrue(np.all(np.abs(y_trn_scaled.apply_unscale_Y(Y=y_tst_scaled.Y, method=Scale.MINMAX) - y_tst) < 1e-6))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
