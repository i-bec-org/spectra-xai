import unittest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from spectraxai.dataset import Scale


class TestScaleClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.n_samples = 500
        self.n_features = 20
        self.n_outputs = 2
        self.X, self.Y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_targets=self.n_outputs,
        )
        self.idx_trn = np.arange(300)
        self.idx_val = np.arange(300, self.n_samples)

    def test_standard(self):
        X_trn_scaled, x_scale_params, x_scale_attributes = Scale.apply(
            self.X[self.idx_trn], Scale.STANDARD
        )
        Y_trn_scaled, y_scale_params, y_scale_attributes = Scale.apply(
            self.Y[self.idx_trn], Scale.STANDARD
        )
        X_val_scaled = Scale.apply(
            self.X[self.idx_val], Scale.STANDARD, x_scale_params, x_scale_attributes
        )
        Y_val_scaled = Scale.apply(
            self.Y[self.idx_val], Scale.STANDARD, y_scale_params, y_scale_attributes
        )
        assert_allclose(X_trn_scaled.mean(axis=0), X_val_scaled.mean(axis=0), atol=0.35)
        assert_allclose(Y_trn_scaled.mean(axis=0), Y_val_scaled.mean(axis=0), atol=0.35)
        X_trn_original = Scale.inverse(
            X_trn_scaled, Scale.STANDARD, x_scale_params, x_scale_attributes
        )
        assert_allclose(self.X[self.idx_trn], X_trn_original)
        Y_trn_original = Scale.inverse(
            Y_trn_scaled, Scale.STANDARD, y_scale_params, y_scale_attributes
        )
        assert_allclose(self.Y[self.idx_trn], Y_trn_original)

    def test_minmax(self):
        X_trn_scaled, x_scale_params, x_scale_attributes = Scale.apply(
            self.X[self.idx_trn], Scale.MINMAX
        )
        Y_trn_scaled, y_scale_params, y_scale_attributes = Scale.apply(
            self.Y[self.idx_trn], Scale.MINMAX
        )
        X_val_scaled = Scale.apply(
            self.X[self.idx_val], Scale.MINMAX, x_scale_params, x_scale_attributes
        )
        Y_val_scaled = Scale.apply(
            self.Y[self.idx_val], Scale.MINMAX, y_scale_params, y_scale_attributes
        )
        assert_allclose(X_trn_scaled.mean(axis=0), X_val_scaled.mean(axis=0), atol=0.1)
        assert_allclose(Y_trn_scaled.mean(axis=0), Y_val_scaled.mean(axis=0), atol=0.1)
        X_trn_original = Scale.inverse(
            X_trn_scaled, Scale.MINMAX, x_scale_params, x_scale_attributes
        )
        assert_allclose(self.X[self.idx_trn], X_trn_original)
        Y_trn_original = Scale.inverse(
            Y_trn_scaled, Scale.MINMAX, y_scale_params, y_scale_attributes
        )
        assert_allclose(self.Y[self.idx_trn], Y_trn_original)
