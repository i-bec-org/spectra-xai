import unittest
import numpy as np
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
        self.assertAlmostEqual(X_trn_scaled.mean(), X_val_scaled.mean(), delta=0.1)
        self.assertAlmostEqual(Y_trn_scaled.mean(), Y_val_scaled.mean(), delta=0.1)
