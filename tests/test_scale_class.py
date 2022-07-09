import unittest
from sklearn.datasets import make_regression
from spectraxai.dataset import Scale

class TestScaleClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nsamples = 100
        self.nfeatures = 20
        self.X, self.y = make_regression(
            n_samples=self.nsamples, 
            n_features=self.nfeatures
        )

    def test_standard(self):
        X, x_scale_params, x_scale_attributes = Scale.apply(self.X, Scale.STANDARD)
        Y, y_scale_params, y_scale_attributes = Scale.apply(self.y, Scale.STANDARD)
        X_new, y_new = make_regression(n_samples=50, n_features=self.nfeatures)
        X_new_scaled = Scale.apply(X_new, Scale.STANDARD, x_scale_params, x_scale_attributes)
        y_new_scaled = Scale.apply(y_new, Scale.STANDARD, y_scale_params, y_scale_attributes)
        
