import unittest
from spectraxai.spectra import Spectra, SpectralPreprocessing
from spectraxai.data import load_GR_SSL
from numpy.testing import assert_allclose


class TestStandardModelClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dataset = load_GR_SSL()

    def test_1D_constructor(self):
        spec = Spectra(self.dataset.X[0, :])
        self.assertTrue(spec.X.shape == (1, self.dataset.X.shape[1]))

    def test_2D_constructor(self):
        spec = Spectra(self.dataset.X)
        self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_simple_pre_treatments(self):
        pre_treatments = [
            SpectralPreprocessing.NONE,
            SpectralPreprocessing.ABS,
            SpectralPreprocessing.SNV,
            SpectralPreprocessing.CR,
        ]
        for pre_treatment in pre_treatments:
            spec = Spectra(self.dataset.X).apply(pre_treatment)
            self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_sg_pre_treatments(self):
        pre_treatments = [
            (SpectralPreprocessing.SG0, {"window_length": 21, "polyorder": 3}),
            (SpectralPreprocessing.SG1, {"window_length": 21, "polyorder": 3}),
            (SpectralPreprocessing.SG2, {"window_length": 21, "polyorder": 3}),
        ]
        for pre_treatment in pre_treatments:
            spec = Spectra(self.dataset.X).apply(pre_treatment[0], **pre_treatment[1])
            self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_reversible(self):
        spec = (
            Spectra(self.dataset.X)
            .apply(SpectralPreprocessing.ABS)
            .apply(SpectralPreprocessing.REF)
        )
        assert_allclose(spec.X, self.dataset.X)
