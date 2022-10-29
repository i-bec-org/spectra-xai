import unittest
from spectraxai.spectra import Spectra, SpectralPreprocessing
from spectraxai.data import load_GR_SSL
from numpy.testing import assert_allclose
import pandas


class TestSpectraClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dataset = load_GR_SSL()

    def test_1D_constructor(self):
        spec = Spectra(self.dataset.X[0, :])
        self.assertTrue(spec.X.shape == (1, self.dataset.X.shape[1]))

    def test_2D_constructor(self):
        spec = Spectra(self.dataset.X)
        self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_pandas_constructor(self):
        self.assertRaises(TypeError, Spectra, pandas.DataFrame(self.dataset.X))

    def test_simple_pre_treatments(self):
        pre_treatments = [
            SpectralPreprocessing.NONE,
            SpectralPreprocessing.ABS,
            SpectralPreprocessing.SNV,
            SpectralPreprocessing.CR,
        ]
        for pre_treatment in pre_treatments:
            spec = Spectra(self.dataset.X).preprocess(pre_treatment)
            self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_sg_pre_treatments(self):
        pre_treatments = [
            (SpectralPreprocessing.SG0, {"window_length": 21, "polyorder": 3}),
            (SpectralPreprocessing.SG1, {"window_length": 21, "polyorder": 3}),
            (SpectralPreprocessing.SG2, {"window_length": 21, "polyorder": 3}),
        ]
        for pre_treatment in pre_treatments:
            spec = Spectra(self.dataset.X).preprocess(pre_treatment)
            self.assertTrue(spec.X.shape == self.dataset.X.shape)

    def test_reversible(self):
        spec = (
            Spectra(self.dataset.X)
            .preprocess(SpectralPreprocessing.ABS)
            .preprocess(SpectralPreprocessing.REF)
        )
        assert_allclose(spec.X, self.dataset.X)
        spec = Spectra(self.dataset.X).absorbance().reflectance()
        assert_allclose(spec.X, self.dataset.X)

    def test_sequence(self):
        spec1 = Spectra(self.dataset.X).absorbance()
        spec2 = Spectra(self.dataset.X).preprocess(SpectralPreprocessing.ABS)
        assert_allclose(spec1.X, spec2.X)
        spec1 = Spectra(self.dataset.X).absorbance().snv()
        spec2 = Spectra(self.dataset.X).preprocess(
            [SpectralPreprocessing.ABS, SpectralPreprocessing.SNV]
        )
        assert_allclose(spec1.X, spec2.X)
        spec1 = Spectra(self.dataset.X).sg(deriv=1, window_length=21, polyorder=3).snv()
        spec2 = Spectra(self.dataset.X).preprocess(
            [
                (SpectralPreprocessing.SG1, {"window_length": 21, "polyorder": 3}),
                SpectralPreprocessing.SNV,
            ]
        )
        assert_allclose(spec1.X, spec2.X)
