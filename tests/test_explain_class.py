# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from typing import cast

from matplotlib import pyplot as plt
import numpy as np
from spectraxai.explain import PreHocAnalysis, FeatureRanking
from spectraxai.data import load_GR_SSL


class TestPreHocAnalysisClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.xai1D = PreHocAnalysis(load_GR_SSL())
        cls.xai2D = PreHocAnalysis(load_GR_SSL(properties=["OM", "Sand_Fraction"]))

    def test_feature_importance(self) -> None:
        for xai in [self.xai1D, self.xai2D]:
            for method in FeatureRanking:
                imp = xai.feature_importance(method)
                self.assertTrue(imp.shape[0] == xai.dataset.n_outputs)
                self.assertTrue(imp.shape[1] == xai.dataset.n_features)

    def test_bar_plot_feature_importance(self) -> None:
        for xai in [self.xai1D, self.xai2D]:
            for method in FeatureRanking:
                xai.bar_plot_importance(method)
                _, ax = plt.subplots(nrows=1, ncols=xai.dataset.n_outputs)
                xai.bar_plot_importance(method, ax=ax)
                plt.close("all")

    def test_mean_spectrum_excludes_bands(self) -> None:
        ax = self.xai1D.mean_spectrum(excluded_bands=[(400, 500)])
        wavelengths = np.asarray(self.xai1D.dataset.X_names, dtype=float)
        excluded = (wavelengths >= 400) & (wavelengths <= 500)
        plotted_wavelengths = np.concatenate(
            [cast(np.ndarray, line.get_xdata()) for line in ax.lines]
        )
        self.assertTrue(excluded.any())
        self.assertFalse(np.isin(plotted_wavelengths, wavelengths[excluded]).any())
        self.assertTrue(all(np.isfinite(line.get_ydata()).all() for line in ax.lines))
        plt.close("all")

    def test_mean_spectrum_by_range_excludes_bands(self) -> None:
        values = self.xai1D.dataset.Y[:, 0]
        y_ranges = [np.linspace(values.min() - 1, values.max() + 1, 3)]
        axes = self.xai1D.mean_spectrum_by_range(y_ranges, excluded_bands=[(400, 500)])
        ax = axes[0, 0]
        wavelengths = np.asarray(self.xai1D.dataset.X_names, dtype=float)
        excluded = (wavelengths >= 400) & (wavelengths <= 500)

        self.assertTrue(ax.lines)
        for line in ax.lines:
            plotted_values = cast(np.ndarray, line.get_ydata())
            self.assertTrue(np.isnan(plotted_values[excluded]).all())
            self.assertTrue(np.isfinite(plotted_values[~excluded]).any())
        plt.close("all")
