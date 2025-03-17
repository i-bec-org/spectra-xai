# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from matplotlib import pyplot as plt
from spectraxai.explain import PreHocAnalysis, FeatureRanking
from spectraxai.data import load_GR_SSL


class TestPreHocAnalysisClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.xai1D = PreHocAnalysis(load_GR_SSL())
        self.xai2D = PreHocAnalysis(load_GR_SSL(properties=["OM", "Sand_Fraction"]))

    def test_feature_importance(self):
        for xai in [self.xai1D, self.xai2D]:
            for method in FeatureRanking:
                imp = xai.feature_importance(method)
                self.assertTrue(imp.shape[0] == xai.dataset.n_outputs)
                self.assertTrue(imp.shape[1] == xai.dataset.n_features)

    def test_bar_plot_feature_importance(self):
        for xai in [self.xai1D, self.xai2D]:
            for method in FeatureRanking:
                xai.bar_plot_importance(method)
                fig, ax = plt.subplots(nrows=1, ncols=xai.dataset.n_outputs)
                xai.bar_plot_importance(method, ax=ax)
                plt.close("all")
