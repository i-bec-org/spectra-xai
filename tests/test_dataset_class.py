# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from spectraxai.dataset import Dataset, DatasetSplit
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
        yclass = np.random.randint(2, size=self.nsamples)
        self.datasetY1dim = Dataset(X, y)
        self.datasetY2dim = Dataset(X, Y)
        self.datasetYclass = Dataset(X, yclass)
        self.split_size = 0.6

    def test_constructor(self):
        X_del = np.delete(self.datasetY1dim.X, 1, 0)
        # X and Y do not have the same number of rows
        self.assertRaises(AssertionError, Dataset, X_del, self.datasetY1dim.Y)
        # X is not a 2-D matrix
        self.assertRaises(
            AssertionError,
            Dataset,
            np.array(list(range(self.nsamples))),
            self.datasetY1dim.Y,
        )
        # X_names has wrong size
        self.assertRaises(
            AssertionError,
            Dataset,
            self.datasetY1dim.X,
            self.datasetY1dim.Y,
            ["feature1"],
        )
        # Y_names has wrong size
        self.assertRaises(
            AssertionError,
            Dataset,
            self.datasetY2dim.X,
            self.datasetY2dim.Y,
            Y_names=["output1"],
        )

    def _assert_indices(self, idx_trn, idx_tst):
        self.assertTrue(idx_trn.ndim == idx_tst.ndim == 1)
        self.assertTrue(idx_trn.shape[0] == int(self.split_size * self.nsamples))
        self.assertTrue(idx_tst.shape[0] == int((1 - self.split_size) * self.nsamples))
        self.assertFalse(bool(set(idx_trn).intersection(idx_tst)))

    def _assertions_for_folded_splits(self, idx_trn, idx_tst, N_FOLDS):
        self.assertTrue(idx_trn.shape[0] == idx_tst.shape[0] == N_FOLDS)
        for i in range(N_FOLDS):
            self.assertTrue(idx_trn[i].ndim == idx_tst[i].ndim == 1)
            self.assertFalse(bool(set(idx_trn[i]).intersection(idx_tst[i])))
            self.assertAlmostEqual(
                idx_trn[i].size,
                (N_FOLDS - 1) / N_FOLDS * self.nsamples,
                delta=0.02 * self.nsamples,
            )
            self.assertAlmostEqual(
                idx_tst[i].size, 1 / N_FOLDS * self.nsamples, delta=0.02 * self.nsamples
            )
            self.assertFalse(idx_trn[i].dtype == np.dtype("O"))

    def test_wrong_params(self):
        # Wrong opt param
        splits_a = [DatasetSplit.RANDOM, DatasetSplit.KENNARD_STONE]
        for method in splits_a:
            self.assertRaises(
                TypeError,
                self.datasetY1dim.train_test_split,
                method,
                "str",
            )
            self.assertRaises(TypeError, self.datasetY1dim.train_test_split, method, -2)
            self.assertRaises(
                ValueError, self.datasetY1dim.train_test_split, method, -2.0
            )
            self.assertRaises(
                ValueError, self.datasetY1dim.train_test_split, method, 10.0
            )

        splits_b = [DatasetSplit.CROSS_VALIDATION, DatasetSplit.STRATIFIED]
        for method in splits_b:
            self.assertRaises(
                TypeError,
                self.datasetY1dim.train_test_split,
                method,
                "str",
            )
            self.assertRaises(
                TypeError, self.datasetY1dim.train_test_split, method, 0.2
            )
            self.assertRaises(
                ValueError, self.datasetY1dim.train_test_split, method, -5
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
        # Pass out of bound values to explicit split
        self.assertRaises(
            AssertionError,
            self.datasetY1dim.train_test_split_explicit,
            tst=np.array([-1, 0]),
        )
        self.assertRaises(
            AssertionError,
            self.datasetY1dim.train_test_split_explicit,
            trn=np.array([0, 1, 100000]),
        )

    def test_random_split(self):
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            idx_trn, idx_tst = dataset.train_test_split(
                DatasetSplit.RANDOM, self.split_size
            )
            self._assert_indices(idx_trn, idx_tst)

    def test_ks_split(self):
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            idx_trn, idx_tst = dataset.train_test_split(
                DatasetSplit.KENNARD_STONE, self.split_size
            )
            self._assert_indices(idx_trn, idx_tst)

    def test_explicit_split(self):
        trn = np.random.choice(
            np.arange(self.nsamples),
            int(self.nsamples * self.split_size),
            replace=False,
        )
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            idx_trn, idx_tst = dataset.train_test_split_explicit(trn=trn)
            self._assert_indices(idx_trn, idx_tst)
        tst = np.random.choice(
            np.arange(self.nsamples),
            int(self.nsamples * (1 - self.split_size)),
            replace=False,
        )
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            idx_trn, idx_tst = dataset.train_test_split_explicit(tst=tst)
            self._assert_indices(idx_trn, idx_tst)

    def test_explicit_split_folded(self):
        n_splits = 5
        folds = np.array_split(np.arange(self.nsamples), n_splits)
        indices = np.arange(self.nsamples)
        trn = np.array(
            [
                np.concatenate(
                    [indices[fold] for j, fold in enumerate(folds) if j != i], axis=0
                )
                for i in range(n_splits)
            ],
            dtype=object,
        )
        for dataset in [self.datasetY1dim, self.datasetY2dim]:
            idx_trn, idx_tst = dataset.train_test_split_explicit(trn=trn)
            self._assertions_for_folded_splits(idx_trn, idx_tst, n_splits)
            idx_trn, idx_tst = dataset.train_test_split_explicit(tst=idx_tst)
            self._assertions_for_folded_splits(idx_trn, idx_tst, n_splits)

    def test_folded_splits(self):
        N_FOLDS = 5
        # Cross-validation
        for dataset in [self.datasetY1dim, self.datasetY2dim, self.datasetYclass]:
            idx_trn, idx_tst = dataset.train_test_split(
                DatasetSplit.CROSS_VALIDATION, N_FOLDS
            )
            self._assertions_for_folded_splits(
                idx_trn,
                idx_tst,
                N_FOLDS,
            )
            # Stratified
            idx_trn, idx_tst = self.datasetYclass.train_test_split(
                DatasetSplit.STRATIFIED, N_FOLDS
            )
            self._assertions_for_folded_splits(
                idx_trn,
                idx_tst,
                N_FOLDS,
            )

    def test_subset(self):
        self.assertRaises(ValueError, self.datasetY1dim.subset, np.array([]))
        self.assertRaises(
            ValueError, self.datasetY1dim.subset, np.random.random((2, 2))
        )
        self.assertRaises(
            ValueError, self.datasetY1dim.subset, np.random.randint(1, 5, size=(4, 4))
        )
        self.assertRaises(ValueError, self.datasetY1dim.subset, np.array([-1, 2, 10e7]))
        self.datasetY2dim.subset(np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
