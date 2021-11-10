import numpy as np
from itertools import chain
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import indexable, _safe_indexing
from sklearn.preprocessing import StandardScaler


class KFold(_BaseKFold):
    def __init__(self, n_splits=5, **kwargs):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        del self.shuffle
        del self.random_state

    def _iter_test_indices(self, X=None, y=None, groups=None):
        n_samples = _num_samples(X)

        _ks = _KennardStone()
        indices = _ks._get_indexes(X)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class KSSplit(BaseShuffleSplit):
    @_deprecate_positional_args
    def __init__(self, n_splits=10, *, test_size=None, train_size=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, groups=None):
        _ks = _KennardStone()
        inds = _ks._get_indexes(X)

        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        for _ in range(self.n_splits):
            ind_test = inds[:n_test]
            ind_train = inds[n_test : (n_test + n_train)]
            yield ind_train, ind_test


def train_test_split(*arrays, test_size=None, train_size=None, **kwargs):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    CVClass = KSSplit
    cv = CVClass(test_size=n_test, train_size=n_train)

    train, test = next(cv.split(X=arrays[0]))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


class _KennardStone:
    def __init__(self, scale=True, prior="test"):
        self.scale = scale
        self.prior = prior

    def _get_indexes(self, X):
        X = np.array(X)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self._original_X = X.copy()

        distance_to_ave = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
        i_farthest = np.argmax(distance_to_ave)
        i_selected = [i_farthest]
        i_remaining = np.arange(len(X))

        X = np.delete(X, i_selected, axis=0)
        i_remaining = np.delete(i_remaining, i_selected, axis=0)
        indexes = self._sort(X, i_selected, i_remaining)

        if self.prior == "test":
            return list(reversed(indexes))
        elif self.prior == "train":
            return indexes
        else:
            raise NotImplementedError

    def _sort(self, X, i_selected, i_remaining):
        samples_selected = self._original_X[i_selected]

        min_distance_to_samples_selected = np.min(
            np.sum(
                (np.expand_dims(samples_selected, 1) - np.expand_dims(X, 0)) ** 2,
                axis=2,
            ),
            axis=0,
        )

        i_farthest = np.argmax(min_distance_to_samples_selected)
        i_selected.append(i_remaining[i_farthest])

        X = np.delete(X, i_farthest, axis=0)
        i_remaining = np.delete(i_remaining, i_farthest, 0)

        if len(i_remaining):
            return self._sort(X, i_selected, i_remaining)
        else:
            return i_selected
