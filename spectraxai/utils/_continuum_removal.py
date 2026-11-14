# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.spatial import ConvexHull
from collections.abc import Sequence


def continuum_removal(
    spectrum: np.ndarray, wvs: Sequence[float] | None = None
) -> np.ndarray:
    """Calculate the continuum removal pre-treatment.

    Parameters
    ----------
    spectrum: `np.ndarray`
        Either a vector or a matrix of the spectra signatures.

    wvs: `list`, optional
        An optional list of the wavelengths

    Returns
    -------
    `np.ndarray`
        Either a vector or a matrix of the pre-processed spectra

    """
    if spectrum.ndim not in (1, 2):
        raise ValueError("spectrum should be a 1-D vector or a 2-D matrix")

    if spectrum.ndim == 2:
        return np.array([continuum_removal(s, wvs) for s in spectrum])

    if wvs is None or len(wvs) == 0:
        wvs = list(range(len(spectrum)))
    else:
        wvs = list(wvs)
        if len(wvs) != len(spectrum):
            raise ValueError("wvs should have the same length as the spectrum")

    # Work on increasing positions (np.interp needs them), e.g. for wavenumbers
    # stored from high to low, and return the result in the original order.
    order = np.argsort(wvs, kind="stable")
    if not np.array_equal(order, np.arange(len(wvs))):
        result = continuum_removal(spectrum[order], [wvs[i] for i in order])
        restored = np.empty_like(result)
        restored[order] = result
        return restored

    spectrum = 1 / spectrum  # needed for absorbance
    mat = np.column_stack(
        ([wvs[0] - 1] + wvs + [wvs[-1] + 1], [0] + list(spectrum) + [0])
    )
    hull = ConvexHull(mat)
    ids = np.array(sorted(hull.vertices)[1:-1]) - 1
    cont = np.interp(wvs, np.array(wvs)[ids], spectrum[ids])
    return spectrum / cont
