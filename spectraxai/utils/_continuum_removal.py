# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.spatial import ConvexHull


def continuum_removal(spectrum: np.ndarray, wvs: list = []):
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
    assert spectrum.ndim == 1 or spectrum.ndim == 2

    if len(wvs) == 0:
        wvs = list(range(len(spectrum)))

    if spectrum.ndim == 2:
        return np.array([continuum_removal(s) for s in spectrum])
    else:
        spectrum = 1 / spectrum  # needed for absorbance
        mat = np.column_stack(
            ([wvs[0] - 1] + wvs + [wvs[-1] + 1], [0] + list(spectrum) + [0])
        )
        hull = ConvexHull(mat)
        ids = np.array(sorted(hull.vertices)[1:-1]) - 1
        cont = np.interp(wvs, np.array(wvs)[ids], spectrum[ids])
        return spectrum / cont
