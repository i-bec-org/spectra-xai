import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull


def continuum_removal(spectrum: np.ndarray, wvs: list = []):
    assert spectrum.ndim == 1 or spectrum.ndim == 2
    
    if len(wvs) == 0:
        wvs = list(range(len(spectrum)))

    if spectrum.ndim == 2:
        return np.array([continuum_removal(s) for s in spectrum])
    else:
        spectrum = 1 / spectrum # needed for absorbance
        mat = np.column_stack(([wvs[0] - 1] + wvs + [wvs[-1] + 1], [0] + list(spectrum) + [0]))
        hull = ConvexHull(mat)
        ids = np.array(sorted(hull.vertices)[1:-1]) - 1
        cont = np.interp(wvs, np.array(wvs)[ids], spectrum[ids])
        return spectrum / cont

