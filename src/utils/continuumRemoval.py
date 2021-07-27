import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def continuum_removal(spectrum, wvs=[], make_plot=False):
    mat = np.column_stack(
        ([wvs[0] - 1] + wvs + [wvs[-1] + 1], [0] + list(spectrum) + [0])
    )
    hull = ConvexHull(mat)
    ids = np.array(sorted(hull.vertices)[1:-1]) - 1
    cont = np.interp(wvs, np.array(wvs)[ids], spectrum[ids])

    if make_plot:
        plt.plot(mat[:, 0], mat[:, 1], "o")
        for simplex in hull.simplices:
            plt.plot(mat[simplex, 0], mat[simplex, 1], "k-")
        plt.plot(wvs, spectrum / cont, "yo")

    return spectrum / cont
