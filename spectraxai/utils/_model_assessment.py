from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate accuracy metrics for regression.

    Returns a dictionary containing RMSE (Root Mean Squared Error), $R^$ (coefficient
    of determination) and RPIQ (Ratio of Performance to InterQuartile distance).

    Parameters
    ----------
    y_true: `np.ndarray`
        A 1D vector of the true output (also known as observed)

    y_pred: `np.ndarray`
        A 1D vector of the predicted output (e.g., by a model)

    Returns
    -------
    `dict` [`str`, `float`]
        A dictionary containing the name of the accuracy metric and its value.
        Returns RMSE, R2 and RPIQ.

    """
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    return {
        "N": len(y_true),
        "RMSE": RMSE,
        "R2": r2_score(y_true, y_pred),
        "RPIQ": (np.percentile(y_true, 75) - np.percentile(y_true, 25)) / RMSE,
    }


def scatter_plot(y_true, y_pred, ax=None, use_kde=False):
    """Create a scatter plot between observed and predicted values.

    Parameters
    ----------
    y_true: `np.ndarray`
        A 1D vector of the true output (also known as observed)

    y_pred: `np.ndarray`
        A 1D vector of the predicted output (e.g., by a model)

    ax: `plt.Axes`, optional
        An optional matplotlib axes to plot into. Defaults to None, in which
        case a new figure is created.

    use_kde: `bool`, optional
        If True, also plot a kde estimation. Helpful when many samples are present.
        Default value is False.

    Returns
    -------
        `plt.Axes`
            The matplotlib axes with the plot

    """
    if not ax:
        plt.figure()
        ax = plt.gca()
    if use_kde:
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        ax.scatter(y_true, y_pred, c=z, s=2, cmap="YlOrRd_r", rasterized=True)
    else:
        ax.scatter(y_true, y_pred, alpha=0.5)
    y_min = min(min(y_true), min(y_pred))
    y_max = max(max(y_true), max(y_pred))
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.plot([y_min, y_max], [y_min, y_max], ls="--", c=".3")
    m, b = np.polyfit(y_true, y_pred, 1)
    step = (y_max - y_min) / 100
    x_lsq = np.arange(y_min, y_max + step, step)
    ax.plot(x_lsq, [xl * m + b for xl in x_lsq], "-")
    res = metrics(y_true, y_pred)
    ax.set_title(
        "RMSE: {:.4f}, R2: {:.4f}, RPIQ: {:.4f}".format(
            res["RMSE"], res["R2"], res["RPIQ"]
        )
    )
    return ax
