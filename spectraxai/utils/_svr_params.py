"""Utility functions for the Support Vector Regression algorithm.

Includes functions to estimate its hyperparameters.

"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def sigest(x, frac=0.5, scale=True):
    r"""Estimate SVM/SVR $\gamma$ based on 0.1 - 0.9 quantile of $||x-x'||^2$.

    Based on sigest function of kernlab. More details may be found here:
    * https://www.rdocumentation.org/packages/kernlab/
    * https://rdrr.io/cran/kernlab/src/R/sigest.R

    Parameters
    ----------
    x: `np.ndarray`
        A 2D numpy array containing the input features
    frac: `float`
        The fraction of the data used to estimate the value
    scale: `bool`
        Whether to scale the input array or not. Defaults to True.

    Returns
    -------
    `np.ndarray`
        A 1D numpy array of exactly 3 suggested sorted values.
    """
    if scale:
        x = (x - x.mean()) / x.std()
    m = np.shape(x)[0]
    n = int(frac * m)
    index1 = np.random.choice(list(range(m)), size=n, replace=True)
    index2 = np.random.choice(list(range(m)), size=n, replace=True)
    temp = x[index1, :] - x[index2, :]
    dist = np.sum(temp**2, axis=1)
    return 1 / np.quantile(dist[np.nonzero(dist)], q=[0.9, 0.5, 0.1])


def estimate_C(y):
    r"""Estimate SVR C from $\max{(|\overline{y}-3\sigma_y|,|\overline{y}+3\sigma_y|)}$.

    Rule-of-thumb from dx.doi.org/10.1016/S0893-6080(03)00169-2, Eq. 13.

    Parameters
    ----------
    y: `np.ndarray`
        A 1-D np array containing the output values

    Returns
    -------
    `float`
        A float value with a suggestion for C

    """
    ymean = y.mean()
    ysigma = y.std()
    return max(abs(ymean - 3 * ysigma), abs(ymean + 3 * ysigma))


def estimate_epsilon(X, Y, ndegree=5, half_features=False):
    r"""Estimate SVR $\epsilon$ based on the noise variance.

    See Equations 17 and 22 of https://dx.doi.org/10.1016/S0893-6080(03)00169-2.

    Parameters
    ----------
    X: `np.ndarray`
        The input features in 2-D format

    Y: `np.ndarray`
        The output vector in 1-D format

    ndegree: `int`
        The degree of the polynomial fit. Defaults to 5.

    half_features: `bool`
        Whether to use half of the features for faster estimations. Defaults to False.

    Returns
    -------
    `float`
        An estimated optimal value for $\epsilon$

    """
    n, m = X.shape[0], X.shape[1]
    # Randomly select some data to use for training
    index1 = np.random.choice(list(range(n)), size=int(n / 5), replace=True)
    index2 = np.setdiff1d(list(range(n)), index1)
    features = np.arange(0, m, 2 if half_features else 1)
    # Create a multivariate polynomial fit
    model = make_pipeline(
        PolynomialFeatures(degree=ndegree), LinearRegression(n_jobs=-1)
    )
    model.fit(X[index1][:, features], Y[index1])
    Yhat = model.predict(X[index2][:, features])
    # Equation 22
    sigmaSquared = np.sum((Y[index2] - Yhat) ** 2) / (n - ndegree)
    # Equation 17
    return 3 * np.sqrt(sigmaSquared * np.log(n) / n)
