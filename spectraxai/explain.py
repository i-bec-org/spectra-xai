"""The module `spectraxai.explain` supports the interpretability analysis.

This module defines the class `PreHocAnalysis` to operate only on a given
`spectraxai.dataset.Dataset` and generate some insights solely from the data, as
well as a `PostHocAnalysis` to operate on a `spectraxai.models.StandardModel`
which has been trained on a given `spectraxai.dataset.Dataset`.

"""

from enum import Enum
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, f_regression
import pandas
import seaborn
import sage

from spectraxai.dataset import Dataset
from spectraxai.spectra import SpectralPreprocessing


class FeatureRanking(str, Enum):
    """Types of methods for calculating feature ranking."""

    CORR = "Pearson's correlation"
    MI = "Mutual information"
    F_STATISTIC = "F-statistic"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.value


class _Explain:
    """A general class for providing pre-hoc and post-hoc explainability analysis."""

    dataset: Dataset
    """A `spectraxai.dataset.Dataset` object used to inspect and/or train the model"""

    def __init__(self, dataset: Dataset):
        """Initialize using a dataset.

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            The dataset to inspect and/or used to train the model

        """
        self.dataset = dataset
        plt.style.use("seaborn-whitegrid")

    def _check_bar_array(self, array: np.ndarray):
        if not array.size:
            raise ValueError("The array containing the bar values cannot be empty")
        if array.shape[0] != self.dataset.n_features:
            raise ValueError(
                "The number of bars should equal the number of input features"
            )

    def _stylize_plot(self, ax) -> plt.Axes:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto"))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(which="minor", linewidth=0.6, alpha=0.3)
        ax.tick_params(direction="out", which="major", length=6, width=1)
        ax.tick_params(direction="out", which="minor", length=3, width=1)
        ax.set_xlabel("Wavelength")
        return ax

    def _bar_plot(
        self,
        height: np.ndarray,
        yerr: np.ndarray = None,
        ax: plt.Axes = None,
        ylabel: str = "",
    ) -> plt.Axes:
        """Plot a bar plot of the feature importance.

        Parameters
        ----------
        height: `np.ndarray`
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the importance of each feature

        yerr: `np.ndarray`, optional
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the vertical errorbars to the bar tips

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        ylabel: str, optional
            The label of the y axis

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if ax is None:
            plt.figure(figsize=(11.69, 8.27), dpi=200)
            ax = plt.gca()
        self._check_bar_array(height)
        if yerr is not None:
            self._check_bar_array(yerr)
        pos = np.arange(len(height))
        ax.bar(pos, height, yerr=yerr)
        ax.set_xticks(pos, self.dataset.X_names)
        ax = self._stylize_plot(ax)
        ax.set_ylabel(ylabel)
        return ax


class PreHocAnalysis(_Explain):
    """A class to provide methods for pre-hoc explainability analysis."""

    def _corr(self) -> np.ndarray:
        """Calculate Pearson's correlation between input features and output(s).

        Returns
        -------
        `np.ndarray`
            A 2-D np.array containing the correlation for each output property of size
            (`spectraxai.dataset.Dataset.n_outputs`,
            `spectraxai.dataset.Dataset.n_features`)

        """
        return np.array(
            [
                [
                    np.corrcoef(self.dataset.X[:, i], self.dataset.Y[:, j])[0][1]
                    for i in range(self.dataset.n_features)
                ]
                for j in range(self.dataset.n_outputs)
            ]
        )

    def _mi(self) -> np.ndarray:
        """Calculate the mutual information between all inputs and the output(s).

        Returns
        -------
        `np.ndarray`
            A 2-D np.array containing the mutual information for each output property
            of size (`spectraxai.dataset.Dataset.n_outputs`,
            `spectraxai.dataset.Dataset.n_features`)

        """
        normalize = lambda a: a / np.max(a)
        return np.array(
            [
                normalize(mutual_info_regression(self.dataset.X, self.dataset.Y[:, j]))
                for j in range(self.dataset.n_outputs)
            ]
        )

    def _f_statistic(self) -> np.ndarray:
        """Calculate F-statistic in the `dataset`.

        Performs univariate linear regression tests returning F-statistic between all
        input features and the output(s).

        Returns
        -------
        `np.ndarray`
            A 2-D np.array containing the F-statistic for each output property of size
            (`spectraxai.dataset.Dataset.n_outputs`,
            `spectraxai.dataset.Dataset.n_features`)

        """
        normalize = lambda a: a / np.max(a)
        return np.array(
            [
                normalize(f_regression(self.dataset.X, self.dataset.Y[:, j])[0])
                for j in range(self.dataset.n_outputs)
            ]
        )

    def feature_importance(self, method: FeatureRanking) -> np.ndarray:
        """Calculate feature importance between the input features and the output(s).

        Parameters
        ----------
        method: `FeatureRanking`
            The method to calculate the feature importance.

        Returns
        -------
        `np.ndarray`
            The feature importance according to the selected method, which is a 2-D
            np.array containing the ranking for each output property of size
            (`spectraxai.dataset.Dataset.n_outputs`,
            `spectraxai.dataset.Dataset.n_features`)

        """
        if not isinstance(method, FeatureRanking):
            raise ValueError(
                "Method must be one of the types defined in FeatureRanking"
            )
        if method == FeatureRanking.CORR:
            return self._corr()
        elif method == FeatureRanking.MI:
            return self._mi()
        elif method == FeatureRanking.F_STATISTIC:
            return self._f_statistic()
        else:
            raise NotImplementedError("This type is not yet supported")

    def correlogram(
        self, top: int = 5, method: FeatureRanking = FeatureRanking.CORR
    ) -> plt.Axes:
        """Plot a correlogram between the most important inputs and the output(s).

        Parameters
        ----------
        top: `int`, optional
            The number of most important features to consider. Defaults to 5.

        method: `FeatureRanking`, optional
            The method to calculate the feature importance.
            Defaults to FeatureRanking.CORR.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        metric = self.feature_importance(method)
        fig, axes = plt.subplots(
            self.dataset.n_outputs, top, figsize=(11.69, 8.27), squeeze=False, dpi=200
        )
        for i, corr in enumerate(metric):
            ind = np.argpartition(np.abs(corr), -top)[-top:]
            ind = ind[np.argsort(corr[ind])]
            for j in range(top):
                x, y = self.dataset.X[:, ind[j]], self.dataset.Y[:, i]
                axes[i, j].scatter(x=x, y=y, s=4)
                y_lim = axes[i, j].get_ylim()
                m, b = np.polyfit(x, y, 1)
                axes[i, j].plot(x, m * x + b, c="k")
                axes[i, j].set_ylim(y_lim)
                axes[i, j].set_title("{} {:.2f}".format(method.value, corr[ind[j]]))
                axes[i, j].set_xlabel("Feature {}".format(self.dataset.X_names[ind[j]]))
                axes[i, j].set_ylabel("Output {}".format(self.dataset.Y_names[i]))
        plt.tight_layout()
        return axes

    def bar_plot_importance(
        self,
        method: FeatureRanking = FeatureRanking.CORR,
        ax: Union[plt.Axes, "np.ndarray[plt.Axes]"] = None,
    ):
        """Plot a bar plot of feature ranking.

        The feature ranking is calculated between the input features and the output(s)
        according to the method specified.

        Parameters
        ----------
        method: `FeatureRanking`, optional
            The method to calculate the feature importance.
            Defaults to FeatureRanking.CORR.

        ax: `plt.Axes` or array of `plt.Axes`, optional
            The axes on where to draw the bar plot (for one or more output variables)

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        metric = self.feature_importance(method)
        if ax is None:
            fig, ax = plt.subplots(
                self.dataset.n_outputs,
                1,
                figsize=(11.69, 8.27),
                squeeze=False,
                sharex=True,
                sharey=True,
                dpi=200,
            )
            ax = ax.flatten()
        for i, corr in enumerate(metric):
            self._bar_plot(height=corr, ax=ax[i], ylabel=method.value)
            ax[i].set_title(self.dataset.Y_names[i])
            self._stylize_plot(ax[i])
            if i + 1 < self.dataset.n_outputs:
                ax[i].set_xlabel("")
        return ax

    def mean_spectrum(self, ax=None) -> plt.Axes:
        r"""Plot mean spectrum $\pm$ sd of the dataset.

        Parameters
        ----------
        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None,
            in which case a new figure is created.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if ax is None:
            plt.figure(figsize=(11.69, 8.27), dpi=200)
            ax = plt.gca()

        df = pandas.DataFrame(data=self.dataset.X, columns=self.dataset.X_names)

        ax = seaborn.lineplot(
            data=pandas.melt(df, var_name="Wavelength"),
            x="Wavelength",
            y="value",
            ci="sd",
            ax=ax,
        )
        return self._stylize_plot(ax)

    def mean_spectrum_by_range(
        self,
        y_ranges: List,
        preprocesses: List[SpectralPreprocessing] = [],
        ylims: List = [],
    ) -> plt.Axes:
        """Plot the mean spectrum (across all samples) per output range.

        Parameters
        ----------
        y_ranges: List[np.array]
            A list of length `spectraxai.dataset.n_outputs` containing the ranges to
            calculate the mean spectrum wrapped in an np.array. For example,
            np.array([0, 1, 5]) means calculate the means from [0, 1) and [1, 5].

        preprocesses: `List[spectraxai.spectra.SpectralPreprocessing]`, optional
            An optional list of preprocessing techniques to plot simultaneously on the
            same figure. If omitted, it only plots the spectra of the passed dataset.

        ylims: List[List], optional
            An optional list of the ylim to use on each of the supplied preprocesses.
            If preprocesses was omitted this can be a list of length 1 to act on the
            dataset's spectra.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if len(y_ranges) != self.dataset.n_outputs:
            raise AssertionError(
                "The length of y_ranges should equal the dataset's n_outputs"
            )
        if preprocesses and ylims:
            if len(preprocesses) != len(ylims):
                raise AssertionError(
                    "The length of the preprocesses and ylims lists should be the same"
                )
        if not preprocesses:
            preprocesses = [SpectralPreprocessing.NONE]
        fig, axes = plt.subplots(
            self.dataset.n_outputs,
            1,
            figsize=(11.69, 8.27 * self.dataset.n_outputs),
            dpi=200,
            squeeze=False,
        )
        dfs = []
        for method in preprocesses:
            dataset = self.dataset.preprocess(method)
            dfs.append(
                pandas.DataFrame(
                    np.hstack((dataset.X, dataset.Y)),
                    columns=np.hstack((dataset.X_names, dataset.Y_names)),
                )
            )
        for i in range(self.dataset.n_outputs):
            cmap = seaborn.cubehelix_palette(
                y_ranges[i].shape[0] - 1, start=0.5, rot=-0.75
            )
            ax = axes[i, 0]
            y_name = self.dataset.Y_names[i]

            for j in range(len(preprocesses)):
                groupedPerProp = (
                    dfs[j]
                    .groupby(pandas.cut(dfs[j][y_name], y_ranges[i]))
                    .mean()
                    .drop(columns=self.dataset.Y_names)
                )
                groupedPerProp.index = groupedPerProp.index.astype(str)
                plot_ax = ax if j == 0 else ax.twinx()
                for k in range(len(y_ranges[i]) - 1):
                    x = groupedPerProp.columns
                    y = groupedPerProp.iloc[k].to_numpy()
                    plot_ax.plot(
                        x, y, label=groupedPerProp.index[k], linewidth=2, c=cmap[k]
                    )
                    plot_ax.yaxis.set_major_locator(ticker.LinearLocator(5))
                if j > 0:
                    plot_ax.grid(None)
                if ylims:
                    plot_ax.set_ylim(ylims[j])

            ax = self._stylize_plot(ax)

            plot_ax.legend(
                groupedPerProp.index,
                frameon=True,
                loc="best",
                ncol=int(len(y_ranges[i]) / 4),
                title=y_name,
            )

        return axes


class PostHocAnalysis(_Explain):
    """A class to provide methods for post-hoc explainability analysis."""

    def bar_plot_importance(
        self, importance: np.ndarray, ax: plt.Axes = None
    ) -> plt.Axes:
        """Plot a bar plot of the feature importance.

        Parameters
        ----------
        importance: `np.ndarray`
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the importance of each feature

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None,
            in which case a new figure is created.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        return self._bar_plot(height=importance, ax=None, ylabel="Importance")

    def circular_bar_plot_importance(
        self, importance: np.ndarray, ax: plt.Axes = None, top: int = None
    ) -> plt.Axes:
        """Plot a circular (spiral) bar plot of the feature importance.

        Parameters
        ----------
        importance: `np.ndarray`
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the importance of each feature

        ax: `plt.Axes
            An optional matplotlib axes to plot into. Defaults to None,
            in which case a new figure is created.

        top: `int`, optional
            The number of most important features to consider. Defaults to None, where
            all features all plotted.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if ax is None:
            plt.figure(figsize=(20, 10))
            ax = plt.subplot(111, polar=True)
            plt.axis("off")
        self._check_bar_array(importance)
        # Build a dataset
        df = pandas.DataFrame({"Name": self.dataset.X_names, "Value": importance})

        # Reorder the dataframe
        df = df.sort_values(by=["Value"])

        # Keep only these values
        if top is not None:
            df = df.iloc[-top:]

        # Constants = parameters controlling the plot layout:
        upperLimit = 100
        lowerLimit = 30
        labelPadding = 4

        # Compute max and min in the dataset
        max = df["Value"].max()

        # Heights are a conversion of each item value in those new coordinates
        # In our example, 0 in the dataset will be converted to the lowerLimit (10)
        # The maximum will be converted to the upperLimit (100)
        slope = (upperLimit - lowerLimit) / max
        heights = slope * df.Value + lowerLimit

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2 * np.pi / len(df.index)

        # Compute the angle each bar is centered on:
        indexes = list(range(1, len(df.index) + 1))
        angles = [element * width for element in indexes]

        # Draw bars
        bars = ax.bar(
            x=angles,
            height=heights,
            width=width,
            bottom=lowerLimit,
            linewidth=2,
            edgecolor="white",
            color="#61a4b2",
        )

        # Add labels
        for bar, angle, height, label in zip(bars, angles, heights, df["Name"]):

            # Labels are rotated. Rotation must be specified in degrees :(
            rotation = np.rad2deg(angle)

            # Flip some labels upside down
            alignment = ""
            if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                alignment = "right"
                rotation = rotation + 180
            else:
                alignment = "left"

            # Finally add the labels
            ax.text(
                x=angle,
                y=lowerLimit + bar.get_height() + labelPadding,
                s=label,
                ha=alignment,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
            )

        return ax

    def bar_plot_permutation_importance(
        self, model, dataset: Dataset = None, ax: plt.axes = None
    ):
        """Create a bar plot using permutation feature importance.

        Parameters
        ----------
        model: object
            The estimator that has already been fitted

        dataset: `spectraxai.dataset.Dataset`, optional
            An optional dataset to calculate the scoring, which can be a hold-out set
            different from the training data used to train the estimator. If this is
            not supplied, `dataset` will be used instead.

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if dataset is None:
            dataset = self.dataset
        result = permutation_importance(
            model, dataset.X, dataset.Y, n_repeats=10, n_jobs=-1
        )
        ax = self._bar_plot(height=result.importances_mean, yerr=result.importances_std)
        ax.set_title("Feature importances using permutation")
        ax.set_ylabel("Mean decrease in impurity")
        return ax

    def sage_importance(self, model):
        """Calculate SAGE based importance.

        SAGE (Shapley Additive Global importancE) is a game-theoretic approach for
        understanding black-box machine learning models.

        Parameters
        ----------
        model: object
            The estimator that has already been fitted

        Returns
        -------
        `np.ndarray`
            The feature importance according to SAGE

        """
        # Set up an imputer to handle missing features
        imputer = sage.MarginalImputer(model, self.dataset.X)

        # Set up an estimator
        estimator = sage.KernelEstimator(imputer, "mse")

        # Calculate SAGE values
        return estimator(self.dataset.X, self.dataset.Y)
