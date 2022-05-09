from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas
import seaborn
import sage

from spectraxai.dataset import Dataset
from spectraxai.spectra import SpectralPreprocessing


class Explain:
    """
    A general class to provide methods for providing pre-hoc and post-hoc explainability analysis.
    """

    dataset: Dataset
    """A `spectraxai.dataset.Dataset` object used to inspect and/or train the model"""

    def __init__(self, dataset: Dataset):
        """
        Parameters
        ----------

        dataset: `spectraxai.dataset.Dataset`
            The dataset to inspect and/or used to train the model
        """
        self.dataset = dataset
        plt.style.use("seaborn-whitegrid")


class PreHocAnalysis(Explain):
    """
    A class to provide methods for providing pre-hoc explainability analysis.
    """

    def correlogram(self, top: int = 5, method: str = "corr") -> plt.Axes:
        """Plot a correlogram between the most important input features and the output(s).

        Parameters
        ----------
        top: `int`, optional
            The number of most important features to consider. Defaults to 5.
        method: `str`, optional
            The method to calculate the feature importance. Acceptable values
            are "corr" for Pearson's correlation and "mi" for mutual information.
            Defaults to "corr".

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot
        """
        if method not in ["corr", "mi"]:
            raise ValueError("Method may either be corr or mi")
        fig, axes = plt.subplots(
            self.dataset.n_outputs, top, figsize=(11.69 * 2, 8.27), squeeze=False
        )
        metric = self.dataset.corr() if method == "corr" else self.dataset.mi()
        for i, corr in enumerate(metric):
            ind = np.argpartition(np.abs(corr), -top)[-top:]
            ind = ind[np.argsort(corr[ind])]
            for j in range(top):
                x, y = self.dataset.X[:, ind[j]], self.dataset.Y[:, i]
                axes[i, j].scatter(x=x, y=y)
                y_lim = axes[i, j].get_ylim()
                m, b = np.polyfit(x, y, 1)
                axes[i, j].plot(x, m * x + b, c="k")
                axes[i, j].set_ylim(y_lim)
                axes[i, j].set_title("Correlation {0:.2f}".format(corr[ind[j]]))
                axes[i, j].set_xlabel(
                    "Feature {0}".format(self.dataset.X_names[ind[j]])
                )
                axes[i, j].set_ylabel("Output {0}".format(self.dataset.Y_names[i]))
        return axes

    def mean_spectrum_by_range(
        self,
        y_ranges: List,
        preprocesses: List[SpectralPreprocessing] = [],
        ylims: List = [],
    ) -> plt.Axes:
        """
        Creates a plot of the mean spectrum (across all samples) for each output range.

        Parameters
        ----------
        y_ranges: List[np.array]
            A list of length `spectraxai.dataset.n_outputs` containing the ranges to calculate the mean spectrum wrapped in an np.array.
            For example np.array([0, 1, 5]) means calculate the means from outputs 0 to 1 and 1 to 5.

        preprocesses: `List[spectraxai.spectra.SpectralPreprocessing]`, optional
            An optional list of preprocessing techniques to plot simultaneously on the same figure.
            If omitted, it will only plot the spectra of the passed dataset.

        ylims: List[List], optional
            An optional list of the ylim to use on each of the supplied preprocesses.
            If preprocesses was omitted this can be a list of length 1 to act on the dataset's spectra.

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

            ax.set_xlabel("Wavelength")

            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(True, which="major", linewidth=1.0, alpha=0.5)
            ax.grid(True, which="minor", linewidth=0.6, alpha=0.3)

            ax.tick_params(direction="out", which="major", length=6, width=1)
            ax.tick_params(direction="out", which="minor", length=3, width=1)

            plot_ax.legend(
                groupedPerProp.index,
                frameon=True,
                loc="best",
                ncol=int(len(y_ranges[i]) / 4),
                title=y_name,
            )

        return axes


class PostHocAnalysis(Explain):
    """
    A class to provide methods for providing post-hoc explainability analysis.
    """

    def bar_plot_importance(self, importance: List, ax: plt.Axes = None) -> plt.Axes:
        """Plots a bar plot of the feature importance

        Parameters
        ----------
        importance: `List`
            A list containing the importance of each feature

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which case a new figure is created.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        pos = np.arange(len(importance))
        ax.bar(pos, importance)
        ax.set_xticks(pos, self.dataset.X_names)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto"))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Importance")
        return ax

    def circular_bar_plot_importance(
        self, importance: List, ax: plt.Axes = None
    ) -> plt.Axes:
        """Plots a circular (spiral) bar plot of the feature importance

        Parameters
        ----------
        importance: `List`
            A list containing the importance of each feature

        ax: `plt.Axes
            An optional matplotlib axes to plot into. Defaults to None, in which case a new figure is created.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot
        """
        if ax is None:
            plt.figure(figsize=(20, 10))
            ax = plt.subplot(111, polar=True)
            plt.axis("off")
        # Build a dataset
        df = pandas.DataFrame({"Name": self.dataset.X_names, "Value": importance})

        # Reorder the dataframe
        df = df.sort_values(by=["Value"])

        # Constants = parameters controling the plot layout:
        upperLimit = 100
        lowerLimit = 30
        labelPadding = 4

        # Compute max and min in the dataset
        max = df["Value"].max()

        # Let's compute heights: they are a conversion of each item value in those new coordinates
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

    def sage_importance(self, model):
        # Set up an imputer to handle missing features
        imputer = sage.MarginalImputer(model, self.dataset.X)

        # Set up an estimator
        estimator = sage.KernelEstimator(imputer, "mse")

        # Calculate SAGE values
        return estimator(self.dataset.X, self.dataset.Y)
