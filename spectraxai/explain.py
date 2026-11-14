# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.explain` supports the interpretability analysis.

This module defines the class `PreHocAnalysis` to operate only on a given
`spectraxai.dataset.Dataset` and generate some insights solely from the data, as
well as a `PostHocAnalysis` to operate on a `spectraxai.models.StandardModel`
which has been trained on a given `spectraxai.dataset.Dataset`.

"""

from enum import Enum
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cbook import normalize_kwargs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator
from sklearn.utils import Bunch
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, f_regression
import pandas
import seaborn
import sage
from sage.core import Explanation

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.plotting import DPI, FIGSIZE, _as_axes_array, _styled_unless
from spectraxai.spectra import SpectralPreprocessing


class FeatureRanking(str, Enum):
    """Types of methods for calculating feature ranking."""

    CORR = "Pearson's correlation"
    MI = "Mutual information"
    F_STATISTIC = "F-statistic"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.value


class _Explain:
    """A general class for providing pre-hoc and post-hoc explainability analysis."""

    dataset: Dataset
    """A `spectraxai.dataset.Dataset` object used to inspect and/or train the model."""

    random_state: int | None
    """Seed for stochastic analyses (mutual information, permutation importance)."""

    def __init__(self, dataset: Dataset, *, random_state: int | None = None) -> None:
        """Initialize using a dataset.

        Parameters
        ----------
        dataset: `spectraxai.dataset.Dataset`
            The dataset to inspect and/or used to train the model

        random_state: `int`, optional
            Seed for stochastic analyses (mutual information, permutation
            importance). Defaults to None (non-reproducible, as in earlier versions).

        """
        self.dataset = dataset
        self.random_state = random_state

    def _wavelengths(self) -> np.ndarray:
        """The dataset's wavelengths, or an error if they are unknown."""
        wavelengths = self.dataset.wavelengths
        if wavelengths is None:
            raise SpectraXAIValueError(
                "The dataset has no wavelengths: X_names must contain finite numeric "
                "wavelengths, or pass wavelengths= to Dataset"
            )
        return wavelengths

    def _check_bar_array(self, array: np.ndarray) -> None:
        if not array.size:
            raise SpectraXAIValueError(
                "The array containing the bar values cannot be empty"
            )
        if array.shape[0] != self.dataset.n_features:
            raise SpectraXAIValueError(
                "The number of bars should equal the number of input features"
            )

    def _stylize_plot(self, ax: Axes) -> Axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", linewidth=0.6, alpha=0.3)
        ax.tick_params(direction="out", which="major", length=6, width=1)
        ax.tick_params(direction="out", which="minor", length=3, width=1)
        ax.set_xlabel("Wavelength")
        return ax

    def _bar_plot(
        self,
        height: np.ndarray,
        yerr: np.ndarray | None = None,
        ax: Axes | None = None,
        ylabel: str = "",
        **kwargs: Any,
    ) -> Axes:
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

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        self._check_bar_array(height)
        if yerr is not None:
            self._check_bar_array(yerr)
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        pos = np.arange(len(height))
        ax.bar(pos, height, yerr=yerr, **kwargs)
        ax.set_xticks(pos, [str(name) for name in self.dataset.X_names])
        ax = self._stylize_plot(ax)
        ax.set_ylabel(ylabel)
        return ax


def _normalize_by_max(values: np.ndarray) -> np.ndarray:
    """Scale values so that the maximum is 1, leaving all-zero arrays unchanged."""
    peak = np.max(values)
    return values / peak if peak > 0 else values


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
        return np.array(
            [
                _normalize_by_max(
                    mutual_info_regression(
                        self.dataset.X,
                        self.dataset.Y[:, j],
                        random_state=self.random_state,
                    )
                )
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
        return np.array(
            [
                _normalize_by_max(
                    np.nan_to_num(f_regression(self.dataset.X, self.dataset.Y[:, j])[0])
                )
                for j in range(self.dataset.n_outputs)
            ]
        )

    def feature_importance(self, method: object) -> np.ndarray:
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
            raise SpectraXAIValueError(
                "Method must be one of the types defined in FeatureRanking"
            )
        if method == FeatureRanking.CORR:
            return self._corr()
        elif method == FeatureRanking.MI:
            return self._mi()
        elif method == FeatureRanking.F_STATISTIC:
            return self._f_statistic()
        raise SpectraXAIValueError(f"Unsupported feature ranking: {method!r}")

    @_styled_unless("axes")
    def correlogram(
        self,
        top: int = 5,
        method: FeatureRanking = FeatureRanking.CORR,
        *,
        axes: np.ndarray | Axes | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Plot a correlogram between the most important inputs and the output(s).

        Parameters
        ----------
        top: `int`, optional
            The number of most important features to consider. Defaults to 5.

        method: `FeatureRanking`, optional
            The method to calculate the feature importance.
            Defaults to FeatureRanking.CORR.

        axes: array of `plt.Axes`, optional
            ``n_outputs * top`` axes to draw into (any shape; read row by row, one
            row per output). Defaults to None, in which case a new figure is created.

        **kwargs
            Passed on to `matplotlib.axes.Axes.scatter`.

        Returns
        -------
        `np.ndarray`
            A 2-D array of shape (`n_outputs`, `top`) with the matplotlib axes

        """
        if not 1 <= top <= self.dataset.n_features:
            raise SpectraXAIValueError(
                "top should be between 1 and the number of input features"
            )
        metric = self.feature_importance(method)
        created = axes is None
        if axes is None:
            _, axes = plt.subplots(
                self.dataset.n_outputs, top, figsize=FIGSIZE, squeeze=False, dpi=DPI
            )
        else:
            axes = _as_axes_array(
                axes, self.dataset.n_outputs * top, "n_outputs * top"
            ).reshape(self.dataset.n_outputs, top)
        scatter_kwargs = {"s": 4, **kwargs}
        for i, corr in enumerate(metric):
            ind = np.argpartition(np.abs(corr), -top)[-top:]
            ind = ind[np.argsort(corr[ind])]
            for j in range(top):
                x, y = self.dataset.X[:, ind[j]], self.dataset.Y[:, i]
                axes[i, j].scatter(x=x, y=y, **scatter_kwargs)
                y_lim = axes[i, j].get_ylim()
                m, b = np.polyfit(x, y, 1)
                axes[i, j].plot(x, m * x + b, c="k")
                axes[i, j].set_ylim(y_lim)
                axes[i, j].set_title("{} {:.2f}".format(method.value, corr[ind[j]]))
                axes[i, j].set_xlabel("Feature {}".format(self.dataset.X_names[ind[j]]))
                axes[i, j].set_ylabel("Output {}".format(self.dataset.Y_names[i]))
        if created:
            axes[0, 0].get_figure().tight_layout()
        return axes

    @_styled_unless("ax")
    def bar_plot_importance(
        self,
        method: FeatureRanking = FeatureRanking.CORR,
        ax: Axes | np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Plot a bar plot of feature ranking.

        The feature ranking is calculated between the input features and the output(s)
        according to the method specified.

        Parameters
        ----------
        method: `FeatureRanking`, optional
            The method to calculate the feature importance.
            Defaults to FeatureRanking.CORR.

        ax: `plt.Axes` or array of `plt.Axes`, optional
            The axes on where to draw the bar plot, one per output variable (any
            shape). Defaults to None, in which case a new figure is created.

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        `np.ndarray`
            A 1-D array with one matplotlib axes per output

        """
        metric = self.feature_importance(method)
        if ax is None:
            _, ax = plt.subplots(
                self.dataset.n_outputs,
                1,
                figsize=FIGSIZE,
                squeeze=False,
                sharex=True,
                sharey=True,
                dpi=DPI,
            )
            ax = ax.flatten()
        else:
            ax = _as_axes_array(ax, self.dataset.n_outputs, "one per output")
        for i, corr in enumerate(metric):
            self._bar_plot(height=corr, ax=ax[i], ylabel=method.value, **kwargs)
            ax[i].set_title(self.dataset.Y_names[i])
            self._stylize_plot(ax[i])
            if i + 1 < self.dataset.n_outputs:
                ax[i].set_xlabel("")
        return ax

    @_styled_unless("ax")
    def mean_spectrum(
        self,
        ax: Axes | None = None,
        excluded_bands: Sequence[tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> Axes:
        r"""Plot the mean spectrum ± standard deviation of the dataset.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot into. If None, a new figure is created.

        excluded_bands : sequence of tuple[float, float], optional
            Inclusive wavelength intervals to leave blank, such as atmospheric
            absorption bands.

        **kwargs
            Passed on to `matplotlib.axes.Axes.plot` for the mean line (e.g.
            ``color`` or ``label``); the ±SD band uses the same color.

        Returns
        -------
        Axes
            The matplotlib axes with the plot.

        """

        wavelengths = self._wavelengths()

        bands = np.asarray(
            excluded_bands if excluded_bands is not None else [],
            dtype=float,
        ).reshape(-1, 2)

        if not np.isfinite(bands).all() or np.any(bands[:, 0] > bands[:, 1]):
            raise SpectraXAIValueError(
                "excluded_bands must contain finite (start, end) wavelength pairs "
                "with start <= end"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        order = np.argsort(wavelengths)
        wavelengths = wavelengths[order]
        spectra = np.asarray(self.dataset.X)[:, order]

        mean = spectra.mean(axis=0)
        sd = spectra.std(axis=0, ddof=1)

        excluded = np.zeros(wavelengths.shape, dtype=bool)
        for start, end in bands:
            excluded |= (wavelengths >= start) & (wavelengths <= end)
        is_valid = ~excluded

        if is_valid.any():
            # Find indices where validity changes or use split on consecutive indices
            valid_indices = np.flatnonzero(is_valid)
            # Split into arrays of continuous indices (where index difference > 1)
            split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
            segments = np.split(valid_indices, split_points)

            line_kwargs = {"linewidth": 1.5, **normalize_kwargs(kwargs, Line2D)}
            color = line_kwargs.pop("color", None)
            label = line_kwargs.pop("label", None)
            for seg_idx in segments:
                if len(seg_idx) == 0:
                    continue

                seg_x = wavelengths[seg_idx]
                seg_mean = mean[seg_idx]
                seg_lower = seg_mean - sd[seg_idx]
                seg_upper = seg_mean + sd[seg_idx]

                # Plot each continuous segment cleanly
                line_plots = ax.plot(
                    seg_x, seg_mean, color=color, label=label, **line_kwargs
                )
                label = None  # one legend entry for all segments
                if color is None:
                    color = line_plots[0].get_color()

                ax.fill_between(
                    seg_x,
                    seg_lower,
                    seg_upper,
                    color=color,
                    alpha=0.2,
                    edgecolor=None,
                )
        return self._stylize_plot(ax)

    @_styled_unless("axes")
    def mean_spectrum_by_range(
        self,
        y_ranges: Sequence[np.ndarray],
        preprocesses: Sequence[SpectralPreprocessing] | None = None,
        ylims: Sequence[Sequence[float]] | None = None,
        excluded_bands: Sequence[tuple[float, float]] | None = None,
        *,
        axes: np.ndarray | Axes | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
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

        excluded_bands : sequence of tuple[float, float], optional
            Inclusive wavelength intervals to leave blank, such as atmospheric
            absorption bands.

        axes: `plt.Axes` or array of `plt.Axes`, optional
            One axes per output to draw into (any shape). Defaults to None, in which
            case a new figure is created.

        **kwargs
            Passed on to `matplotlib.axes.Axes.plot` for every range's line.

        Returns
        -------
        `np.ndarray`
            A 2-D array of shape (`n_outputs`, 1) with the matplotlib axes

        """
        if len(y_ranges) != self.dataset.n_outputs:
            raise SpectraXAIValueError(
                "The length of y_ranges should equal the dataset's n_outputs"
            )
        preprocesses = [] if preprocesses is None else preprocesses
        ylims = [] if ylims is None else ylims
        if preprocesses and ylims:
            if len(preprocesses) != len(ylims):
                raise SpectraXAIValueError(
                    "The length of the preprocesses and ylims lists should be the same"
                )
        if not preprocesses:
            preprocesses = [SpectralPreprocessing.NONE]

        bands = np.asarray(
            excluded_bands if excluded_bands is not None else [],
            dtype=float,
        ).reshape(-1, 2)

        if not np.isfinite(bands).all() or np.any(bands[:, 0] > bands[:, 1]):
            raise SpectraXAIValueError(
                "excluded_bands must contain finite (start, end) wavelength pairs "
                "with start <= end"
            )

        if axes is None:
            _, axes = plt.subplots(
                self.dataset.n_outputs,
                1,
                figsize=(FIGSIZE[0], FIGSIZE[1] * self.dataset.n_outputs),
                dpi=DPI,
                squeeze=False,
            )
        else:
            axes = _as_axes_array(axes, self.dataset.n_outputs, "one per output")
            axes = axes.reshape(self.dataset.n_outputs, 1)
        line_kwargs = {"linewidth": 2, **normalize_kwargs(kwargs, Line2D)}
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
                    .groupby(pandas.cut(dfs[j][y_name], y_ranges[i].tolist()))
                    .mean()
                    .drop(columns=self.dataset.Y_names)
                )
                groupedPerProp.index = groupedPerProp.index.astype(str)
                plot_ax = ax if j == 0 else ax.twinx()

                wavelengths = self._wavelengths()

                # Compute excluded mask based on bands
                excluded = np.zeros(wavelengths.shape, dtype=bool)
                for start, end in bands:
                    excluded |= (wavelengths >= start) & (wavelengths <= end)

                for k in range(len(y_ranges[i]) - 1):
                    y = groupedPerProp.iloc[k].to_numpy().copy()

                    # Mask out excluded bands and invalid points using NaN so lines
                    # break cleanly without messing up the underlying axis structure
                    # or grid lines.
                    y[excluded | ~np.isfinite(wavelengths)] = np.nan

                    plot_ax.plot(
                        wavelengths,
                        y,
                        label=groupedPerProp.index[k],
                        **{"color": cmap[k], **line_kwargs},
                    )
                    plot_ax.yaxis.set_major_locator(LinearLocator(5))

                if j > 0:
                    plot_ax.grid(None)
                if ylims:
                    plot_ax.set_ylim(ylims[j])

            ax = self._stylize_plot(ax)

            plot_ax.legend(
                groupedPerProp.index,
                frameon=True,
                loc="best",
                ncol=int(len(y_ranges[i]) / 4) if len(y_ranges[i]) >= 4 else 1,
                title=y_name,
            )

        return axes


class PostHocAnalysis(_Explain):
    """A class to provide methods for post-hoc explainability analysis."""

    @_styled_unless("ax")
    def bar_plot_importance(
        self, importance: np.ndarray, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Plot a bar plot of the feature importance.

        Parameters
        ----------
        importance: `np.ndarray`
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the importance of each feature

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None,
            in which case a new figure is created.

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        return self._bar_plot(height=importance, ax=ax, ylabel="Importance", **kwargs)

    @_styled_unless("ax")
    def circular_bar_plot_importance(
        self,
        importance: np.ndarray,
        ax: Axes | None = None,
        top: int | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a circular (spiral) bar plot of the feature importance.

        Parameters
        ----------
        importance: `np.ndarray`
            A numpy array of shape (`spectraxai.dataset.Dataset.n_features`,1)
            containing the importance of each feature

        ax: `plt.Axes`, optional
            An optional matplotlib axes with a polar projection to plot into, e.g.
            ``fig.add_subplot(projection="polar")``. Defaults to None, in which case
            a new figure is created (with the polar frame hidden).

        top: `int`, optional
            The number of most important features to consider. Defaults to None, where
            all features all plotted.

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar` (e.g. ``color``).

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        self._check_bar_array(importance)
        if ax is None:
            fig = plt.figure(figsize=(20, 10))
            ax = cast(Axes, fig.add_subplot(111, polar=True))
            ax.axis("off")
        elif ax.name != "polar":
            raise SpectraXAIValueError(
                "ax should use a polar projection, e.g. "
                "fig.add_subplot(projection='polar')"
            )
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
        bar_kwargs = {
            "linewidth": 2,
            "edgecolor": "white",
            "color": "#61a4b2",
            **normalize_kwargs(kwargs, Patch),
        }
        bars = ax.bar(
            x=angles, height=heights, width=width, bottom=lowerLimit, **bar_kwargs
        )

        # Add labels
        for bar, angle, _, label in zip(bars, angles, heights, df["Name"]):

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

    @_styled_unless("ax")
    def bar_plot_permutation_importance(
        self,
        model: Any,
        dataset: Dataset | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
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

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if dataset is None:
            dataset = self.dataset
        result = cast(
            Bunch,
            permutation_importance(
                model,
                dataset.X,
                dataset.Y,
                n_repeats=10,
                n_jobs=-1,
                random_state=self.random_state,
            ),
        )
        ax = self._bar_plot(
            height=result.importances_mean,
            yerr=result.importances_std,
            ax=ax,
            **kwargs,
        )
        ax.set_title("Feature importances using permutation")
        ax.set_ylabel("Mean decrease in score")
        return ax

    def sage_importance(self, model: Any) -> Explanation:
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
