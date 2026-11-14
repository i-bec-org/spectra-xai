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

    def _stylize_plot(self, ax: Axes, index_axis: bool = False) -> Axes:
        """Style an axes whose x axis is spectral.

        ``index_axis`` tells whether x holds feature indices (bar plots) rather
        than wavelengths. Wavenumber axes are drawn from high to low.

        """
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", linewidth=0.6, alpha=0.3)
        ax.tick_params(direction="out", which="major", length=6, width=1)
        ax.tick_params(direction="out", which="minor", length=3, width=1)
        ax.set_xlabel(self.dataset.spectral_label)
        self._orient_spectral_axis(ax, index_axis)
        return ax

    def _orient_spectral_axis(self, ax: Axes, index_axis: bool) -> None:
        """Draw wavenumbers from high to low and wavelengths from low to high.

        Idempotent, so it can be applied more than once to the same axes.

        """
        wavelengths = self.dataset.wavelengths
        if index_axis:
            if wavelengths is None or wavelengths.size < 2:
                return
            # Feature indices follow the column order of the data
            ascending_data = bool(wavelengths[-1] > wavelengths[0])
            inverted = ascending_data == self.dataset.is_wavenumber
        else:
            inverted = self.dataset.is_wavenumber
        if ax.xaxis_inverted() != inverted:
            ax.invert_xaxis()

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
        ax = self._stylize_plot(ax, index_axis=True)
        ax.set_ylabel(ylabel)
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

        bands = np.sort(bands, axis=1)  # accept (high, low) wavenumber pairs
        if not np.isfinite(bands).all():
            raise SpectraXAIValueError(
                "excluded_bands must contain finite (start, end) wavelength pairs"
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

    @_styled_unless("ax")
    def plot_selected_bands(
        self,
        support: Any,
        ax: Axes | None = None,
        excluded_bands: Sequence[tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Shade the selected wavelengths over the mean spectrum ± SD.

        Parameters
        ----------
        support: `np.ndarray` or fitted selector
            A boolean mask (or integer indices) of the selected features, or a
            fitted selector from `spectraxai.selection` (any object with
            ``get_support()``).

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        excluded_bands: sequence of tuple[float, float], optional
            Wavelength intervals to leave blank in the mean spectrum (see
            `PreHocAnalysis.mean_spectrum`).

        **kwargs
            Passed on to `matplotlib.axes.Axes.axvspan` for the shaded bands (e.g.
            ``color`` or ``alpha``).

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if hasattr(support, "get_support"):
            support = support.get_support()
        support = np.asarray(support)
        mask = np.zeros(self.dataset.n_features, dtype=bool)
        if support.dtype == bool:
            if support.shape != (self.dataset.n_features,):
                raise SpectraXAIValueError(
                    "support should have one value per input feature"
                )
            mask = support
        else:
            mask[support.astype(int)] = True
        wavelengths = self._wavelengths()
        ax = self.mean_spectrum(ax=ax, excluded_bands=excluded_bands)

        order = np.argsort(wavelengths)
        wvs, selected = wavelengths[order], mask[order]
        # Each band spans half-way to its neighbours
        mids = (wvs[1:] + wvs[:-1]) / 2
        lower = np.concatenate(
            ([wvs[0] - (mids[0] - wvs[0]) if mids.size else wvs[0] - 0.5], mids)
        )
        upper = np.concatenate(
            (mids, [wvs[-1] + (wvs[-1] - mids[-1]) if mids.size else wvs[-1] + 0.5])
        )
        span_kwargs = {
            "color": "tab:orange",
            "alpha": 0.3,
            "linewidth": 0,
            "label": "Selected",
            **normalize_kwargs(kwargs, Patch),
        }
        idx = np.flatnonzero(selected)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []
        for run in runs:
            ax.axvspan(float(lower[run[0]]), float(upper[run[-1]]), **span_kwargs)
            span_kwargs["label"] = None  # one legend entry
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
            self._stylize_plot(ax[i], index_axis=True)
            if i + 1 < self.dataset.n_outputs:
                ax[i].set_xlabel("")
        return ax

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

        bands = np.sort(bands, axis=1)  # accept (high, low) wavenumber pairs
        if not np.isfinite(bands).all():
            raise SpectraXAIValueError(
                "excluded_bands must contain finite (start, end) wavelength pairs"
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

    def _importance_array(self, importance: Any) -> np.ndarray:
        """Accept an importance array or a fitted `StandardModel`."""
        if hasattr(importance, "feature_importance"):
            values, _ = importance.feature_importance()
            if values is None:
                raise SpectraXAIValueError(
                    "The model has no feature importance of its own; pass an array "
                    "(e.g. from fit_and_predict(..., return_permutation_importance="
                    "True))"
                )
            importance = values
        importance = np.asarray(importance, dtype=float).ravel()
        self._check_bar_array(importance)
        return importance

    @_styled_unless("ax")
    def plot_importance_spectrum(
        self,
        importance: Any,
        ax: Axes | None = None,
        excluded_bands: Sequence[tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> tuple[Axes, Axes]:
        """Overlay the feature importance on the mean spectrum ± SD.

        The mean spectrum is drawn on ``ax`` and the importance on a second y axis
        sharing the wavelength axis, as commonly shown in spectroscopy papers.

        Parameters
        ----------
        importance: `np.ndarray` or fitted `spectraxai.models.StandardModel`
            The importance of each feature (e.g. the ``feature_importance`` column
            of `spectraxai.models.StandardModel.fit_and_predict`), or a fitted
            model whose own importance is used.

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        excluded_bands: sequence of tuple[float, float], optional
            Wavelength intervals to leave blank (see `mean_spectrum`).

        **kwargs
            Passed on to `matplotlib.axes.Axes.plot` for the importance line (e.g.
            ``color``); the area under it is filled with the same color.

        Returns
        -------
        `tuple` of `plt.Axes`
            ``(ax, importance_ax)``: the axes with the mean spectrum and the twin
            axes with the importance.

        """
        importance = self._importance_array(importance)
        wavelengths = self._wavelengths()
        ax = self.mean_spectrum(
            ax=ax, excluded_bands=excluded_bands, color="0.35", label="Mean ± SD"
        )
        ax.set_ylabel("Spectrum")
        importance_ax = ax.twinx()
        importance_ax.grid(False)
        order = np.argsort(wavelengths)
        line_kwargs = {
            "color": "tab:red",
            "linewidth": 1.5,
            "label": "Importance",
            **normalize_kwargs(kwargs, Line2D),
        }
        (line,) = importance_ax.plot(
            wavelengths[order], importance[order], **line_kwargs
        )
        importance_ax.fill_between(
            wavelengths[order],
            0,
            importance[order],
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
        importance_ax.set_ylabel("Importance")
        return ax, importance_ax

    @_styled_unless("ax")
    def plot_pls_coefficients(
        self, model: Any, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Plot the regression coefficients of a fitted PLS model per wavelength.

        Parameters
        ----------
        model: fitted `spectraxai.models.StandardModel` or `PLSRegression`
            A PLS model trained on this dataset's features.

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        **kwargs
            Passed on to `matplotlib.axes.Axes.plot` (one line per output).

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        estimator = getattr(model, "best_model", model)
        coef = getattr(estimator, "coef_", None)
        if coef is None or not hasattr(estimator, "x_weights_"):
            raise SpectraXAIValueError("model should be a fitted PLS model")
        coef = np.atleast_2d(np.asarray(coef, dtype=float))
        if coef.shape[1] != self.dataset.n_features:
            coef = coef.T
        if coef.shape[1] != self.dataset.n_features:
            raise SpectraXAIValueError(
                "The model was not trained on this dataset's features"
            )
        wavelengths = self._wavelengths()
        order = np.argsort(wavelengths)
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        names = (
            self.dataset.Y_names
            if coef.shape[0] == self.dataset.n_outputs
            else [f"Y{i + 1}" for i in range(coef.shape[0])]
        )
        for row, name in zip(coef, names):
            ax.plot(wavelengths[order], row[order], **{"label": name, **kwargs})
        ax.axhline(0, color="0.3", linewidth=0.8)
        self._stylize_plot(ax)
        ax.set_ylabel("PLS coefficient")
        if coef.shape[0] > 1:
            ax.legend(frameon=True)
        return ax

    @_styled_unless("ax")
    def plot_sage_importance(
        self, explanation: Any, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Bar plot of SAGE values with their standard deviation as error bars.

        Parameters
        ----------
        explanation: SAGE `Explanation` or fitted model
            The result of `sage_importance`, or a fitted model to explain (then
            `sage_importance` is called; this can be slow).

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
        if not hasattr(explanation, "values"):
            explanation = self.sage_importance(explanation)
        values = np.asarray(explanation.values, dtype=float).ravel()
        std = getattr(explanation, "std", None)
        yerr = None if std is None else np.asarray(std, dtype=float).ravel()
        ax = self._bar_plot(height=values, yerr=yerr, ax=ax, **kwargs)
        ax.set_ylabel("SAGE value")
        return ax

    def shap_importance(
        self,
        model: Any,
        X: np.ndarray | None = None,
        background: int | np.ndarray = 50,
        max_evals: int | None = None,
    ) -> Any:
        """Calculate SHAP values with the model-agnostic permutation explainer.

        Requires the optional dependency ``shap`` (``pip install spectraxai[shap]``).

        Parameters
        ----------
        model: object
            The fitted model (e.g. a `spectraxai.models.StandardModel`), used through
            its ``predict`` method.

        X: `np.ndarray`, optional
            The samples to explain. Defaults to the dataset's inputs.

        background: `int` or `np.ndarray`, optional
            The background samples that "remove" features, or how many dataset
            samples to draw for it. Defaults to 50.

        max_evals: `int`, optional
            Model evaluations per explained sample. Defaults to
            ``2 * n_features + 1``, the minimum of the permutation explainer.

        Returns
        -------
        `shap.Explanation`
            The SHAP values, of shape (n_samples, n_features) for a single output
            (with a trailing outputs axis for several).

        """
        try:
            import shap
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise ImportError(
                "SHAP is an optional dependency: pip install spectraxai[shap]"
            ) from error
        X = self.dataset.X if X is None else np.asarray(X)
        if isinstance(background, int):
            rng = np.random.default_rng(self.random_state)
            size = min(background, self.dataset.n_samples)
            background = self.dataset.X[
                rng.choice(self.dataset.n_samples, size, replace=False)
            ]
        explainer = shap.Explainer(
            model.predict,
            shap.maskers.Independent(background, max_samples=len(background)),
            algorithm="permutation",
            feature_names=[str(name) for name in self.dataset.X_names],
            seed=self.random_state,
        )
        evals = 2 * self.dataset.n_features + 1 if max_evals is None else max_evals
        return explainer(X, max_evals=evals, silent=True)

    @_styled_unless("ax")
    def plot_shap_summary(
        self,
        explanation: Any,
        ax: Axes | None = None,
        output: int = 0,
        **kwargs: Any,
    ) -> Axes:
        """Bar plot of the mean absolute SHAP value of every wavelength.

        Parameters
        ----------
        explanation: `shap.Explanation` or fitted model
            The result of `shap_importance`, or a fitted model to explain (then
            `shap_importance` is called with its defaults).

        ax: `plt.Axes`, optional
            An optional matplotlib axes to plot into. Defaults to None, in which
            case a new figure is created.

        output: `int`, optional
            Which output to show, for multi-output models. Defaults to 0.

        **kwargs
            Passed on to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        `plt.Axes`
            The matplotlib axes with the plot

        """
        if not hasattr(explanation, "values"):
            explanation = self.shap_importance(explanation)
        values = np.asarray(explanation.values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, output]
        ax = self._bar_plot(height=np.abs(values).mean(axis=0), ax=ax, **kwargs)
        ax.set_ylabel("mean(|SHAP value|)")
        return ax
