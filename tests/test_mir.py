# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mid-infrared spectra: wavenumbers in cm⁻¹, often stored from high to low.

Uses the packaged soil MIR spectra (`spectraxai.data.load_GB_MIR`) and synthetic
spectra stored in both column orders.

"""

import warnings
from collections.abc import Iterator
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectraxai.data import load_GB_MIR, load_GR_SSL
from spectraxai.dataset import Dataset, DatasetSplit
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.selection import RFE, SelectKBest, VIPSelector
from spectraxai.spectra import Spectra, SpectralPreprocessing as SP

CH_BAND = (2840.0, 2940.0)  # aliphatic C-H stretching
REFERENCE = (2600.0, 2700.0)
WAVENUMBER_LABEL = "Wavenumber (cm$^{-1}$)"


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def mir() -> Spectra:
    return load_GB_MIR()


def _semi_synthetic(mir: Spectra, seed: int = 0) -> np.ndarray:
    """A target driven by the C-H band height above a nearby baseline."""
    w = cast(np.ndarray, mir.wavelengths)
    band = (w >= CH_BAND[0]) & (w <= CH_BAND[1])
    ref = (w >= REFERENCE[0]) & (w <= REFERENCE[1])
    y = mir.X[:, band].mean(axis=1) - mir.X[:, ref].mean(axis=1)
    return y + np.random.default_rng(seed).normal(0, 0.02 * y.std(), y.shape)


@pytest.fixture(scope="module")
def mir_dataset(mir: Spectra) -> Dataset:
    return Dataset(
        mir.X,
        _semi_synthetic(mir),
        X_names=list(cast(np.ndarray, mir.wavelengths)),
        Y_names=["CH_band"],
        spectral_unit="cm-1",
    )


def _synthetic_mir(descending: bool, n: int = 60, seed: int = 0) -> Dataset:
    """Gaussian absorption bands on 4000-400 cm-1, stored in either order."""
    rng = np.random.default_rng(seed)
    w = np.arange(400.0, 4001.0, 8.0)
    centers = np.array([1030.0, 1630.0, 2920.0, 3620.0])
    heights = rng.uniform(0.2, 1.0, size=(n, centers.size))
    X = 0.3 + heights @ np.exp(-(((w[None, :] - centers[:, None]) / 40) ** 2))
    X = X + rng.normal(0, 0.002, X.shape)
    y = heights @ np.array([0.0, 0.5, 2.0, 0.0])
    if descending:
        w, X = w[::-1], X[:, ::-1]
    return Dataset(X, y, X_names=list(w), spectral_unit="cm-1")


# -- The packaged MIR data -------------------------------------------------------------


def test_load_gb_mir(mir: Spectra) -> None:
    w = cast(np.ndarray, mir.wavelengths)
    assert mir.X.shape == (427, 467) and w.shape == (467,)
    assert w[0] > 3990 and w[-1] < 400, "stored from high to low wavenumber"
    assert np.all(np.diff(w) < 0)
    assert np.isfinite(mir.X).all() and mir.X.min() > 0
    subsampled = load_GB_MIR(subsampling=4)
    np.testing.assert_array_equal(cast(np.ndarray, subsampled.wavelengths), w[::4])
    with pytest.raises(ValueError):
        load_GB_MIR(subsampling=0)


# -- Spectral units --------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, expected",
    [("cm-1", "cm-1"), ("cm^-1", "cm-1"), ("1/cm", "cm-1"), ("CM⁻¹", "cm-1")]
    + [("nm", "nm"), ("µm", "um"), (None, None)],
)
def test_spectral_unit_is_normalized(
    mir_dataset: Dataset, unit: str | None, expected: str | None
) -> None:
    dataset = Dataset(mir_dataset.X, mir_dataset.Y, spectral_unit=unit)
    assert dataset.spectral_unit == expected
    assert dataset.is_wavenumber == (expected == "cm-1")


def test_unknown_spectral_unit_raises(mir_dataset: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError):
        Dataset(mir_dataset.X, mir_dataset.Y, spectral_unit="eV")


def test_unit_and_order_survive_dataset_operations(mir_dataset: Dataset) -> None:
    w = cast(np.ndarray, mir_dataset.wavelengths)
    assert np.all(np.diff(w) < 0), "the column order is kept"
    for derived in [
        mir_dataset.subset(np.arange(10)),
        mir_dataset.preprocess(SP.SNV),
        mir_dataset.select_features(np.arange(0, 467, 2)),
    ]:
        assert derived.spectral_unit == "cm-1"
        assert np.all(np.diff(cast(np.ndarray, derived.wavelengths)) < 0)
    assert mir_dataset.spectral_label == WAVENUMBER_LABEL
    assert load_GR_SSL().spectral_label == "Wavelength (nm)"


# -- Pre-processing does not depend on the column order --------------------------------

ORDER_INVARIANT = [
    SP.NONE,
    SP.SNV,
    SP.MSC,
    SP.MEAN_CENTER,
    SP.DETREND,
    SP.CR,
    (SP.SG0, {"window_length": 11, "polyorder": 2}),
    (SP.SG2, {"window_length": 11, "polyorder": 2}),
    [SP.CR, SP.SNV],
]


@pytest.mark.parametrize("method", ORDER_INVARIANT, ids=str)
def test_preprocessing_is_independent_of_column_order(
    mir: Spectra, method: Any
) -> None:
    w = cast(np.ndarray, mir.wavelengths)
    X = mir.X[:40]
    high_to_low = Spectra(X, wavelengths=w).preprocess(method).X
    low_to_high = Spectra(X[:, ::-1], wavelengths=w[::-1]).preprocess(method).X
    np.testing.assert_allclose(high_to_low, low_to_high[:, ::-1], atol=1e-10)


def test_first_derivative_follows_the_column_order(mir: Spectra) -> None:
    # Savitzky-Golay derivatives are taken along the stored columns, so the first
    # derivative changes sign with the order (the second does not, see above)
    w = cast(np.ndarray, mir.wavelengths)
    options = {"window_length": 11, "polyorder": 2}
    a = Spectra(mir.X[:5], wavelengths=w).preprocess((SP.SG1, options)).X
    b = Spectra(mir.X[:5, ::-1], wavelengths=w[::-1]).preprocess((SP.SG1, options)).X
    np.testing.assert_allclose(a, -b[:, ::-1], atol=1e-10)


def test_continuum_removal_regression_for_descending_wavenumbers(
    mir: Spectra,
) -> None:
    # Regression: continuum removal interpolated against decreasing positions
    w = cast(np.ndarray, mir.wavelengths)
    descending = Spectra(mir.X[:3], wavelengths=w).cr().X
    ascending = Spectra(mir.X[:3, ::-1], wavelengths=w[::-1]).cr().X[:, ::-1]
    np.testing.assert_allclose(descending, ascending, atol=1e-12)
    assert np.all(descending <= 1 + 1e-9), "the continuum lies above the spectrum"


def test_preprocessor_passes_wavenumbers_to_continuum_removal(
    mir_dataset: Dataset,
) -> None:
    w = cast(np.ndarray, mir_dataset.wavelengths)
    pre = SpectralPreprocessor([SP.CR, SP.SNV], wavelengths=w)
    np.testing.assert_allclose(
        pre.fit_transform(mir_dataset.X[:10]),
        mir_dataset.subset(np.arange(10)).preprocess([SP.CR, SP.SNV]).X,
    )


# -- Plots draw wavenumbers from high to low -------------------------------------------


def _left_to_right(ax: Any) -> tuple[float, float]:
    return ax.get_xlim()


@pytest.mark.parametrize(
    "descending", [True, False], ids=["high-to-low", "low-to-high"]
)
def test_spectral_plots_are_reversed_and_labelled(descending: bool) -> None:
    dataset = _synthetic_mir(descending)
    pre, post = PreHocAnalysis(dataset), PostHocAnalysis(dataset)
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    model.fit(dataset)
    selector = SelectKBest(k=5).fit(dataset)
    y_ranges = [np.quantile(dataset.Y[:, 0], [0, 0.5, 1]) + [-1, 0, 1]]
    axes = [
        pre.mean_spectrum(),
        pre.mean_spectrum_by_range(y_ranges)[0, 0],
        pre.plot_selected_bands(selector),
        post.plot_importance_spectrum(model)[0],
        post.plot_pls_coefficients(model),
    ]
    for ax in axes:
        left, right = _left_to_right(ax)
        assert left > right, "wavenumbers should decrease from left to right"
        assert ax.get_xlabel() == WAVENUMBER_LABEL
    plt.close("all")


@pytest.mark.parametrize(
    "descending", [True, False], ids=["high-to-low", "low-to-high"]
)
def test_bar_plots_put_the_highest_wavenumber_on_the_left(descending: bool) -> None:
    dataset = _synthetic_mir(descending)
    w = cast(np.ndarray, dataset.wavelengths)
    importance = np.arange(dataset.n_features, dtype=float)
    for ax in [
        PostHocAnalysis(dataset).bar_plot_importance(importance),
        PreHocAnalysis(dataset).bar_plot_importance(FeatureRanking.CORR)[0],
    ]:
        left, right = _left_to_right(ax)
        leftmost = int(np.clip(round(left), 0, len(w) - 1))  # bar at the left edge
        assert w[leftmost] == w.max()
        assert ax.get_xlabel() == WAVENUMBER_LABEL
    plt.close("all")


def test_nir_plots_keep_wavelengths_increasing() -> None:
    nir = load_GR_SSL(subsampling=50)
    ax = PreHocAnalysis(nir).mean_spectrum()
    left, right = _left_to_right(ax)
    assert left < right and ax.get_xlabel() == "Wavelength (nm)"
    ax = PostHocAnalysis(nir).bar_plot_importance(np.ones(nir.n_features))
    assert not ax.xaxis_inverted()


def test_reversal_is_idempotent_and_respects_user_axes(mir_dataset: Dataset) -> None:
    pre = PreHocAnalysis(mir_dataset)
    _, ax = plt.subplots()
    pre.mean_spectrum(ax=ax)
    pre.mean_spectrum(ax=ax)  # drawing twice must not flip it back
    assert ax.xaxis_inverted()


def test_excluded_bands_accept_high_to_low_pairs(mir_dataset: Dataset) -> None:
    # e.g. the atmospheric CO2 band, written as wavenumbers usually are
    ax = PreHocAnalysis(mir_dataset).mean_spectrum(excluded_bands=[(2400, 2280)])
    plotted = np.concatenate([np.asarray(line.get_xdata()) for line in ax.lines])
    assert not np.any((plotted >= 2280) & (plotted <= 2400))
    assert len(ax.lines) == 2


# -- Models, selection and splits on real MIR spectra ----------------------------------


def test_pls_on_mir(mir_dataset: Dataset) -> None:
    idx_trn, _ = mir_dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.75)
    res = StandardModel(Model.PLS, cv=3).fit_and_predict(mir_dataset, idx_trn=idx_trn)
    assert res.loc[0, "R2"] > 0.9
    assert len(res.loc[0, "feature_importance"]) == mir_dataset.n_features


def test_correlation_ranking_finds_the_ch_band(mir_dataset: Dataset) -> None:
    selector = SelectKBest(k=15).fit(mir_dataset)
    w = selector.selected_wavelengths(mir_dataset)
    assert np.mean((w >= 2580) & (w <= 2980)) >= 0.8


@pytest.mark.parametrize(
    "selector",
    [
        VIPSelector(n_components=8),
        RFE(
            StandardModel(Model.PLS, init_hyperparameters={"n_components": 8}),
            n_features_to_select=30,
            step=0.2,
        ),
    ],
    ids=["vip", "rfe"],
)
def test_model_based_selection_on_mir(mir_dataset: Dataset, selector: Any) -> None:
    # MIR bands are strongly collinear, so model-based selection need not pick
    # the C-H band itself; the kept wavenumbers must still predict the target
    trn = np.arange(320)
    reduced = selector.fit(mir_dataset.subset(trn)).transform(mir_dataset)
    assert reduced.spectral_unit == "cm-1"
    np.testing.assert_array_equal(
        reduced.wavelengths, selector.selected_wavelengths(mir_dataset)
    )
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 8})
    res = model.fit_and_predict(reduced, idx_trn=trn)
    assert res.loc[0, "R2"] > 0.9


def test_clhs_on_mir(mir_dataset: Dataset) -> None:
    idx_trn, idx_tst = mir_dataset.train_test_split(
        DatasetSplit.CLHS, 0.7, random_state=0
    )
    assert len(idx_trn) == round(0.7 * mir_dataset.n_samples)
    assert len(np.intersect1d(idx_trn, idx_tst)) == 0
