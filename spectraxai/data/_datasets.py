# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.data` loads sample datasets.

Currently the only dataset packaged and supplied with this library is the Greek
part of the GEO-CRADLE SSL, see `load_GR_SSL`.

"""

from importlib import resources
from collections.abc import Sequence

import pandas

from ..dataset import Dataset
from ..spectra import Spectra

DATA_FOLDER = "spectraxai.data"


def load_GR_SSL(
    subsampling: int = 20, properties: Sequence[str] | None = None
) -> Dataset:
    """Load the GEO-CRADLE SSL for the Greek soils.

    This dataset is part of the GEO-CRADLE SSL, which may be found
    [here](http://datahub.geocradle.eu/dataset/regional-soil-spectral-library).

    Parameters
    ----------
    subsampling: `int`
        The spectral subsampling to perform; each data point corresponds to 1nm.
        Defaults to 20, i.e. 20 nm.

    properties: `list` [`str`]
        A list of strings containing the property names.
        Defaults to soil organic matter.

    """
    properties = ["OM"] if properties is None else properties
    resource = resources.files(DATA_FOLDER).joinpath("SSL_GR.csv")
    with resource.open("rb") as file:
        df = pandas.read_csv(file)
        return Dataset(
            X=df.loc[:, "350":"2500":subsampling],
            Y=df.loc[:, properties],
            X_names=range(350, 2501, subsampling),
            Y_names=properties,
            spectral_unit="nm",
        )


def load_GB_MIR(subsampling: int = 1) -> Spectra:
    """Load mid-infrared spectra of soils from the Countryside Survey of GB, 2007.

    Diffuse reflectance MIR absorbance spectra of 427 arable and grassland
    topsoils of Great Britain, every 4th point of the original data (about
    7.7 cm⁻¹ apart) from 3996 to 399 cm⁻¹, with the columns ordered from high to
    low wavenumber as MIR spectra are commonly stored. No soil properties are
    included, so this returns spectra only; pair them with your own outputs::

        mir = load_GB_MIR()
        dataset = Dataset(mir.X, y, X_names=list(mir.wavelengths),
                          spectral_unit="cm-1")

    Source: Keith, A.M., Dodd, B.A., Turner, G., Vane, C.H., Wood, C.M. (2020).
    Soil mid-infrared spectroscopy data from arable and grassland in Countryside
    Survey, Great Britain 2007. NERC Environmental Information Data Centre.
    https://doi.org/10.5285/d66ca0a6-403d-4f5a-bef1-8ee177f1e1b3. Contains public
    sector information licensed under the Open Government Licence v3.0.

    Parameters
    ----------
    subsampling: `int`
        Keep every ``subsampling``-th wavenumber of the packaged data. Defaults to
        1 (all 467 wavenumbers).

    Returns
    -------
    `spectraxai.spectra.Spectra`
        The spectra, with their wavenumbers (in cm⁻¹) as ``wavelengths``.

    """
    if not isinstance(subsampling, int) or subsampling < 1:
        raise ValueError("subsampling should be a positive int")
    resource = resources.files(DATA_FOLDER).joinpath("MIR_GB_CS2007.csv")
    with resource.open("rb") as file:
        df = pandas.read_csv(file)
    df = df.iloc[:, ::subsampling]
    return Spectra(
        df.to_numpy(dtype=float), wavelengths=df.columns.to_numpy(dtype=float)
    )
