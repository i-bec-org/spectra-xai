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
        )
