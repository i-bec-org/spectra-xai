"""The module `spectraxai.data` loads sample datasets.

Currently the only dataset packaged and supplied with this library is the Greek
part of the GEO-CRADLE SSL, see `load_GR_SSL`.

"""

from importlib import resources

import pandas

from ..dataset import Dataset

DATA_FOLDER = "spectraxai.data"


def load_GR_SSL(subsampling=20, properties=["OM"]):
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
    with resources.path(DATA_FOLDER, "SSL_GR.csv") as p:
        df = pandas.read_csv(p)
        return Dataset(
            X=df.loc[:, "350":"2500":subsampling],
            Y=df.loc[:, properties],
            X_names=range(350, 2501, subsampling),
            Y_names=properties,
        )
