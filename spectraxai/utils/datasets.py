from importlib import resources

import pandas

from ..dataset import Dataset

DATA_FOLDER = "spectraxai.data"


def load_GR_SSL(subsampling=20, properties=["OM"]):
    df = pandas.read_csv(resources.path(DATA_FOLDER, "SSL_GR.csv"))
    return Dataset(
        X=df.loc[:, "350":"2500":subsampling],
        Y=df.loc[:, properties],
        X_names=range(350, 2501, subsampling),
        Y_names=properties,
    )
