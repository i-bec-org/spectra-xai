import os

import pandas

from ..dataset import Dataset

DATA_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


def load_GR_SSL(subsampling=20, properties=["OM"]):
    df = pandas.read_csv(os.path.join(DATA_FOLDER, "SSL_GR.csv"))
    return Dataset(
        X=df.loc[:, "350":"2500":subsampling],
        Y=df.loc[:, properties],
        X_names=range(350, 2501, subsampling),
        Y_names=properties,
    )
