"""The module `spectraxai.data` loads sample datasets.

Currently the only dataset packaged and supplied with this library is the Greek
part of the GEO-CRADLE SSL, see `spectraxai.data.load_GR_SSL`.

"""

from ._datasets import load_GR_SSL

__all__ = ["load_GR_SSL"]
