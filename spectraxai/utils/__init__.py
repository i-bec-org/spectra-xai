"""The module `spectraxai.utils` defines some utility functions.

In general they are small functions to be used by the other modules.

"""

from ._continuum_removal import continuum_removal
from ._svr_params import sigest, estimate_C, estimate_epsilon
from ._model_assessment import metrics

__all__ = [
    "continuum_removal",
    "sigest",
    "estimate_C",
    "estimate_epsilon",
    "metrics",
]
