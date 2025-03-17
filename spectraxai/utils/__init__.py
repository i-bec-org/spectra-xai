# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.utils` defines some utility functions.

In general they are small functions to be used by the other modules.

"""

from ._continuum_removal import continuum_removal
from ._svr_params import sigest, estimate_C, estimate_epsilon
from ._model_assessment import metrics, scatter_plot

__all__ = [
    "continuum_removal",
    "sigest",
    "estimate_C",
    "estimate_epsilon",
    "metrics",
    "scatter_plot",
]
