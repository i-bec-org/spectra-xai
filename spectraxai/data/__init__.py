# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.data` loads sample datasets.

Currently the only dataset packaged and supplied with this library is the Greek
part of the GEO-CRADLE SSL, see `spectraxai.data.load_GR_SSL`.

"""

from ._datasets import load_GR_SSL

__all__ = ["load_GR_SSL"]
