# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.data` loads sample datasets.

Two datasets are packaged and supplied with this library:

- `spectraxai.data.load_GR_SSL`: the Greek part of the GEO-CRADLE soil spectral
  library, vis–NIR reflectance (350–2500 nm) with soil properties;
- `spectraxai.data.load_GB_MIR`: mid-infrared absorbance spectra (4000–400
  cm⁻¹) of 427 soils of the Countryside Survey of Great Britain 2007, without
  soil properties (spectra only).

"""

from ._datasets import load_GB_MIR, load_GR_SSL

__all__ = ["load_GR_SSL", "load_GB_MIR"]
