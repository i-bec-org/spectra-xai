# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.exceptions` defines the library's exceptions."""


class SpectraXAIValueError(ValueError, AssertionError):
    """Raised when an argument passed to a `spectraxai` function is invalid.

    It subclasses both `ValueError` (the idiomatic exception for invalid arguments)
    and `AssertionError` (which earlier versions of `spectraxai` raised), so that
    existing ``except AssertionError`` handlers keep working.

    """
