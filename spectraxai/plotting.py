# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.plotting` holds the plotting conventions of the library.

Every plotting function in `spectraxai`:

- accepts an ``ax`` (or ``axes`` for multi-panel plots) argument to draw into an
  existing figure, and returns what it drew on;
- never changes global matplotlib state (no ``plt.style.use``, no
  ``tight_layout`` on a figure it did not create);
- forwards extra keyword arguments to its main matplotlib call.

When a function creates the figure itself, it applies the library's `style`. To
use the same look for your own figures, wrap them in the context manager:

```python
import matplotlib.pyplot as plt
from spectraxai.plotting import style

with style():
    fig, axes = plt.subplots(1, 2)
    xai.bar_plot_importance(importance, ax=axes[0])
    xai.mean_spectrum(ax=axes[1])
```

"""

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, ParamSpec, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from spectraxai.exceptions import SpectraXAIValueError

STYLE = "seaborn-v0_8-whitegrid"
"""The matplotlib style applied to figures created by `spectraxai`."""

FIGSIZE = (11.69, 8.27)
"""Default size (A4 landscape, in inches) of figures created by `spectraxai`."""

DPI = 200
"""Default resolution of figures created by `spectraxai`."""


@contextmanager
def style() -> Iterator[None]:
    """Temporarily apply the `spectraxai` matplotlib style.

    Figures and axes created inside the block use the style; global settings are
    restored on exit, except for interactive mode: notebook backends switch it on
    when the first figure is created, and undoing that would stop later figures
    from being displayed.

    """
    interactive: list[bool] = []
    try:
        with plt.style.context(STYLE):
            try:
                yield
            finally:
                interactive.append(mpl.is_interactive())
    finally:
        if interactive:
            mpl.interactive(interactive[0])


def _style_if(active: bool) -> ContextManager[Any]:
    """Return `style` when the library creates the figure, otherwise a no-op."""
    return style() if active else nullcontext()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _styled_unless(param: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Apply `style` to a plotting call unless the caller supplied ``param``.

    A caller that passes its own axes owns the figure, so its style is respected.

    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            supplied = signature.bind_partial(*args, **kwargs).arguments.get(param)
            with _style_if(supplied is None):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _as_axes_array(ax: Axes | np.ndarray | Any, size: int, what: str) -> np.ndarray:
    """Flatten user-supplied axes into a 1-D object array of the expected size."""
    axes = np.asarray([ax] if isinstance(ax, Axes) else ax, dtype=object).ravel()
    if axes.size != size or not all(isinstance(a, Axes) for a in axes):
        raise SpectraXAIValueError(f"Expected {size} matplotlib axes ({what})")
    return axes
