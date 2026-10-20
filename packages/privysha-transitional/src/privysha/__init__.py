"""Transitional backward-compatibility module for privysha -> asha."""

from __future__ import annotations

import warnings

warnings.warn(
    "The 'privysha' package has been renamed to 'asha-si'. "
    "Please update your requirements to 'asha-si' and your imports to 'from asha import ...'. "
    "See https://pypi.org/project/asha-si/ for details.",
    FutureWarning,
    stacklevel=2,
)

# Re-export entire public API from asha
try:
    import asha
    from asha import *  # noqa: F401, F403
    from asha import __all__ as _asha_all

    __all__ = list(_asha_all)
except ImportError:  # pragma: no cover
    __all__ = []

__version__ = "0.4.2"
