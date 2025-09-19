"""
aspire-gw: Gravitational-wave extensions to aspire
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .gw_aspire import GWAspire

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GWAspire",
]
