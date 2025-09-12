"""
poppy-gw: Gravitational-wave extensions to poppy
"""
from importlib.metadata import PackageNotFoundError, version
import logging

from .poppy_gw import GWPoppy  

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GWPoppy",
]
