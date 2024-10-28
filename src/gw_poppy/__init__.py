"""
bayesian-poppy: Bayesian posterior post-processing in python
"""
from importlib.metadata import PackageNotFoundError, version
import logging
from poppy.backend import set_backend

from .gw_poppy import GWPoppy  

set_backend()

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GWPoppy",
]
