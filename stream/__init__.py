"""STREAM (State-space Temporal REcommendation Adaptive Memory).

This package implements a light-weight reference implementation of the STREAM
framework for adaptive recommendation with state-space overlays. The modules
exposed here provide convenient access to the most frequently used utilities.
"""

from . import config as config
from . import dataio as dataio
from . import metrics as metrics
from . import utils as utils

__all__ = [
    "config",
    "dataio",
    "metrics",
    "utils",
]
