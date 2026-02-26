"""OpenThermoacoustics: A Python thermoacoustic engine and refrigerator design tool."""

__version__ = "0.1.0"

from openthermoacoustics import gas, geometry, segments, solver, validation, viz
from openthermoacoustics.config import (
    ConfigError,
    load_config,
    parse_config,
    run_from_config,
    save_config,
)
from openthermoacoustics.engine import Network
from openthermoacoustics.state import AcousticState

__all__ = [
    "gas",
    "geometry",
    "segments",
    "solver",
    "validation",
    "viz",
    "Network",
    "AcousticState",
    "load_config",
    "save_config",
    "parse_config",
    "run_from_config",
    "ConfigError",
    "__version__",
]
