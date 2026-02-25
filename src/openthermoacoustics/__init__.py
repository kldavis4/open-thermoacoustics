"""OpenThermoacoustics: A Python thermoacoustic engine and refrigerator design tool."""

__version__ = "0.1.0"

from openthermoacoustics import gas, geometry, segments, solver, viz
from openthermoacoustics.engine import Network
from openthermoacoustics.config import (
    load_config,
    save_config,
    parse_config,
    run_from_config,
    ConfigError,
)
from openthermoacoustics.state import AcousticState

__all__ = [
    "gas",
    "geometry",
    "segments",
    "solver",
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
