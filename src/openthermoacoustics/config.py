"""Configuration file support for thermoacoustic network definitions.

This module provides functions for loading and saving thermoacoustic network
configurations from YAML and JSON files. It enables declarative specification
of thermoacoustic systems without writing Python code.

Supported formats:
    - YAML (.yaml, .yml) - requires PyYAML optional dependency
    - JSON (.json) - always available

Example configuration file:

    gas:
      type: helium
      mean_pressure: 3.0e6

    frequency_guess: 84.0

    segments:
      - type: hard_end
      - type: duct
        length: 0.5
        radius: 0.05
      - type: stack
        length: 0.1
        porosity: 0.5
        hydraulic_radius: 0.5e-3
        T_hot: 700
        T_cold: 300
      - type: hard_end

    solver:
      guesses:
        frequency: 84.0
      targets:
        U1_end_real: 0.0
        U1_end_imag: 0.0
      options:
        T_m_start: 300.0
        tol: 1.0e-9
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openthermoacoustics.engine import Network
    from openthermoacoustics.solver.shooting import SolverResult

# Try to import PyYAML for YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]


class ConfigError(Exception):
    """Exception raised for configuration file errors."""

    pass


def load_config(filepath: str) -> "Network":
    """
    Load a thermoacoustic network from a YAML or JSON configuration file.

    Parameters
    ----------
    filepath : str
        Path to the configuration file. File format is determined by extension:
        - .yaml, .yml: YAML format (requires PyYAML)
        - .json: JSON format

    Returns
    -------
    Network
        Configured thermoacoustic network ready for solving.

    Raises
    ------
    ConfigError
        If the file cannot be loaded or parsed.
    FileNotFoundError
        If the file does not exist.
    ImportError
        If YAML file is specified but PyYAML is not installed.

    Examples
    --------
    >>> network = load_config("engine.yaml")
    >>> result = network.solve()
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configuration files. "
                "Install with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ConfigError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )

    if config is None:
        raise ConfigError(f"Configuration file is empty: {filepath}")

    return parse_config(config)


def save_config(network: "Network", filepath: str) -> None:
    """
    Save a thermoacoustic network to a YAML or JSON configuration file.

    Parameters
    ----------
    network : Network
        The thermoacoustic network to save.
    filepath : str
        Path to the output file. Format determined by extension:
        - .yaml, .yml: YAML format (requires PyYAML)
        - .json: JSON format

    Raises
    ------
    ConfigError
        If the network cannot be serialized.
    ImportError
        If YAML file is specified but PyYAML is not installed.

    Examples
    --------
    >>> save_config(network, "engine.yaml")
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    config = _network_to_config(network)

    if suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configuration files. "
                "Install with: pip install pyyaml"
            )
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    else:
        raise ConfigError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )


def parse_config(config: dict[str, Any]) -> "Network":
    """
    Parse a configuration dictionary into a Network.

    Parameters
    ----------
    config : dict
        Configuration dictionary with the following structure:
        - gas: Gas specification (type, mean_pressure)
        - frequency_guess: Initial frequency guess (Hz)
        - segments: List of segment specifications
        - solver: Optional solver settings (guesses, targets, options)

    Returns
    -------
    Network
        Configured thermoacoustic network.

    Raises
    ------
    ConfigError
        If the configuration is invalid or incomplete.

    Examples
    --------
    >>> config = {
    ...     "gas": {"type": "helium", "mean_pressure": 3e6},
    ...     "frequency_guess": 84.0,
    ...     "segments": [
    ...         {"type": "hard_end"},
    ...         {"type": "duct", "length": 0.5, "radius": 0.05},
    ...         {"type": "hard_end"},
    ...     ],
    ... }
    >>> network = parse_config(config)
    """
    # Import here to avoid circular imports
    from openthermoacoustics.engine import Network

    # Parse gas configuration
    gas = _parse_gas(config)

    # Get frequency guess
    frequency_guess = config.get("frequency_guess", 100.0)
    if not isinstance(frequency_guess, (int, float)):
        raise ConfigError(
            f"frequency_guess must be a number, got {type(frequency_guess).__name__}"
        )

    # Create network
    network = Network(gas=gas, frequency_guess=float(frequency_guess))

    # Parse and add segments
    segments_config = config.get("segments", [])
    if not isinstance(segments_config, list):
        raise ConfigError("segments must be a list")

    for i, seg_config in enumerate(segments_config):
        try:
            segment = _parse_segment(seg_config)
            network.add(segment)
        except ConfigError as e:
            raise ConfigError(f"Error in segment {i}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Error creating segment {i}: {e}") from e

    return network


def run_from_config(filepath: str) -> "SolverResult":
    """
    Load a configuration file and solve the thermoacoustic network.

    This is a convenience function that loads a configuration and immediately
    solves it using the solver settings specified in the configuration file.

    Parameters
    ----------
    filepath : str
        Path to the configuration file (YAML or JSON).

    Returns
    -------
    SolverResult
        Results from solving the thermoacoustic network.

    Raises
    ------
    ConfigError
        If the file cannot be loaded, parsed, or solved.

    Examples
    --------
    >>> result = run_from_config("engine.yaml")
    >>> print(f"Frequency: {result.frequency:.2f} Hz")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configuration files. "
                "Install with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ConfigError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )

    if config is None:
        raise ConfigError(f"Configuration file is empty: {filepath}")

    # Parse the network
    network = parse_config(config)

    # Get solver settings
    solver_config = config.get("solver", {})
    guesses = solver_config.get("guesses", {})
    targets = solver_config.get("targets", {})
    options = solver_config.get("options", {})

    # Build solve kwargs from options
    solve_kwargs: dict[str, Any] = {}

    # Map options to solve() parameters
    if "T_m_start" in options:
        solve_kwargs["T_m_start"] = float(options["T_m_start"])
    if "tol" in options:
        solve_kwargs["tol"] = float(options["tol"])
    if "method" in options:
        solve_kwargs["method"] = str(options["method"])
    if "maxiter" in options:
        solve_kwargs["maxiter"] = int(options["maxiter"])
    if "verbose" in options:
        solve_kwargs["verbose"] = bool(options["verbose"])

    # Handle frequency and phase guesses (independent checks)
    if "frequency" in guesses:
        solve_kwargs["frequency"] = float(guesses["frequency"])
    if "p1_phase" in guesses:
        solve_kwargs["p1_phase"] = float(guesses["p1_phase"])

    # Handle pressure amplitude guess
    if "p1_amplitude" in guesses:
        solve_kwargs["p1_amplitude"] = float(guesses["p1_amplitude"])

    # Handle targets - convert to Network.solve() format
    if targets:
        solve_kwargs["targets"] = targets

    # Solve the network
    try:
        result = network.solve(**solve_kwargs)
    except Exception as e:
        raise ConfigError(f"Error solving network: {e}") from e

    return result


def _parse_gas(config: dict[str, Any]) -> Any:
    """
    Parse gas configuration.

    Parameters
    ----------
    config : dict
        Full configuration dictionary containing 'gas' key.

    Returns
    -------
    Gas
        Configured gas instance.

    Raises
    ------
    ConfigError
        If gas configuration is invalid.
    """
    from openthermoacoustics.gas import Air, Argon, Helium, Nitrogen

    gas_config = config.get("gas")
    if gas_config is None:
        raise ConfigError("Missing required 'gas' configuration")

    if not isinstance(gas_config, dict):
        raise ConfigError("gas must be a dictionary")

    gas_type = gas_config.get("type")
    if gas_type is None:
        raise ConfigError("gas.type is required")

    mean_pressure = gas_config.get("mean_pressure")
    if mean_pressure is None:
        raise ConfigError("gas.mean_pressure is required")

    if not isinstance(mean_pressure, (int, float)):
        raise ConfigError(
            f"gas.mean_pressure must be a number, got {type(mean_pressure).__name__}"
        )

    # Map gas type to class
    gas_classes = {
        "helium": Helium,
        "argon": Argon,
        "nitrogen": Nitrogen,
        "air": Air,
    }

    gas_type_lower = gas_type.lower()
    if gas_type_lower not in gas_classes:
        raise ConfigError(
            f"Unknown gas type: {gas_type}. "
            f"Supported types: {', '.join(gas_classes.keys())}"
        )

    return gas_classes[gas_type_lower](mean_pressure=float(mean_pressure))


def _parse_segment(seg_config: dict[str, Any]) -> Any:
    """
    Parse a single segment configuration.

    Parameters
    ----------
    seg_config : dict
        Segment configuration dictionary.

    Returns
    -------
    Segment
        Configured segment instance.

    Raises
    ------
    ConfigError
        If segment configuration is invalid.
    """
    from openthermoacoustics.segments import (
        Compliance,
        Cone,
        Duct,
        HardEnd,
        HeatExchanger,
        Inertance,
        SoftEnd,
        Stack,
    )
    from openthermoacoustics.geometry import CircularPore, ParallelPlate, WireScreen

    if not isinstance(seg_config, dict):
        raise ConfigError("Segment configuration must be a dictionary")

    seg_type = seg_config.get("type")
    if seg_type is None:
        raise ConfigError("Segment type is required")

    seg_type_lower = seg_type.lower()

    # Parse geometry if specified
    geometry = None
    geometry_type = seg_config.get("geometry")
    if geometry_type is not None:
        geometry_map = {
            "circular": CircularPore,
            "circular_pore": CircularPore,
            "parallel_plate": ParallelPlate,
            "screen": WireScreen,
            "wire_screen": WireScreen,
        }
        geometry_type_lower = geometry_type.lower()
        if geometry_type_lower not in geometry_map:
            raise ConfigError(
                f"Unknown geometry type: {geometry_type}. "
                f"Supported types: {', '.join(geometry_map.keys())}"
            )
        geometry = geometry_map[geometry_type_lower]()

    # Get optional name
    name = seg_config.get("name", "")

    # Parse segment based on type
    if seg_type_lower == "hard_end":
        return HardEnd(name=name)

    elif seg_type_lower == "soft_end":
        return SoftEnd(name=name)

    elif seg_type_lower == "duct":
        length = seg_config.get("length")
        radius = seg_config.get("radius")
        if length is None:
            raise ConfigError("duct.length is required")
        if radius is None:
            raise ConfigError("duct.radius is required")
        return Duct(
            length=float(length),
            radius=float(radius),
            geometry=geometry,
            name=name,
        )

    elif seg_type_lower == "cone":
        length = seg_config.get("length")
        radius_in = seg_config.get("radius_in")
        radius_out = seg_config.get("radius_out")
        if length is None:
            raise ConfigError("cone.length is required")
        if radius_in is None:
            raise ConfigError("cone.radius_in is required")
        if radius_out is None:
            raise ConfigError("cone.radius_out is required")
        return Cone(
            length=float(length),
            radius_in=float(radius_in),
            radius_out=float(radius_out),
            geometry=geometry,
            name=name,
        )

    elif seg_type_lower == "stack":
        length = seg_config.get("length")
        porosity = seg_config.get("porosity")
        hydraulic_radius = seg_config.get("hydraulic_radius")
        if length is None:
            raise ConfigError("stack.length is required")
        if porosity is None:
            raise ConfigError("stack.porosity is required")
        if hydraulic_radius is None:
            raise ConfigError("stack.hydraulic_radius is required")

        T_hot = seg_config.get("T_hot")
        T_cold = seg_config.get("T_cold")
        solid_thermal_conductivity = seg_config.get("solid_thermal_conductivity", 0.0)

        kwargs: dict[str, Any] = {
            "length": float(length),
            "porosity": float(porosity),
            "hydraulic_radius": float(hydraulic_radius),
            "geometry": geometry,
            "solid_thermal_conductivity": float(solid_thermal_conductivity),
            "name": name,
        }

        if T_hot is not None:
            kwargs["T_hot"] = float(T_hot)
        if T_cold is not None:
            kwargs["T_cold"] = float(T_cold)

        return Stack(**kwargs)

    elif seg_type_lower == "heat_exchanger":
        length = seg_config.get("length")
        porosity = seg_config.get("porosity")
        hydraulic_radius = seg_config.get("hydraulic_radius")
        temperature = seg_config.get("temperature")
        if length is None:
            raise ConfigError("heat_exchanger.length is required")
        if porosity is None:
            raise ConfigError("heat_exchanger.porosity is required")
        if hydraulic_radius is None:
            raise ConfigError("heat_exchanger.hydraulic_radius is required")
        if temperature is None:
            raise ConfigError("heat_exchanger.temperature is required")
        return HeatExchanger(
            length=float(length),
            porosity=float(porosity),
            hydraulic_radius=float(hydraulic_radius),
            temperature=float(temperature),
            geometry=geometry,
            name=name,
        )

    elif seg_type_lower == "compliance":
        volume = seg_config.get("volume")
        if volume is None:
            raise ConfigError("compliance.volume is required")
        return Compliance(volume=float(volume), name=name)

    elif seg_type_lower == "inertance":
        length = seg_config.get("length")
        radius = seg_config.get("radius")
        area = seg_config.get("area")
        include_resistance = seg_config.get("include_resistance", False)
        if length is None:
            raise ConfigError("inertance.length is required")
        if radius is None and area is None:
            raise ConfigError("inertance requires either radius or area")
        return Inertance(
            length=float(length),
            radius=float(radius) if radius is not None else None,
            area=float(area) if area is not None else None,
            include_resistance=bool(include_resistance),
            name=name,
        )

    else:
        raise ConfigError(
            f"Unknown segment type: {seg_type}. "
            "Supported types: hard_end, soft_end, duct, cone, stack, "
            "heat_exchanger, compliance, inertance"
        )


def _network_to_config(network: "Network") -> dict[str, Any]:
    """
    Convert a Network to a configuration dictionary.

    Parameters
    ----------
    network : Network
        The network to convert.

    Returns
    -------
    dict
        Configuration dictionary that can be serialized to YAML/JSON.

    Raises
    ------
    ConfigError
        If the network contains unsupported segment types.
    """
    from openthermoacoustics.segments import (
        Compliance,
        Cone,
        Duct,
        HardEnd,
        HeatExchanger,
        Inertance,
        SoftEnd,
        Stack,
    )

    config: dict[str, Any] = {}

    # Gas configuration
    gas = network.gas
    gas_type = gas.__class__.__name__.lower()
    config["gas"] = {
        "type": gas_type,
        "mean_pressure": gas.mean_pressure,
    }

    # Frequency guess
    config["frequency_guess"] = network.frequency_guess

    # Segments
    segments_config: list[dict[str, Any]] = []
    for segment in network.segments:
        seg_config = _segment_to_config(segment)
        segments_config.append(seg_config)

    config["segments"] = segments_config

    return config


def _segment_to_config(segment: Any) -> dict[str, Any]:
    """
    Convert a segment to a configuration dictionary.

    Parameters
    ----------
    segment : Segment
        The segment to convert.

    Returns
    -------
    dict
        Configuration dictionary for the segment.

    Raises
    ------
    ConfigError
        If the segment type is not supported.
    """
    from openthermoacoustics.segments import (
        Compliance,
        Cone,
        Duct,
        HardEnd,
        HeatExchanger,
        Inertance,
        SoftEnd,
        Stack,
    )
    from openthermoacoustics.geometry import CircularPore, ParallelPlate, WireScreen

    config: dict[str, Any] = {}

    if isinstance(segment, HardEnd):
        config["type"] = "hard_end"

    elif isinstance(segment, SoftEnd):
        config["type"] = "soft_end"

    elif isinstance(segment, Duct):
        config["type"] = "duct"
        config["length"] = segment.length
        config["radius"] = segment.radius

    elif isinstance(segment, Cone):
        config["type"] = "cone"
        config["length"] = segment.length
        config["radius_in"] = segment.radius_in
        config["radius_out"] = segment.radius_out

    elif isinstance(segment, Stack):
        config["type"] = "stack"
        config["length"] = segment.length
        config["porosity"] = segment.porosity
        config["hydraulic_radius"] = segment.hydraulic_radius
        if segment.T_hot is not None:
            config["T_hot"] = segment.T_hot
        if segment.T_cold is not None:
            config["T_cold"] = segment.T_cold
        if segment.solid_thermal_conductivity != 0.0:
            config["solid_thermal_conductivity"] = segment.solid_thermal_conductivity

    elif isinstance(segment, HeatExchanger):
        config["type"] = "heat_exchanger"
        config["length"] = segment.length
        config["porosity"] = segment.porosity
        config["hydraulic_radius"] = segment.hydraulic_radius
        config["temperature"] = segment.temperature

    elif isinstance(segment, Compliance):
        config["type"] = "compliance"
        config["volume"] = segment.volume

    elif isinstance(segment, Inertance):
        config["type"] = "inertance"
        config["length"] = segment.tube_length
        if segment.radius is not None:
            config["radius"] = segment.radius
        else:
            # Access the area from the base class
            config["area"] = segment._area
        config["include_resistance"] = segment.include_resistance

    else:
        raise ConfigError(
            f"Cannot serialize segment of type: {type(segment).__name__}"
        )

    # Add name if present
    if hasattr(segment, "_name") and segment._name:
        config["name"] = segment._name

    # Add geometry if present
    geometry = getattr(segment, "_geometry", None)
    if geometry is not None:
        if isinstance(geometry, CircularPore):
            config["geometry"] = "circular"
        elif isinstance(geometry, ParallelPlate):
            config["geometry"] = "parallel_plate"
        elif isinstance(geometry, WireScreen):
            config["geometry"] = "screen"

    return config
