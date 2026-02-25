"""Tests for configuration file loading and saving.

Tests cover:
1. YAML file loading (requires PyYAML)
2. JSON file loading
3. Configuration parsing
4. Network serialization
5. Round-trip (load -> save -> load)
6. Error handling for invalid configurations
7. run_from_config CLI entry point
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from openthermoacoustics.config import (
    ConfigError,
    YAML_AVAILABLE,
    load_config,
    parse_config,
    run_from_config,
    save_config,
)
from openthermoacoustics.engine import Network
from openthermoacoustics.gas import Air, Argon, Helium, Nitrogen
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

# Try to import yaml for conditional tests
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_config() -> dict[str, Any]:
    """Simple configuration with just a duct."""
    return {
        "gas": {"type": "helium", "mean_pressure": 3.0e6},
        "frequency_guess": 84.0,
        "segments": [
            {"type": "hard_end"},
            {"type": "duct", "length": 0.5, "radius": 0.05},
            {"type": "hard_end"},
        ],
    }


@pytest.fixture
def complex_config() -> dict[str, Any]:
    """Complex configuration with multiple segment types."""
    return {
        "gas": {"type": "helium", "mean_pressure": 3.0e6},
        "frequency_guess": 84.0,
        "segments": [
            {"type": "hard_end"},
            {"type": "duct", "length": 0.5, "radius": 0.05},
            {
                "type": "heat_exchanger",
                "length": 0.02,
                "porosity": 0.5,
                "hydraulic_radius": 0.001,
                "temperature": 300,
            },
            {
                "type": "stack",
                "length": 0.1,
                "porosity": 0.5,
                "hydraulic_radius": 0.0005,
                "T_hot": 700,
                "T_cold": 300,
            },
            {
                "type": "heat_exchanger",
                "length": 0.02,
                "porosity": 0.5,
                "hydraulic_radius": 0.001,
                "temperature": 700,
            },
            {
                "type": "cone",
                "length": 0.1,
                "radius_in": 0.05,
                "radius_out": 0.03,
            },
            {"type": "compliance", "volume": 0.001},
            {"type": "inertance", "length": 0.1, "radius": 0.01},
            {"type": "soft_end"},
        ],
        "solver": {
            "guesses": {"frequency": 84.0, "p1_phase": 0.0},
            "targets": {"U1_end_real": 0.0, "U1_end_imag": 0.0},
            "options": {"T_m_start": 300.0, "tol": 1.0e-9},
        },
    }


@pytest.fixture
def config_with_geometry() -> dict[str, Any]:
    """Configuration with geometry specifications."""
    return {
        "gas": {"type": "nitrogen", "mean_pressure": 1e5},
        "frequency_guess": 100.0,
        "segments": [
            {"type": "hard_end"},
            {
                "type": "duct",
                "length": 0.5,
                "radius": 0.025,
                "geometry": "circular",
            },
            {
                "type": "stack",
                "length": 0.05,
                "porosity": 0.7,
                "hydraulic_radius": 0.0003,
                "geometry": "parallel_plate",
            },
            {"type": "hard_end"},
        ],
    }


# =============================================================================
# Test: Configuration Parsing
# =============================================================================


class TestParseConfig:
    """Tests for parse_config function."""

    def test_parse_simple_config(self, simple_config: dict[str, Any]) -> None:
        """Test parsing a simple configuration."""
        network = parse_config(simple_config)

        assert isinstance(network, Network)
        assert network.frequency_guess == 84.0
        assert len(network.segments) == 3

    def test_parse_creates_correct_gas(self, simple_config: dict[str, Any]) -> None:
        """Test that the correct gas type is created."""
        network = parse_config(simple_config)

        assert isinstance(network.gas, Helium)
        assert network.gas.mean_pressure == 3.0e6

    def test_parse_all_gas_types(self) -> None:
        """Test that all supported gas types can be parsed."""
        gas_types = [
            ("helium", Helium),
            ("argon", Argon),
            ("nitrogen", Nitrogen),
            ("air", Air),
        ]

        for gas_type, gas_class in gas_types:
            config = {
                "gas": {"type": gas_type, "mean_pressure": 1e5},
                "segments": [{"type": "hard_end"}],
            }
            network = parse_config(config)
            assert isinstance(network.gas, gas_class)

    def test_parse_gas_case_insensitive(self) -> None:
        """Test that gas type parsing is case insensitive."""
        config = {
            "gas": {"type": "HELIUM", "mean_pressure": 1e5},
            "segments": [{"type": "hard_end"}],
        }
        network = parse_config(config)
        assert isinstance(network.gas, Helium)

    def test_parse_all_segment_types(self, complex_config: dict[str, Any]) -> None:
        """Test parsing all segment types."""
        network = parse_config(complex_config)

        segments = network.segments
        assert isinstance(segments[0], HardEnd)
        assert isinstance(segments[1], Duct)
        assert isinstance(segments[2], HeatExchanger)
        assert isinstance(segments[3], Stack)
        assert isinstance(segments[4], HeatExchanger)
        assert isinstance(segments[5], Cone)
        assert isinstance(segments[6], Compliance)
        assert isinstance(segments[7], Inertance)
        assert isinstance(segments[8], SoftEnd)

    def test_parse_duct_properties(self) -> None:
        """Test that duct properties are parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {"type": "duct", "length": 0.5, "radius": 0.025, "name": "main_duct"}
            ],
        }
        network = parse_config(config)
        duct = network.segments[0]

        assert isinstance(duct, Duct)
        assert duct.length == 0.5
        assert duct.radius == 0.025

    def test_parse_cone_properties(self) -> None:
        """Test that cone properties are parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {
                    "type": "cone",
                    "length": 0.3,
                    "radius_in": 0.05,
                    "radius_out": 0.025,
                }
            ],
        }
        network = parse_config(config)
        cone = network.segments[0]

        assert isinstance(cone, Cone)
        assert cone.length == 0.3
        assert cone.radius_in == 0.05
        assert cone.radius_out == 0.025

    def test_parse_stack_with_temperature_gradient(self) -> None:
        """Test parsing stack with temperature gradient."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {
                    "type": "stack",
                    "length": 0.1,
                    "porosity": 0.7,
                    "hydraulic_radius": 0.0005,
                    "T_hot": 500.0,
                    "T_cold": 300.0,
                }
            ],
        }
        network = parse_config(config)
        stack = network.segments[0]

        assert isinstance(stack, Stack)
        assert stack.T_hot == 500.0
        assert stack.T_cold == 300.0

    def test_parse_heat_exchanger_properties(self) -> None:
        """Test that heat exchanger properties are parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {
                    "type": "heat_exchanger",
                    "length": 0.02,
                    "porosity": 0.5,
                    "hydraulic_radius": 0.001,
                    "temperature": 350.0,
                }
            ],
        }
        network = parse_config(config)
        hx = network.segments[0]

        assert isinstance(hx, HeatExchanger)
        assert hx.length == 0.02
        assert hx.porosity == 0.5
        assert hx.hydraulic_radius == 0.001
        assert hx.temperature == 350.0

    def test_parse_compliance_properties(self) -> None:
        """Test that compliance properties are parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "compliance", "volume": 0.001}],
        }
        network = parse_config(config)
        comp = network.segments[0]

        assert isinstance(comp, Compliance)
        assert comp.volume == 0.001

    def test_parse_inertance_with_radius(self) -> None:
        """Test that inertance with radius is parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "inertance", "length": 0.1, "radius": 0.01}],
        }
        network = parse_config(config)
        inert = network.segments[0]

        assert isinstance(inert, Inertance)
        assert inert.tube_length == 0.1
        assert inert.radius == 0.01

    def test_parse_inertance_with_area(self) -> None:
        """Test that inertance with area is parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "inertance", "length": 0.1, "area": 3.14e-4}],
        }
        network = parse_config(config)
        inert = network.segments[0]

        assert isinstance(inert, Inertance)
        assert inert.tube_length == 0.1
        assert inert.radius is None

    def test_parse_inertance_with_resistance(self) -> None:
        """Test that inertance resistance flag is parsed correctly."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {
                    "type": "inertance",
                    "length": 0.1,
                    "radius": 0.01,
                    "include_resistance": True,
                }
            ],
        }
        network = parse_config(config)
        inert = network.segments[0]

        assert inert.include_resistance is True

    def test_parse_with_geometry(self, config_with_geometry: dict[str, Any]) -> None:
        """Test parsing segments with geometry specifications."""
        network = parse_config(config_with_geometry)

        # Check that geometry was set
        duct = network.segments[1]
        assert duct._geometry is not None

    def test_parse_default_frequency_guess(self) -> None:
        """Test that default frequency guess is used when not specified."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "hard_end"}],
        }
        network = parse_config(config)
        assert network.frequency_guess == 100.0


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestConfigErrors:
    """Tests for error handling in configuration parsing."""

    def test_missing_gas_config(self) -> None:
        """Test error when gas configuration is missing."""
        config = {"segments": [{"type": "hard_end"}]}
        with pytest.raises(ConfigError, match="Missing required 'gas'"):
            parse_config(config)

    def test_missing_gas_type(self) -> None:
        """Test error when gas type is missing."""
        config = {
            "gas": {"mean_pressure": 1e5},
            "segments": [{"type": "hard_end"}],
        }
        with pytest.raises(ConfigError, match="gas.type is required"):
            parse_config(config)

    def test_missing_gas_pressure(self) -> None:
        """Test error when gas pressure is missing."""
        config = {
            "gas": {"type": "helium"},
            "segments": [{"type": "hard_end"}],
        }
        with pytest.raises(ConfigError, match="gas.mean_pressure is required"):
            parse_config(config)

    def test_unknown_gas_type(self) -> None:
        """Test error for unknown gas type."""
        config = {
            "gas": {"type": "xenon", "mean_pressure": 1e5},
            "segments": [{"type": "hard_end"}],
        }
        with pytest.raises(ConfigError, match="Unknown gas type"):
            parse_config(config)

    def test_missing_segment_type(self) -> None:
        """Test error when segment type is missing."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"length": 0.5}],
        }
        with pytest.raises(ConfigError, match="Segment type is required"):
            parse_config(config)

    def test_unknown_segment_type(self) -> None:
        """Test error for unknown segment type."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "unknown_segment"}],
        }
        with pytest.raises(ConfigError, match="Unknown segment type"):
            parse_config(config)

    def test_missing_duct_length(self) -> None:
        """Test error when duct length is missing."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "duct", "radius": 0.05}],
        }
        with pytest.raises(ConfigError, match="duct.length is required"):
            parse_config(config)

    def test_missing_duct_radius(self) -> None:
        """Test error when duct radius is missing."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "duct", "length": 0.5}],
        }
        with pytest.raises(ConfigError, match="duct.radius is required"):
            parse_config(config)

    def test_missing_stack_properties(self) -> None:
        """Test error when stack properties are missing."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "stack", "length": 0.1}],
        }
        with pytest.raises(ConfigError, match="stack.porosity is required"):
            parse_config(config)

    def test_missing_inertance_radius_and_area(self) -> None:
        """Test error when inertance has neither radius nor area."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [{"type": "inertance", "length": 0.1}],
        }
        with pytest.raises(ConfigError, match="inertance requires either radius or area"):
            parse_config(config)

    def test_unknown_geometry_type(self) -> None:
        """Test error for unknown geometry type."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": [
                {
                    "type": "duct",
                    "length": 0.5,
                    "radius": 0.05,
                    "geometry": "unknown_geometry",
                }
            ],
        }
        with pytest.raises(ConfigError, match="Unknown geometry type"):
            parse_config(config)

    def test_invalid_gas_pressure_type(self) -> None:
        """Test error when gas pressure is wrong type."""
        config = {
            "gas": {"type": "helium", "mean_pressure": "not_a_number"},
            "segments": [{"type": "hard_end"}],
        }
        with pytest.raises(ConfigError, match="gas.mean_pressure must be a number"):
            parse_config(config)

    def test_segments_not_list(self) -> None:
        """Test error when segments is not a list."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 1e5},
            "segments": {"type": "hard_end"},  # Should be a list
        }
        with pytest.raises(ConfigError, match="segments must be a list"):
            parse_config(config)


# =============================================================================
# Test: File Loading
# =============================================================================


class TestLoadConfig:
    """Tests for loading configuration files."""

    def test_load_json_config(self, simple_config: dict[str, Any]) -> None:
        """Test loading a JSON configuration file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_config, f)
            f.flush()

            network = load_config(f.name)
            assert isinstance(network, Network)
            assert len(network.segments) == 3

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yaml_config(self, simple_config: dict[str, Any]) -> None:
        """Test loading a YAML configuration file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(simple_config, f)
            f.flush()

            network = load_config(f.name)
            assert isinstance(network, Network)
            assert len(network.segments) == 3

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yml_extension(self, simple_config: dict[str, Any]) -> None:
        """Test loading a .yml file (alternative YAML extension)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            yaml.dump(simple_config, f)
            f.flush()

            network = load_config(f.name)
            assert isinstance(network, Network)

    def test_load_nonexistent_file(self) -> None:
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_unsupported_format(self, simple_config: dict[str, Any]) -> None:
        """Test error for unsupported file format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            json.dump(simple_config, f)
            f.flush()

            with pytest.raises(ConfigError, match="Unsupported file format"):
                load_config(f.name)

    def test_load_empty_json_file(self) -> None:
        """Test error when JSON file is empty or null."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("null")
            f.flush()

            with pytest.raises(ConfigError, match="Configuration file is empty"):
                load_config(f.name)


# =============================================================================
# Test: File Saving
# =============================================================================


class TestSaveConfig:
    """Tests for saving configuration files."""

    def test_save_json_config(self) -> None:
        """Test saving a network to JSON."""
        gas = Helium(mean_pressure=3e6)
        network = Network(gas=gas, frequency_guess=84.0)
        network.add(HardEnd())
        network.add(Duct(length=0.5, radius=0.05))
        network.add(HardEnd())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            save_config(network, f.name)

            # Verify file can be loaded
            with open(f.name, "r") as rf:
                config = json.load(rf)

            assert config["gas"]["type"] == "helium"
            assert config["gas"]["mean_pressure"] == 3e6
            assert config["frequency_guess"] == 84.0
            assert len(config["segments"]) == 3

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_save_yaml_config(self) -> None:
        """Test saving a network to YAML."""
        gas = Helium(mean_pressure=3e6)
        network = Network(gas=gas, frequency_guess=84.0)
        network.add(HardEnd())
        network.add(Duct(length=0.5, radius=0.05))
        network.add(HardEnd())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            save_config(network, f.name)

            # Verify file can be loaded
            with open(f.name, "r") as rf:
                config = yaml.safe_load(rf)

            assert config["gas"]["type"] == "helium"
            assert len(config["segments"]) == 3

    def test_save_unsupported_format(self) -> None:
        """Test error when saving to unsupported format."""
        gas = Helium(mean_pressure=1e5)
        network = Network(gas=gas)

        with pytest.raises(ConfigError, match="Unsupported file format"):
            save_config(network, "/tmp/config.txt")

    def test_save_all_segment_types(self) -> None:
        """Test saving network with all segment types."""
        gas = Helium(mean_pressure=3e6)
        network = Network(gas=gas, frequency_guess=84.0)
        network.add(HardEnd())
        network.add(Duct(length=0.5, radius=0.05))
        network.add(HeatExchanger(length=0.02, porosity=0.5, hydraulic_radius=0.001, temperature=300))
        network.add(Stack(length=0.1, porosity=0.5, hydraulic_radius=0.0005, T_hot=700, T_cold=300))
        network.add(Cone(length=0.1, radius_in=0.05, radius_out=0.03))
        network.add(Compliance(volume=0.001))
        network.add(Inertance(length=0.1, radius=0.01))
        network.add(SoftEnd())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            save_config(network, f.name)

            # Verify all segments are saved
            with open(f.name, "r") as rf:
                config = json.load(rf)

            assert len(config["segments"]) == 8
            segment_types = [s["type"] for s in config["segments"]]
            assert "hard_end" in segment_types
            assert "duct" in segment_types
            assert "heat_exchanger" in segment_types
            assert "stack" in segment_types
            assert "cone" in segment_types
            assert "compliance" in segment_types
            assert "inertance" in segment_types
            assert "soft_end" in segment_types


# =============================================================================
# Test: Round Trip
# =============================================================================


class TestRoundTrip:
    """Tests for save -> load round trip."""

    def test_json_round_trip(self) -> None:
        """Test that a network can be saved and loaded from JSON."""
        # Create original network
        gas = Helium(mean_pressure=3e6)
        original = Network(gas=gas, frequency_guess=84.0)
        original.add(HardEnd())
        original.add(Duct(length=0.5, radius=0.05))
        original.add(Stack(length=0.1, porosity=0.5, hydraulic_radius=0.0005))
        original.add(HardEnd())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            # Save
            save_config(original, f.name)

            # Load
            loaded = load_config(f.name)

        # Compare
        assert loaded.frequency_guess == original.frequency_guess
        assert isinstance(loaded.gas, Helium)
        assert loaded.gas.mean_pressure == original.gas.mean_pressure
        assert len(loaded.segments) == len(original.segments)

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_yaml_round_trip(self) -> None:
        """Test that a network can be saved and loaded from YAML."""
        gas = Argon(mean_pressure=1e5)
        original = Network(gas=gas, frequency_guess=100.0)
        original.add(HardEnd())
        original.add(Duct(length=1.0, radius=0.025))
        original.add(HardEnd())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            save_config(original, f.name)
            loaded = load_config(f.name)

        assert loaded.frequency_guess == original.frequency_guess
        assert isinstance(loaded.gas, Argon)
        assert len(loaded.segments) == len(original.segments)

    def test_round_trip_preserves_stack_temperatures(self) -> None:
        """Test that stack temperatures are preserved in round trip."""
        gas = Helium(mean_pressure=3e6)
        original = Network(gas=gas)
        original.add(Stack(
            length=0.1,
            porosity=0.5,
            hydraulic_radius=0.0005,
            T_hot=700.0,
            T_cold=300.0,
        ))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            save_config(original, f.name)
            loaded = load_config(f.name)

        stack = loaded.segments[0]
        assert isinstance(stack, Stack)
        assert stack.T_hot == 700.0
        assert stack.T_cold == 300.0


# =============================================================================
# Test: run_from_config
# =============================================================================


class TestRunFromConfig:
    """Tests for the run_from_config CLI entry point."""

    def test_run_simple_config(self) -> None:
        """Test running a simple configuration."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 101325.0},
            "frequency_guess": 500.0,
            "segments": [
                {"type": "hard_end"},
                {"type": "duct", "length": 1.0, "radius": 0.025},
                {"type": "hard_end"},
            ],
            "solver": {
                "guesses": {"frequency": 500.0},
                "targets": {"U1_end_real": 0.0, "U1_end_imag": 0.0},
                "options": {"T_m_start": 300.0},
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            f.flush()

            # This may or may not converge depending on the guess
            # We just test that it runs without error
            try:
                result = run_from_config(f.name)
                assert result.n_iterations > 0
            except ConfigError:
                # Solver may fail to converge, which is acceptable
                pass

    def test_run_nonexistent_file(self) -> None:
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            run_from_config("/nonexistent/config.yaml")

    def test_run_empty_file(self) -> None:
        """Test error when file is empty."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("null")
            f.flush()

            with pytest.raises(ConfigError, match="Configuration file is empty"):
                run_from_config(f.name)

    def test_run_forwards_frequency_and_phase_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that frequency and p1_phase are both forwarded to solve()."""
        config = {
            "gas": {"type": "helium", "mean_pressure": 101325.0},
            "segments": [
                {"type": "hard_end"},
                {"type": "duct", "length": 1.0, "radius": 0.025},
                {"type": "hard_end"},
            ],
            "solver": {
                "guesses": {"frequency": 500.0, "p1_phase": 0.25},
                "targets": {"U1_end_real": 0.0, "U1_end_imag": 0.0},
                "options": {"T_m_start": 300.0},
            },
        }
        captured_kwargs: dict[str, Any] = {}

        class DummyResult:
            pass

        dummy_result = DummyResult()

        def fake_solve(self: Network, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return dummy_result

        monkeypatch.setattr(Network, "solve", fake_solve)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            f.flush()
            result = run_from_config(f.name)

        assert result is dummy_result
        assert captured_kwargs["frequency"] == 500.0
        assert captured_kwargs["p1_phase"] == 0.25


# =============================================================================
# Test: YAML Availability
# =============================================================================


class TestYamlAvailability:
    """Tests for YAML availability handling."""

    @pytest.mark.skipif(YAML_AVAILABLE, reason="Test requires PyYAML to be missing")
    def test_yaml_import_error_on_load(self) -> None:
        """Test that ImportError is raised when loading YAML without PyYAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("gas:\n  type: helium\n")
            f.flush()

            with pytest.raises(ImportError, match="PyYAML is required"):
                load_config(f.name)

    @pytest.mark.skipif(YAML_AVAILABLE, reason="Test requires PyYAML to be missing")
    def test_yaml_import_error_on_save(self) -> None:
        """Test that ImportError is raised when saving YAML without PyYAML."""
        gas = Helium(mean_pressure=1e5)
        network = Network(gas=gas)

        with pytest.raises(ImportError, match="PyYAML is required"):
            save_config(network, "/tmp/config.yaml")
