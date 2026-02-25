"""Acoustic state accessor for convenient access to derived quantities.

This module provides the AcousticState class, which serves as the Python
equivalent of reference baseline's RPN segment. Instead of Reverse Polish Notation
expressions, users can access derived quantities through Pythonic properties
and methods.

Example
-------
>>> from openthermoacoustics import AcousticState, gas
>>> helium = gas.Helium(mean_pressure=1e6)
>>> state = AcousticState(
...     p1=50000 + 10000j,
...     U1=1e-4 + 2e-5j,
...     T_m=300.0,
...     omega=2 * np.pi * 100,
...     gas=helium
... )
>>> print(f"Density: {state.rho:.3f} kg/m³")
>>> print(f"Sound speed: {state.a:.1f} m/s")
>>> print(f"Acoustic power: {state.acoustic_power:.3f} W")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from openthermoacoustics.utils import (
    penetration_depth_thermal,
    penetration_depth_viscous,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.geometry import Geometry


@dataclass
class AcousticState:
    """
    Accessor for acoustic state variables and derived quantities.

    This class provides convenient access to gas properties, penetration depths,
    and derived acoustic quantities at a specific location in a thermoacoustic
    system. It serves as the Python equivalent of reference baseline's RPN segment.

    Parameters
    ----------
    p1 : complex
        Complex pressure amplitude (Pa).
    U1 : complex
        Complex volumetric velocity amplitude (m³/s).
    T_m : float
        Mean temperature (K).
    omega : float
        Angular frequency (rad/s).
    gas : Gas
        Gas object providing thermophysical properties.
    p_m : float, optional
        Mean pressure (Pa). If not provided, uses gas.mean_pressure.
    x : float, optional
        Position along the acoustic network (m). Default 0.
    area : float, optional
        Cross-sectional area (m²). Used for velocity and displacement calculations.

    Attributes
    ----------
    p1 : complex
        Complex pressure amplitude (Pa).
    U1 : complex
        Complex volumetric velocity amplitude (m³/s).
    T_m : float
        Mean temperature (K).
    omega : float
        Angular frequency (rad/s).
    gas : Gas
        Gas object.
    p_m : float
        Mean pressure (Pa).
    x : float
        Position (m).
    area : float or None
        Cross-sectional area (m²).

    Examples
    --------
    Basic usage:

    >>> state = AcousticState(p1=50000+0j, U1=1e-4+0j, T_m=300, omega=628, gas=helium)
    >>> state.rho  # density
    1.616...
    >>> state.a  # sound speed
    1019.3...
    >>> state.acoustic_power  # Re[p1 * conj(U1)] / 2
    2.5

    With area for velocity calculations:

    >>> state = AcousticState(p1=50000+0j, U1=1e-4+0j, T_m=300, omega=628,
    ...                       gas=helium, area=1e-3)
    >>> state.u1  # velocity amplitude
    0.1+0j
    >>> state.u1_mag  # |u1|
    0.1
    """

    p1: complex
    U1: complex
    T_m: float
    omega: float
    gas: Gas
    p_m: float = None
    x: float = 0.0
    area: float = None

    def __post_init__(self):
        """Initialize derived quantities."""
        if self.p_m is None:
            self.p_m = self.gas.mean_pressure

    # =========================================================================
    # Gas Properties
    # =========================================================================

    @property
    def rho(self) -> float:
        """Mean density (kg/m³)."""
        return self.gas.density(self.T_m)

    @property
    def cp(self) -> float:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self.gas.specific_heat_cp(self.T_m)

    @property
    def cv(self) -> float:
        """Specific heat at constant volume (J/(kg·K))."""
        return self.gas.specific_heat_cv(self.T_m)

    @property
    def gamma(self) -> float:
        """Ratio of specific heats cp/cv."""
        return self.gas.gamma(self.T_m)

    @property
    def a(self) -> float:
        """Sound speed (m/s)."""
        return self.gas.sound_speed(self.T_m)

    @property
    def mu(self) -> float:
        """Dynamic viscosity (Pa·s)."""
        return self.gas.viscosity(self.T_m)

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m·K))."""
        return self.gas.thermal_conductivity(self.T_m)

    @property
    def Pr(self) -> float:
        """Prandtl number (dimensionless)."""
        return self.gas.prandtl(self.T_m)

    @property
    def beta(self) -> float:
        """Thermal expansion coefficient (1/K). For ideal gas, β = 1/T."""
        return 1.0 / self.T_m

    # =========================================================================
    # Frequency and Wavelength
    # =========================================================================

    @property
    def f(self) -> float:
        """Frequency (Hz)."""
        return self.omega / (2 * np.pi)

    @property
    def wavelength(self) -> float:
        """Acoustic wavelength (m)."""
        return self.a / self.f

    @property
    def wavenumber(self) -> float:
        """Acoustic wavenumber k = ω/a (1/m)."""
        return self.omega / self.a

    # =========================================================================
    # Penetration Depths
    # =========================================================================

    @property
    def delta_nu(self) -> float:
        """Viscous penetration depth δ_ν (m)."""
        return penetration_depth_viscous(self.omega, self.rho, self.mu)

    @property
    def delta_kappa(self) -> float:
        """Thermal penetration depth δ_κ (m)."""
        return penetration_depth_thermal(self.omega, self.rho, self.k, self.cp)

    @property
    def dn(self) -> float:
        """Alias for delta_nu (reference baseline convention)."""
        return self.delta_nu

    @property
    def dk(self) -> float:
        """Alias for delta_kappa (reference baseline convention)."""
        return self.delta_kappa

    # =========================================================================
    # Pressure Amplitude
    # =========================================================================

    @property
    def p1_mag(self) -> float:
        """Pressure amplitude |p1| (Pa)."""
        return np.abs(self.p1)

    @property
    def p1_phase(self) -> float:
        """Pressure phase angle (degrees)."""
        return np.angle(self.p1, deg=True)

    @property
    def p1_real(self) -> float:
        """Real part of p1 (Pa)."""
        return self.p1.real

    @property
    def p1_imag(self) -> float:
        """Imaginary part of p1 (Pa)."""
        return self.p1.imag

    @property
    def drive_ratio(self) -> float:
        """Drive ratio |p1|/p_m (dimensionless)."""
        return self.p1_mag / self.p_m

    # =========================================================================
    # Volume Velocity
    # =========================================================================

    @property
    def U1_mag(self) -> float:
        """Volume velocity amplitude |U1| (m³/s)."""
        return np.abs(self.U1)

    @property
    def U1_phase(self) -> float:
        """Volume velocity phase angle (degrees)."""
        return np.angle(self.U1, deg=True)

    @property
    def U1_real(self) -> float:
        """Real part of U1 (m³/s)."""
        return self.U1.real

    @property
    def U1_imag(self) -> float:
        """Imaginary part of U1 (m³/s)."""
        return self.U1.imag

    # =========================================================================
    # Velocity (requires area)
    # =========================================================================

    @property
    def u1(self) -> complex:
        """Complex velocity amplitude u1 = U1/A (m/s). Requires area."""
        if self.area is None:
            raise ValueError("Area not specified. Set area to calculate velocity.")
        return self.U1 / self.area

    @property
    def u1_mag(self) -> float:
        """Velocity amplitude |u1| (m/s). Requires area."""
        return np.abs(self.u1)

    @property
    def u1_phase(self) -> float:
        """Velocity phase angle (degrees). Requires area."""
        return np.angle(self.u1, deg=True)

    # =========================================================================
    # Displacement
    # =========================================================================

    @property
    def xi1(self) -> complex:
        """Complex displacement amplitude ξ1 = U1/(jωA) (m). Requires area."""
        if self.area is None:
            raise ValueError("Area not specified. Set area to calculate displacement.")
        return self.U1 / (1j * self.omega * self.area)

    @property
    def xi1_mag(self) -> float:
        """Displacement amplitude |ξ1| = |U1|/(ωA) (m). Requires area."""
        if self.area is None:
            raise ValueError("Area not specified. Set area to calculate displacement.")
        return self.U1_mag / (self.omega * self.area)

    # =========================================================================
    # Impedance
    # =========================================================================

    @property
    def Z(self) -> complex:
        """Specific acoustic impedance Z = p1/U1 (Pa·s/m³)."""
        if np.abs(self.U1) < 1e-30:
            return complex(np.inf, 0)
        return self.p1 / self.U1

    @property
    def Z_mag(self) -> float:
        """Impedance magnitude |Z| (Pa·s/m³)."""
        return np.abs(self.Z)

    @property
    def Z_phase(self) -> float:
        """Impedance phase angle (degrees)."""
        return np.angle(self.Z, deg=True)

    @property
    def z(self) -> complex:
        """Normalized specific acoustic impedance z = Z / (ρa/A). Requires area."""
        if self.area is None:
            raise ValueError("Area not specified for normalized impedance.")
        Z_char = self.rho * self.a / self.area
        return self.Z / Z_char

    @property
    def z_mag(self) -> float:
        """Normalized impedance magnitude |z|. Requires area."""
        return np.abs(self.z)

    # =========================================================================
    # Power and Energy
    # =========================================================================

    @property
    def acoustic_power(self) -> float:
        """
        Time-averaged acoustic power (W).

        Edot = (1/2) * Re[p1 * conj(U1)]

        This is the rate of energy flow in the positive x direction.
        """
        return 0.5 * np.real(self.p1 * np.conj(self.U1))

    @property
    def Edot(self) -> float:
        """Alias for acoustic_power (reference baseline convention)."""
        return self.acoustic_power

    @property
    def reactive_power(self) -> float:
        """
        Reactive (oscillating) power (W).

        Q = (1/2) * Im[p1 * conj(U1)]
        """
        return 0.5 * np.imag(self.p1 * np.conj(self.U1))

    @property
    def apparent_power(self) -> float:
        """
        Apparent power (W).

        S = (1/2) * |p1| * |U1|
        """
        return 0.5 * self.p1_mag * self.U1_mag

    @property
    def power_factor(self) -> float:
        """
        Power factor = Edot / S = cos(phase difference).

        Ranges from -1 to 1. Magnitude of 1 indicates p1 and U1 are in phase.
        """
        S = self.apparent_power
        if S < 1e-30:
            return 0.0
        return self.acoustic_power / S

    @property
    def phase_difference(self) -> float:
        """Phase difference between p1 and U1 (degrees)."""
        return self.p1_phase - self.U1_phase

    # =========================================================================
    # Acoustic Intensity (requires area)
    # =========================================================================

    @property
    def intensity(self) -> float:
        """
        Time-averaged acoustic intensity (W/m²). Requires area.

        I = Edot / A
        """
        if self.area is None:
            raise ValueError("Area not specified. Set area to calculate intensity.")
        return self.acoustic_power / self.area

    # =========================================================================
    # Thermoviscous Functions
    # =========================================================================

    def f_nu(self, geometry: Geometry, hydraulic_radius: float) -> complex:
        """
        Compute viscous thermoviscous function f_ν for given geometry.

        Parameters
        ----------
        geometry : Geometry
            Pore geometry object (e.g., ParallelPlate, CircularPore).
        hydraulic_radius : float
            Hydraulic radius of the pore (m).

        Returns
        -------
        complex
            Thermoviscous function f_ν.
        """
        return geometry.f_nu(self.omega, self.delta_nu, hydraulic_radius)

    def f_kappa(self, geometry: Geometry, hydraulic_radius: float) -> complex:
        """
        Compute thermal thermoviscous function f_κ for given geometry.

        Parameters
        ----------
        geometry : Geometry
            Pore geometry object (e.g., ParallelPlate, CircularPore).
        hydraulic_radius : float
            Hydraulic radius of the pore (m).

        Returns
        -------
        complex
            Thermoviscous function f_κ.
        """
        return geometry.f_kappa(self.omega, self.delta_kappa, hydraulic_radius)

    # =========================================================================
    # Boundary Layer Approximations
    # =========================================================================

    def f_nu_boundary_layer(self, r_h: float) -> complex:
        """
        Boundary-layer approximation for f_ν.

        Valid when r_h >> δ_ν.

        Parameters
        ----------
        r_h : float
            Hydraulic radius (m).

        Returns
        -------
        complex
            Approximate f_ν = (1-j) * δ_ν / r_h
        """
        return (1 - 1j) * self.delta_nu / r_h

    def f_kappa_boundary_layer(self, r_h: float) -> complex:
        """
        Boundary-layer approximation for f_κ.

        Valid when r_h >> δ_κ.

        Parameters
        ----------
        r_h : float
            Hydraulic radius (m).

        Returns
        -------
        complex
            Approximate f_κ = (1-j) * δ_κ / r_h
        """
        return (1 - 1j) * self.delta_kappa / r_h

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"AcousticState(|p1|={self.p1_mag:.1f} Pa, |U1|={self.U1_mag:.2e} m³/s, "
            f"T_m={self.T_m:.1f} K, f={self.f:.1f} Hz)"
        )

    def summary(self) -> str:
        """Return a formatted summary of the acoustic state."""
        lines = [
            "=" * 60,
            "ACOUSTIC STATE SUMMARY",
            "=" * 60,
            "",
            "Conditions:",
            f"  Temperature: {self.T_m:.2f} K",
            f"  Mean pressure: {self.p_m/1e6:.3f} MPa",
            f"  Frequency: {self.f:.2f} Hz (ω = {self.omega:.2f} rad/s)",
            "",
            "Gas Properties:",
            f"  Density: {self.rho:.4f} kg/m³",
            f"  Sound speed: {self.a:.2f} m/s",
            f"  Specific heat cp: {self.cp:.2f} J/(kg·K)",
            f"  Viscosity: {self.mu:.3e} Pa·s",
            f"  Thermal conductivity: {self.k:.4f} W/(m·K)",
            f"  Prandtl number: {self.Pr:.4f}",
            f"  Gamma (cp/cv): {self.gamma:.4f}",
            "",
            "Penetration Depths:",
            f"  Viscous δ_ν: {self.delta_nu*1e6:.2f} μm",
            f"  Thermal δ_κ: {self.delta_kappa*1e6:.2f} μm",
            "",
            "Pressure:",
            f"  |p1|: {self.p1_mag:.2f} Pa ({self.p1_mag/1000:.4f} kPa)",
            f"  Phase: {self.p1_phase:.2f}°",
            f"  Drive ratio: {self.drive_ratio*100:.3f}%",
            "",
            "Volume Velocity:",
            f"  |U1|: {self.U1_mag:.4e} m³/s ({self.U1_mag*1e6:.4f} cm³/s)",
            f"  Phase: {self.U1_phase:.2f}°",
            "",
            "Impedance:",
            f"  |Z|: {self.Z_mag:.4e} Pa·s/m³",
            f"  Phase: {self.Z_phase:.2f}°",
            "",
            "Power:",
            f"  Acoustic power (Edot): {self.acoustic_power:.4f} W",
            f"  Phase difference (p1-U1): {self.phase_difference:.2f}°",
            f"  Power factor: {self.power_factor:.4f}",
        ]

        if self.area is not None:
            lines.extend([
                "",
                f"With Area = {self.area*1e4:.4f} cm²:",
                f"  Velocity |u1|: {self.u1_mag:.4f} m/s",
                f"  Displacement |ξ1|: {self.xi1_mag*1000:.4f} mm",
                f"  Intensity: {self.intensity:.4f} W/m²",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
