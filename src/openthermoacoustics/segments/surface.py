"""Surface segment with thermal-hysteresis dissipation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import penetration_depth_thermal

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Surface(Segment):
    """
    Surface segment with thermal-hysteresis dissipation.

    A SURFACE is a lumped element representing a surface area exposed to
    oscillating pressure. It models thermal-hysteresis losses in the thermal
    penetration depth at the surface. This segment always absorbs acoustic
    power and is typically used at the ends of ducts before a HARDEND.

    The volume flow changes according to reference baseline governing relations:

        U1_out = U1_in - (1 + j) * ω * p1 / (ρ_m * a²) * (γ - 1) / (1 + ε_s) * S * δ_κ / 2

    where:
    - S is the surface area
    - δ_κ = sqrt(2α/ω) is the thermal penetration depth
    - ε_s is the ratio of thermal effusivities (gas/solid)

    Parameters
    ----------
    area : float
        Surface area exposed to oscillating pressure (m²).
    epsilon_s : float, optional
        Solid thermal effusivity ratio. For an ideal solid with infinite
        thermal conductivity and heat capacity, ε_s = 0. Default is 0.0
        (ideal solid).
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    area : float
        Surface area (m²).
    epsilon_s : float
        Solid thermal effusivity ratio.

    Notes
    -----
    The SURFACE segment:
    - Does not affect mean temperature (T_m_out = T_m_in)
    - Does not affect pressure amplitude (p1_out = p1_in)
    - Changes volume flow due to thermal-hysteresis dissipation

    The solid thermal effusivity ratio ε_s is calculated as:

        ε_s = sqrt((k * ρ_m * c_p) / (k_s * ρ_s * c_s))

    where gas properties (k, ρ_m, c_p) and solid properties (k_s, ρ_s, c_s)
    are thermal conductivity, density, and specific heat respectively.

    For most practical solids with good thermal properties (metals), ε_s << 1,
    so the "ideal" approximation (ε_s = 0) is often sufficient.

    Examples
    --------
    >>> from openthermoacoustics.segments import Surface
    >>> from openthermoacoustics.gas import Helium
    >>> # Create a surface at the end of a duct
    >>> surf = Surface(area=0.05, name="end_cap")
    >>> gas = Helium(mean_pressure=3e6)
    >>> p1_out, U1_out, T_out = surf.propagate(
    ...     p1_in=1e5+0j, U1_in=1e-5+0j, T_m=325.0, omega=539.0, gas=gas
    ... )
    """

    def __init__(
        self,
        area: float,
        epsilon_s: float = 0.0,
        name: str = "",
    ) -> None:
        """
        Initialize a surface segment.

        Parameters
        ----------
        area : float
            Surface area exposed to oscillating pressure (m²).
        epsilon_s : float, optional
            Solid thermal effusivity ratio. Default is 0.0 (ideal solid).
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If area is not positive or epsilon_s is negative.
        """
        if area <= 0:
            raise ValueError(f"Surface area must be positive, got {area}")
        if epsilon_s < 0:
            raise ValueError(f"Epsilon_s must be non-negative, got {epsilon_s}")

        self._area = area
        self._epsilon_s = epsilon_s

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=area, geometry=None)

    @property
    def surface_area(self) -> float:
        """
        Surface area exposed to oscillating pressure.

        Returns
        -------
        float
            Surface area in m².
        """
        return self._area

    @property
    def epsilon_s(self) -> float:
        """
        Solid thermal effusivity ratio.

        Returns
        -------
        float
            Dimensionless ratio ε_s.
        """
        return self._epsilon_s

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        For a lumped element, the derivatives are zero since there is no
        distributed propagation. All physics is captured in the propagate method.

        Parameters
        ----------
        x : float
            Axial position (m). Not used for lumped element.
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        NDArray[np.float64]
            Zero vector [0, 0, 0, 0] since this is a lumped element.
        """
        return np.zeros(4, dtype=np.float64)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the surface.

        Implements reference baseline governing relations:
            U1_out = U1_in - (1+j) * ω * p1 / (ρ_m * a²) * (γ-1)/(1+ε_s) * S * δ_κ/2

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude at output (Pa), equal to input
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
        """
        # Gas properties
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        kappa = gas.thermal_conductivity(T_m)
        cp = gas.specific_heat_cp(T_m)

        # Thermal penetration depth
        delta_kappa = penetration_depth_thermal(omega, rho_m, kappa, cp)

        # Volume flow change due to thermal-hysteresis (reference governing relation)
        # dU = -(1+j) * omega * p1 / (rho_m * a^2) * (gamma-1)/(1+eps_s) * S * delta_kappa/2
        factor = (1 + 1j) * omega / (rho_m * a**2)
        factor *= (gamma - 1) / (1 + self._epsilon_s)
        factor *= self._area * delta_kappa / 2

        # Pressure is unchanged
        p1_out = p1_in

        # Volume flow changes
        U1_out = U1_in - factor * p1_in

        # Temperature is unchanged
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def acoustic_power_dissipated(
        self,
        p1: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> float:
        """
        Calculate the acoustic power dissipated by thermal hysteresis.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        float
            Acoustic power dissipated (W). Always positive.
        """
        # Gas properties
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        kappa = gas.thermal_conductivity(T_m)
        cp = gas.specific_heat_cp(T_m)

        # Thermal penetration depth
        delta_kappa = penetration_depth_thermal(omega, rho_m, kappa, cp)

        # Power dissipation = 0.5 * Re(p1 * conj(dU))
        # where dU = -(1+j) * factor * p1
        # So: p1 * conj(dU) = -p1 * (1-j) * factor * conj(p1) = -(1-j) * factor * |p1|^2
        # Re(...) = -Re((1-j) * factor) * |p1|^2 = -factor * |p1|^2
        # The dissipation is positive (absorbs power), so we take the real part properly

        factor = omega / (rho_m * a**2)
        factor *= (gamma - 1) / (1 + self._epsilon_s)
        factor *= self._area * delta_kappa / 2

        # Power = 0.5 * Re(p1 * conj(dU)) where dU = -(1+j)*factor*p1
        # = 0.5 * Re(-p1 * (1-j) * factor * conj(p1))
        # = 0.5 * Re(-(1-j)) * factor * |p1|^2
        # = 0.5 * factor * |p1|^2  (since Re(-(1-j)) = -1, power is absorbed)
        return 0.5 * factor * abs(p1) ** 2

    def __repr__(self) -> str:
        """Return string representation of the surface."""
        return (
            f"Surface(name='{self._name}', area={self._area}, "
            f"epsilon_s={self._epsilon_s})"
        )


# Alias for reference baseline compatibility
SURFACE = Surface
