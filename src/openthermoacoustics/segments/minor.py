"""Minor loss segment (MINOR).

This module implements the MINOR segment from reference baseline, which models
minor losses (K-factor losses) at abrupt area changes, fittings,
valves, etc. Common applications include:
- Orifice plates
- Pipe fittings (elbows, tees)
- Sudden expansions/contractions
- Gas diodes

The implementation follows reference baseline's equations  for
the case with no steady flow (N_dot = 0).

References
----------
[53] Idelchik, I. E. (1996). Handbook of Hydraulic Resistance.
[59] White, F. M. (2003). Fluid Mechanics.
[42] Backhaus, S., et al. (2004). JASA 116, 2806. (Gas diodes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Minor(Segment):
    """
    Minor loss segment for K-factor losses.

    Models minor losses (form losses) at fittings, area changes, valves,
    and other flow obstructions. Based on the steady-flow relationship:

        Δp = -½ K ρ u²

    Extended to oscillating flow using the quasi-steady approximation.
    This approximation is valid when |ξ₁| >> rh (displacement amplitude
    much larger than hydraulic radius).

    Parameters
    ----------
    area : float
        Reference area for velocity calculation (m²). This is typically
        the smallest area in the geometry (e.g., orifice area), but may
        be the pipe area for valves and fittings.
    K_plus : float
        Minor loss coefficient for flow in +x direction.
    K_minus : float
        Minor loss coefficient for flow in -x direction.
        Set equal to K_plus for symmetric losses.
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    For oscillating flow with no steady component (N_dot = 0), the
    pressure change is given by reference baseline governing relations:

        p1_out - p1_in = -(K- + K+) * (2 ρm |U1|) / (3π A²) * U1

    This reduces acoustic power (dissipates energy). The volume flow
    rate U1 and temperature Tm are unchanged across the segment.

    Common K values (from Idelchik):
    - Sharp-edged orifice: K ≈ (1 - σ)² / σ² (σ = area ratio)
    - Sudden expansion (Borda-Carnot): K = (1 - σ)²
    - Sudden contraction: K ≈ 0.5(1 - σ)
    - 90° elbow: K ≈ 0.3-1.0 depending on radius
    - Tee junction: K ≈ 1.0-2.0

    Examples
    --------
    >>> from openthermoacoustics.segments import Minor
    >>> from openthermoacoustics.gas import Air
    >>> # Orifice plate with K=1.0
    >>> orifice = Minor(area=0.002, K_plus=1.0, K_minus=1.0)
    >>> gas = Air(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = orifice.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        area: float,
        K_plus: float,
        K_minus: float | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a minor loss segment.

        Parameters
        ----------
        area : float
            Reference area for velocity calculation (m²).
        K_plus : float
            Minor loss coefficient for flow in +x direction.
        K_minus : float, optional
            Minor loss coefficient for flow in -x direction.
            Defaults to K_plus (symmetric loss).
        name : str, optional
            Name identifier for the segment.
        """
        if area <= 0:
            raise ValueError(f"Area must be positive, got {area}")
        if K_plus < 0:
            raise ValueError(f"K_plus must be non-negative, got {K_plus}")

        if K_minus is None:
            K_minus = K_plus

        if K_minus < 0:
            raise ValueError(f"K_minus must be non-negative, got {K_minus}")

        self._K_plus = K_plus
        self._K_minus = K_minus

        # MINOR has zero length (lumped element)
        super().__init__(name=name, length=0.0, area=area, geometry=None)

    @property
    def K_plus(self) -> float:
        """Minor loss coefficient for +x direction flow."""
        return self._K_plus

    @property
    def K_minus(self) -> float:
        """Minor loss coefficient for -x direction flow."""
        return self._K_minus

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Return zero derivatives (MINOR is a lumped element).

        MINOR has zero length and applies an instantaneous pressure
        change, so it doesn't use ODE integration. This method returns
        zeros to satisfy the abstract base class requirement.
        """
        return np.zeros(4)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the minor loss.

        Applies the pressure drop from governing relations while leaving
        U1 and Tm unchanged.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s). Not used for MINOR.
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
            U1_out equals U1_in (unchanged).
            T_m_out equals T_m (unchanged).
        """
        # Gas density at local temperature
        rho_m = gas.density(T_m)

        # Velocity amplitude
        U1_mag = np.abs(U1_in)
        A = self._area

        # Pressure drop coefficient from governing relations
        # Δp1 = -(K- + K+) * (2 ρm |U1|) / (3π A²) * U1
        coeff = (self._K_minus + self._K_plus) * 2 * rho_m * U1_mag / (3 * np.pi * A**2)
        delta_p1 = -coeff * U1_in

        p1_out = p1_in + delta_p1

        # U1 and T_m are unchanged
        return p1_out, U1_in, T_m

    def acoustic_power_dissipation(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> float:
        """
        Calculate the time-averaged acoustic power dissipated.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        float
            Time-averaged acoustic power dissipated (W).
            Always positive (energy is lost).
        """
        p1_out, _, _ = self.propagate(p1_in, U1_in, T_m, omega, gas)

        # Acoustic power: E_dot = Re[p1 * U1_conj] / 2
        E_dot_in = 0.5 * np.real(p1_in * np.conj(U1_in))
        E_dot_out = 0.5 * np.real(p1_out * np.conj(U1_in))

        # Power dissipated is the difference
        return E_dot_in - E_dot_out

    def __repr__(self) -> str:
        return (
            f"Minor(name='{self._name}', area={self._area}, "
            f"K+={self._K_plus}, K-={self._K_minus})"
        )


# Alias for reference baseline naming convention
MINOR = Minor
