"""Radiation impedance side branches (reference baseline OPNBRANCH and PISTBRANCH equivalents).

This module provides segments for modeling radiation impedance at openings:
- OPNBRANCH: Radiation into 4π solid angle (unflanged opening)
- PISTBRANCH: Radiation into 2π solid angle (flanged piston)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1  # Bessel function of the first kind, order 1

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class OpenBranch(Segment):
    """
    Side-branch with radiation impedance into 4π solid angle (reference baseline OPNBRANCH).

    Models radiation from an unflanged opening radiating into free space (4π solid angle).
    The impedance is specified as Re(Z)/k² and Im(Z)/k, where k = ω/a is the wave number.

    In the low-frequency limit (kr << 1), the radiation impedance of an unflanged
    circular opening of radius r is approximately:
        Z_rad = (ρm*a)/(πr²) * [(ka)²/2 + j*0.6*ka]

    So for the default circular opening:
        Re(Z)/k² ≈ ρm*a/(2πr²)
        Im(Z)/k ≈ 0.6*ρm*a/(πr²)

    Parameters
    ----------
    re_z_over_k2 : float
        Real part of impedance divided by k² (Pa*s/m).
        For circular opening: ≈ ρm*a / (2*π*r²)
    im_z_over_k : float
        Imaginary part of impedance divided by k (Pa*s/m²).
        For circular opening: ≈ 0.6*ρm*a / (π*r²)
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The actual impedance at angular frequency ω is computed as:
        Re(Z_b) = re_z_over_k2 * k² = re_z_over_k2 * (ω/a)²
        Im(Z_b) = im_z_over_k * k = im_z_over_k * (ω/a)

    This frequency dependence is appropriate for radiation impedance in the
    low-frequency limit where kr << 1.

    References
    ----------
    .. [1] Kinsler et al., "Fundamentals of Acoustics", Chapter 7
    .. [2] published literature, relevant reference

    Examples
    --------
    >>> from openthermoacoustics.segments import OpenBranch
    >>> from openthermoacoustics.gas import Helium
    >>> # For a 1 cm radius opening in 1 atm helium at 300 K
    >>> # ρm ≈ 0.16 kg/m³, a ≈ 1000 m/s, r = 0.01 m
    >>> # Re(Z)/k² ≈ 0.16*1000/(2*pi*0.01²) ≈ 2.5e5 Pa*s/m
    >>> # Im(Z)/k ≈ 0.6*0.16*1000/(pi*0.01²) ≈ 3.1e5 Pa*s/m²
    >>> opn = OpenBranch(re_z_over_k2=2.5e5, im_z_over_k=3.1e5)
    >>> gas = Helium(mean_pressure=1e5)
    """

    def __init__(
        self,
        re_z_over_k2: float,
        im_z_over_k: float,
        name: str = "",
    ) -> None:
        """
        Initialize an OPNBRANCH segment.

        Parameters
        ----------
        re_z_over_k2 : float
            Real part of impedance divided by k² (Pa*s/m).
        im_z_over_k : float
            Imaginary part of impedance divided by k (Pa*s/m²).
        name : str, optional
            Name identifier for the segment.
        """
        self._re_z_over_k2 = re_z_over_k2
        self._im_z_over_k = im_z_over_k

        # Lumped element: length = 0, area = 0 (not applicable)
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    @property
    def re_z_over_k2(self) -> float:
        """Real part of impedance divided by k² (Pa*s/m)."""
        return self._re_z_over_k2

    @property
    def im_z_over_k(self) -> float:
        """Imaginary part of impedance divided by k (Pa*s/m²)."""
        return self._im_z_over_k

    def get_impedance(self, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Get the acoustic impedance at a given angular frequency.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing sound speed.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Acoustic impedance in Pa*s/m³.
        """
        a = gas.sound_speed(T_m)
        k = omega / a  # wave number

        re_z = self._re_z_over_k2 * k**2
        im_z = self._im_z_over_k * k

        return complex(re_z, im_z)

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

        For a lumped element, derivatives are zero (no distributed propagation).
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
        Propagate acoustic state through the radiation branch.

        At the junction:
        - Pressure is continuous: p1_out = p1_in
        - Flow diverted to radiation: U1_rad = p1_in / Z_b
        - Flow continuity: U1_out = U1_in - U1_rad

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
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        Z_b = self.get_impedance(omega, gas, T_m)

        # Pressure is continuous through the junction
        p1_out = p1_in

        # Flow diverted to radiation
        U1_rad = p1_in / Z_b

        # Flow continuity
        U1_out = U1_in - U1_rad

        # Temperature unchanged
        return p1_out, U1_out, T_m

    def branch_flow(self, p1: complex, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Calculate the volumetric velocity radiated through the opening.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Complex volumetric velocity into radiation branch (m³/s).
        """
        Z_b = self.get_impedance(omega, gas, T_m)
        return p1 / Z_b

    def branch_power(self, p1: complex, omega: float, gas: Gas, T_m: float) -> float:
        """
        Calculate the time-averaged acoustic power radiated through the opening.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            Time-averaged radiated power (W).
        """
        Z_b = self.get_impedance(omega, gas, T_m)
        U1_rad = p1 / Z_b
        return 0.5 * np.real(p1 * np.conj(U1_rad))

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OpenBranch(name='{self._name}', "
            f"re_z_over_k2={self._re_z_over_k2:.4e}, "
            f"im_z_over_k={self._im_z_over_k:.4e})"
        )


class PistonBranch(Segment):
    """
    Side-branch with flanged piston radiation impedance (reference baseline PISTBRANCH).

    Models radiation from a circular, flanged piston radiating into half-space
    (2π solid angle). The radiation impedance is calculated using the exact
    formula involving Bessel functions.

    Parameters
    ----------
    radius : float
        Radius of the circular piston (m).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The radiation impedance of a flanged circular piston is given by [1]:

        Z_b = (ρm * a / A) * [R1(2kr) + j * X1(2kr)]

    where:
        A = π * r² (piston area)
        k = ω / a (wave number)
        R1(x) = 1 - 2*J1(x)/x  (resistance function)
        X1(x) = reactive term (see implementation)

    The reactive term X1 uses a polynomial approximation for small arguments
    and an asymptotic form for large arguments:
        - For 2kr > 2.68: X1 ≈ (4/π)/(2kr) + √(8/π)*sin(2kr - 3π/4)/(2kr)^(3/2)
        - For 2kr ≤ 2.68: X1 ≈ (4/π)*(2kr)/3 * (1 - (2kr)²/15)

    References
    ----------
    .. [1] Kinsler et al., "Fundamentals of Acoustics", Chapter 7
    .. [2] published literature, relevant reference, governing relations

    Examples
    --------
    >>> from openthermoacoustics.segments import PistonBranch
    >>> from openthermoacoustics.gas import Helium
    >>> # Flanged piston with 2 cm radius
    >>> piston = PistonBranch(radius=0.02, name="radiator")
    >>> gas = Helium(mean_pressure=1e5)
    >>> p1_in = 1000 + 0j  # Pa
    >>> U1_in = 1e-4 + 0j  # m³/s
    >>> p1_out, U1_out, T_out = piston.propagate(p1_in, U1_in, 300.0, 1000.0, gas)
    """

    # Threshold for switching between polynomial and asymptotic reactive formula
    _THRESHOLD_2KR = 2.68

    def __init__(
        self,
        radius: float,
        name: str = "",
    ) -> None:
        """
        Initialize a PISTBRANCH segment.

        Parameters
        ----------
        radius : float
            Radius of the circular piston (m). Must be positive.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If radius is not positive.
        """
        if radius <= 0:
            raise ValueError(f"Piston radius must be positive, got {radius}")

        self._radius = radius
        self._area = np.pi * radius**2

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=self._area, geometry=None)

    @property
    def radius(self) -> float:
        """Radius of the piston (m)."""
        return self._radius

    @property
    def piston_area(self) -> float:
        """Area of the piston (m²)."""
        return self._area

    def get_impedance(self, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Get the radiation impedance of the flanged piston.

        Uses reference baseline governing relations for the radiation impedance.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing sound speed and density.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Acoustic radiation impedance in Pa*s/m³.
        """
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        k = omega / a  # wave number
        r = self._radius

        # Characteristic impedance times 1/A
        Z_char = rho_m * a / self._area

        # Argument for Bessel function
        x = 2 * k * r  # = 2kr

        # Resistance: R1 = 1 - 2*J1(x)/x
        if x < 1e-10:
            # Small argument limit: J1(x) ≈ x/2, so 2*J1(x)/x ≈ 1
            # R1 ≈ (x²/8) for small x
            R1 = x**2 / 8.0
        else:
            R1 = 1.0 - 2.0 * j1(x) / x

        # Reactance: X1 depends on whether x > 2.68 or not
        # This follows reference baseline governing relations
        if x > self._THRESHOLD_2KR:
            # Asymptotic form for large argument
            # X1 = (4/π)/(2kr) + √(8/π)*sin(2kr - 3π/4)/(2kr)^(3/2)
            X1 = (4.0 / np.pi) / x + np.sqrt(8.0 / np.pi) * np.sin(
                x - 3.0 * np.pi / 4.0
            ) / x**1.5
        else:
            # Polynomial approximation for small argument
            # X1 = (4/π)*(2kr)/3 * (1 - (2kr)²/15)
            X1 = (4.0 / np.pi) * x / 3.0 * (1.0 - x**2 / 15.0)

        return Z_char * complex(R1, X1)

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

        For a lumped element, derivatives are zero (no distributed propagation).
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
        Propagate acoustic state through the piston radiation branch.

        At the junction:
        - Pressure is continuous: p1_out = p1_in
        - Flow diverted to radiation: U1_rad = p1_in / Z_b
        - Flow continuity: U1_out = U1_in - U1_rad

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
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        Z_b = self.get_impedance(omega, gas, T_m)

        # Pressure is continuous through the junction
        p1_out = p1_in

        # Flow diverted to radiation
        U1_rad = p1_in / Z_b

        # Flow continuity
        U1_out = U1_in - U1_rad

        # Temperature unchanged
        return p1_out, U1_out, T_m

    def branch_flow(self, p1: complex, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Calculate the volumetric velocity radiated through the piston.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Complex volumetric velocity into radiation branch (m³/s).
        """
        Z_b = self.get_impedance(omega, gas, T_m)
        return p1 / Z_b

    def branch_power(self, p1: complex, omega: float, gas: Gas, T_m: float) -> float:
        """
        Calculate the time-averaged acoustic power radiated by the piston.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            Time-averaged radiated power (W).
        """
        Z_b = self.get_impedance(omega, gas, T_m)
        U1_rad = p1 / Z_b
        return 0.5 * np.real(p1 * np.conj(U1_rad))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PistonBranch(name='{self._name}', radius={self._radius:.6e})"


# reference baseline aliases
OPNBRANCH = OpenBranch
PISTBRANCH = PistonBranch
