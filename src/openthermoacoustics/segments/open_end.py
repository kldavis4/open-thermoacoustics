"""Open end with radiation impedance segment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class OpenEnd(Segment):
    """
    Open end segment with radiation impedance.

    Models the acoustic radiation impedance at an open pipe termination.
    Unlike the ideal SoftEnd (p1=0), a real open end has finite radiation
    impedance that depends on the pipe radius, frequency, and flange type.

    The radiation impedance for an unflanged pipe (ka << 1):
        Z_rad = (rho*c/A) * [(ka)^2/4 + j*0.6133*ka]

    For a flanged pipe or infinite baffle (ka << 1):
        Z_rad = (rho*c/A) * [(ka)^2/2 + j*0.8216*ka]

    where k = omega/c is the wavenumber and a is the pipe radius.

    The end correction (effective length extension) is:
        delta = 0.6133*a (unflanged)
        delta = 0.8216*a (flanged/infinite baffle)

    Parameters
    ----------
    radius : float
        Pipe radius at the open end (m).
    flange_type : {'unflanged', 'flanged', 'infinite_baffle'}, optional
        Type of pipe termination. Default is 'unflanged'.
        - 'unflanged': Pipe ends in free space (most common)
        - 'flanged': Pipe terminates in a finite flange
        - 'infinite_baffle': Pipe terminates in an infinite baffle
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    radius : float
        Pipe radius (m).
    flange_type : str
        Type of termination.

    Notes
    -----
    The open end is a lumped element with zero length. The radiation
    impedance acts as a termination that reflects acoustic waves.

    At low frequencies (ka << 1), the real part of Z_rad is small,
    meaning most energy is reflected and little is radiated.

    At high frequencies, more accurate Levine-Schwinger formulas
    are used that account for the frequency-dependent behavior.

    The end correction represents the additional acoustic length
    due to the inertia of air oscillating outside the pipe.

    Examples
    --------
    >>> from openthermoacoustics.segments import OpenEnd
    >>> from openthermoacoustics.gas import Helium
    >>> open_end = OpenEnd(radius=0.025, flange_type='unflanged')
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = open_end.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        radius: float,
        flange_type: Literal["unflanged", "flanged", "infinite_baffle"] = "unflanged",
        name: str = "",
    ) -> None:
        """
        Initialize an open end segment.

        Parameters
        ----------
        radius : float
            Pipe radius at the open end (m).
        flange_type : {'unflanged', 'flanged', 'infinite_baffle'}, optional
            Type of pipe termination. Default is 'unflanged'.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If radius is not positive or flange_type is invalid.
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        valid_flange_types = ("unflanged", "flanged", "infinite_baffle")
        if flange_type not in valid_flange_types:
            raise ValueError(
                f"flange_type must be one of {valid_flange_types}, got '{flange_type}'"
            )

        self._radius = radius
        self._flange_type = flange_type
        area = np.pi * radius**2

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=area, geometry=None)

    @property
    def radius(self) -> float:
        """
        Pipe radius at the open end.

        Returns
        -------
        float
            Radius in meters.
        """
        return self._radius

    @property
    def flange_type(self) -> str:
        """
        Type of pipe termination.

        Returns
        -------
        str
            One of 'unflanged', 'flanged', or 'infinite_baffle'.
        """
        return self._flange_type

    def end_correction(self) -> float:
        """
        Calculate the end correction (effective length extension).

        The end correction represents the additional acoustic length
        due to the inertia of air oscillating outside the pipe.

        Returns
        -------
        float
            End correction delta in meters.
        """
        if self._flange_type == "unflanged":
            # Levine-Schwinger result for unflanged pipe
            return 0.6133 * self._radius
        else:
            # Flanged or infinite baffle
            return 0.8216 * self._radius

    def radiation_impedance(self, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Calculate the radiation impedance at the open end.

        For low frequencies (ka << 1), uses the classic approximations.
        For higher frequencies, uses improved Levine-Schwinger formulas.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Radiation impedance Z_rad in Pa*s/m^3.
        """
        rho = gas.density(T_m)
        c = gas.sound_speed(T_m)
        A = self._area

        k = omega / c  # wavenumber
        ka = k * self._radius  # Helmholtz number

        # Characteristic impedance
        Z_0 = rho * c / A

        if self._flange_type == "unflanged":
            # Levine-Schwinger formulas for unflanged pipe
            # Valid for all ka, but simplified for small ka
            if ka < 0.5:
                # Low frequency approximation
                # Real part: radiation resistance
                R_rad = Z_0 * (ka**2) / 4
                # Imaginary part: radiation reactance (mass loading)
                X_rad = Z_0 * 0.6133 * ka
            else:
                # Higher frequency: use improved formulas
                # Resistance approaches rho*c/A as ka -> infinity
                R_rad = Z_0 * (ka**2) / (4 + ka**2)
                # Reactance (inertia) with frequency-dependent end correction
                # End correction decreases at higher frequencies
                delta_factor = 0.6133 * (1 - 0.1 * ka**2) if ka < 1.5 else 0.6133 / (1 + 0.2 * ka)
                X_rad = Z_0 * delta_factor * ka
        else:
            # Flanged or infinite baffle
            if ka < 0.5:
                # Low frequency approximation
                R_rad = Z_0 * (ka**2) / 2
                X_rad = Z_0 * 0.8216 * ka
            else:
                # Higher frequency formulas
                R_rad = Z_0 * (ka**2) / (2 + ka**2)
                delta_factor = 0.8216 * (1 - 0.1 * ka**2) if ka < 1.5 else 0.8216 / (1 + 0.15 * ka)
                X_rad = Z_0 * delta_factor * ka

        return R_rad + 1j * X_rad

    def reflection_coefficient(self, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Calculate the pressure reflection coefficient at the open end.

        The reflection coefficient R relates the reflected wave to the
        incident wave: p_reflected = R * p_incident

        R = (Z_rad - Z_0) / (Z_rad + Z_0)

        where Z_0 = rho*c/A is the characteristic impedance.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            Pressure reflection coefficient (dimensionless).
        """
        rho = gas.density(T_m)
        c = gas.sound_speed(T_m)
        A = self._area

        Z_0 = rho * c / A
        Z_rad = self.radiation_impedance(omega, gas, T_m)

        return (Z_rad - Z_0) / (Z_rad + Z_0)

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
        Propagate acoustic state through the open end.

        The open end applies the radiation impedance as a boundary condition.
        The relationship p1 = Z_rad * U1 must be satisfied at the termination.

        For use in propagation, this method returns the state unchanged,
        similar to other boundary conditions. Use the radiation_impedance()
        and residual() methods to check or enforce the boundary condition.

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
            Tuple of (p1_out, U1_out, T_m_out), equal to the inputs.
        """
        return p1_in, U1_in, T_m

    def residual(self, p1: complex, U1: complex, omega: float, gas: Gas, T_m: float) -> complex:
        """
        Calculate the boundary condition residual.

        For an open end, the residual is p1 - Z_rad * U1, which should
        be zero when the boundary condition is satisfied.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        complex
            The boundary condition residual.
        """
        Z_rad = self.radiation_impedance(omega, gas, T_m)
        return p1 - Z_rad * U1

    def is_satisfied(
        self,
        p1: complex,
        U1: complex,
        omega: float,
        gas: Gas,
        T_m: float,
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Check if the open end boundary condition is satisfied.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).
        tolerance : float, optional
            Relative tolerance for the boundary condition check. Default is 1e-6.

        Returns
        -------
        bool
            True if the boundary condition is satisfied within tolerance.
        """
        res = self.residual(p1, U1, omega, gas, T_m)
        # Normalize by the larger of |p1| or |Z_rad * U1| for relative tolerance
        Z_rad = self.radiation_impedance(omega, gas, T_m)
        norm = max(abs(p1), abs(Z_rad * U1), 1e-20)
        return abs(res) / norm < tolerance

    def __repr__(self) -> str:
        """Return string representation of the open end."""
        return (
            f"OpenEnd(name='{self._name}', radius={self._radius}, "
            f"flange_type='{self._flange_type}')"
        )
