"""Impedance-based side branch (reference baseline BRANCH equivalent).

This module provides the ImpedanceBranch segment which models a side-attached
acoustic impedance that drains flow from the main duct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class ImpedanceBranch(Segment):
    """
    Side-branch with specified acoustic impedance (reference baseline BRANCH equivalent).

    Models a side branch attached to the main duct with a specified acoustic
    impedance Z_b. The side branch drains flow from the main duct according to:

        p1_out = p1_in  (pressure is continuous)
        U1_out = U1_in - p1_in / Z_b  (flow diverted to side branch)

    The flow into the side branch is U1_branch = p1 / Z_b.

    Parameters
    ----------
    impedance : complex, optional
        Fixed acoustic impedance of the side branch (Pa*s/m^3).
        Use Re(Z_b) for resistance, Im(Z_b) for reactance.
        Must provide either this or impedance_func.
    impedance_func : callable, optional
        Function that takes omega (float) and returns complex impedance.
        Signature: impedance_func(omega: float) -> complex
    re_zb : float, optional
        Real part of impedance (Pa*s/m^3). Can be used instead of
        complex impedance parameter.
    im_zb : float, optional
        Imaginary part of impedance (Pa*s/m^3). Used with re_zb.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    impedance_value : complex or None
        Fixed impedance value, or None if using a function.
    impedance_func : callable or None
        Impedance function, or None if using a fixed value.

    Notes
    -----
    reference baseline BRANCH parameters:
    - Re(Zb): Real part of impedance (resistance)
    - Im(Zb): Imaginary part of impedance (reactance)

    Common use cases:
    - RC dissipator: Re(Zb) > 0, Im(Zb) = 0 (pure resistance)
    - Helmholtz resonator: Im(Zb) < 0 at resonance (stiffness-controlled)
    - Quarter-wave stub: Im(Zb) varies with frequency

    The side branch power dissipation is:
        P_branch = 0.5 * Re(p1 * conj(U1_branch)) = 0.5 * |p1|^2 * Re(1/Z_b)

    Examples
    --------
    >>> from openthermoacoustics.segments import ImpedanceBranch
    >>> from openthermoacoustics.gas import Helium
    >>> # Pure resistance dissipator (RC load)
    >>> branch = ImpedanceBranch(re_zb=4e7, im_zb=0.0, name="RC_dissipator")
    >>> gas = Helium(mean_pressure=3e6)
    >>> p1_in = 2.5e5 + 0j
    >>> U1_in = 0.3 * np.exp(-1j * np.pi/2)  # 90° lagging
    >>> p1_out, U1_out, T_out = branch.propagate(p1_in, U1_in, 300.0, 500.0, gas)
    >>> # Flow is reduced by p1/Z_b
    """

    def __init__(
        self,
        impedance: complex | None = None,
        impedance_func: Callable[[float], complex] | None = None,
        re_zb: float | None = None,
        im_zb: float | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize an impedance branch segment.

        Parameters
        ----------
        impedance : complex, optional
            Fixed acoustic impedance (Pa*s/m^3).
        impedance_func : callable, optional
            Function that takes omega and returns complex impedance.
        re_zb : float, optional
            Real part of impedance (Pa*s/m^3).
        im_zb : float, optional
            Imaginary part of impedance (Pa*s/m^3).
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If neither impedance/impedance_func nor re_zb/im_zb is provided,
            or if conflicting parameters are provided.
        """
        # Handle re_zb/im_zb shorthand
        if re_zb is not None or im_zb is not None:
            if impedance is not None or impedance_func is not None:
                raise ValueError(
                    "Cannot provide both re_zb/im_zb and impedance/impedance_func"
                )
            real = re_zb if re_zb is not None else 0.0
            imag = im_zb if im_zb is not None else 0.0
            impedance = complex(real, imag)

        if impedance is None and impedance_func is None:
            raise ValueError(
                "Must provide either impedance, impedance_func, or re_zb/im_zb"
            )
        if impedance is not None and impedance_func is not None:
            raise ValueError("Cannot provide both impedance and impedance_func")

        self._impedance_value: complex | None = impedance
        self._impedance_func: Callable[[float], complex] | None = impedance_func

        # Lumped element: length = 0, area = 0 (not applicable)
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    @property
    def impedance_value(self) -> complex | None:
        """
        Fixed impedance value, if specified.

        Returns
        -------
        complex or None
            Fixed impedance in Pa*s/m^3, or None if using a function.
        """
        return self._impedance_value

    @property
    def impedance_function(self) -> Callable[[float], complex] | None:
        """
        Impedance function, if specified.

        Returns
        -------
        callable or None
            Function that takes omega and returns impedance, or None.
        """
        return self._impedance_func

    def get_impedance(self, omega: float) -> complex:
        """
        Get the acoustic impedance at a given angular frequency.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Acoustic impedance in Pa*s/m^3.
        """
        if self._impedance_value is not None:
            return self._impedance_value
        else:
            assert self._impedance_func is not None  # for type checker
            return self._impedance_func(omega)

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
        Propagate acoustic state through the impedance branch.

        At the junction:
        - Pressure is continuous: p1_out = p1_in
        - Flow diverted to side branch: U1_branch = p1_in / Z_b
        - Flow continuity: U1_out = U1_in - U1_branch

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
            - p1_out: Complex pressure at outlet (Pa), equal to input
            - U1_out: Complex velocity after diversion (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
        """
        Z_b = self.get_impedance(omega)

        # Pressure is continuous through the junction
        p1_out = p1_in

        # Flow diverted to side branch
        U1_branch = p1_in / Z_b

        # Flow continuity: what goes in must come out (minus what goes to branch)
        U1_out = U1_in - U1_branch

        # Temperature is unchanged through the junction
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def branch_flow(self, p1: complex, omega: float) -> complex:
        """
        Calculate the volumetric velocity diverted to the side branch.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Complex volumetric velocity into side branch (m^3/s).
        """
        Z_b = self.get_impedance(omega)
        return p1 / Z_b

    def branch_power(self, p1: complex, omega: float) -> float:
        """
        Calculate the time-averaged acoustic power dissipated in the branch.

        P = 0.5 * Re(p1 * conj(U1_branch)) = 0.5 * |p1|^2 * Re(1/Z_b)

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at the junction (Pa).
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        float
            Time-averaged power dissipation (W).
        """
        Z_b = self.get_impedance(omega)
        U1_branch = p1 / Z_b
        return 0.5 * np.real(p1 * np.conj(U1_branch))

    def __repr__(self) -> str:
        """Return string representation of the impedance branch."""
        if self._impedance_value is not None:
            re = self._impedance_value.real
            im = self._impedance_value.imag
            return f"ImpedanceBranch(name='{self._name}', Z_b={re:+.4e}{im:+.4e}j)"
        else:
            return f"ImpedanceBranch(name='{self._name}', impedance_func=<function>)"


# Alias for reference baseline naming
BRANCH = ImpedanceBranch
