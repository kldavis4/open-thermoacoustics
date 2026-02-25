"""Electrodynamic transducer (loudspeaker/linear alternator) segment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Transducer(Segment):
    """
    Electrodynamic transducer segment representing a loudspeaker or linear alternator.

    Models the electromechanical coupling between electrical and acoustic domains.
    The transducer can operate as:
    - A source (loudspeaker): electrical power is converted to acoustic power
    - A load (linear alternator): acoustic power is converted to electrical power

    The transducer is characterized by:
    - Electrical side: resistance R_e, inductance L_e
    - Mechanical side: moving mass m, spring constant k, mechanical resistance R_m
    - Coupling: force factor Bl, diaphragm area A_d

    The acoustic impedance seen at the diaphragm is:

        Z_a = (Bl)^2 / (Z_e * A_d^2) + Z_m / A_d^2

    where:
        Z_e = R_e + j*omega*L_e  (electrical impedance)
        Z_m = R_m + j*omega*m + k/(j*omega)  (mechanical impedance)

    Parameters
    ----------
    Bl : float
        Force factor (T*m). Product of magnetic field strength and voice coil length.
    R_e : float
        Electrical resistance of the voice coil (Ohm).
    L_e : float
        Electrical inductance of the voice coil (H).
    m : float
        Moving mass of the diaphragm assembly (kg).
    k : float
        Spring constant of the suspension (N/m).
    R_m : float
        Mechanical resistance (damping) (N*s/m).
    A_d : float
        Effective diaphragm area (m^2).
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    Bl : float
        Force factor (T*m).
    R_e : float
        Electrical resistance (Ohm).
    L_e : float
        Electrical inductance (H).
    m : float
        Moving mass (kg).
    k : float
        Spring constant (N/m).
    R_m : float
        Mechanical resistance (N*s/m).
    A_d : float
        Diaphragm area (m^2).

    Notes
    -----
    The transducer is modeled as a lumped element with zero length.
    The governing equations couple electrical and mechanical domains:

    Electrical domain:
        V = Z_e * I + Bl * v
    where v is the diaphragm velocity.

    Mechanical domain:
        F = Bl * I = Z_m * v + A_d * p
    where F is the force, p is acoustic pressure.

    The acoustic impedance (looking into the transducer from the acoustic side)
    depends on the electrical load impedance Z_load:
        Z_a = Z_m / A_d^2 + (Bl)^2 / (A_d^2 * (Z_e + Z_load))

    For blocked electrical terminals (Z_load -> infinity):
        Z_a_blocked = Z_m / A_d^2

    For shorted electrical terminals (Z_load = 0):
        Z_a_free = Z_m / A_d^2 + (Bl)^2 / (A_d^2 * Z_e)

    Examples
    --------
    >>> from openthermoacoustics.segments import Transducer
    >>> from openthermoacoustics.gas import Helium
    >>> # Typical small loudspeaker parameters
    >>> trans = Transducer(
    ...     Bl=5.0,      # T*m
    ...     R_e=6.0,     # Ohm
    ...     L_e=0.5e-3,  # H
    ...     m=0.01,      # kg
    ...     k=2000,      # N/m
    ...     R_m=1.0,     # N*s/m
    ...     A_d=0.01,    # m^2
    ... )
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = trans.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        Bl: float,
        R_e: float,
        L_e: float,
        m: float,
        k: float,
        R_m: float,
        A_d: float,
        name: str = "",
    ) -> None:
        """
        Initialize an electrodynamic transducer segment.

        Parameters
        ----------
        Bl : float
            Force factor (T*m).
        R_e : float
            Electrical resistance (Ohm).
        L_e : float
            Electrical inductance (H).
        m : float
            Moving mass (kg).
        k : float
            Spring constant (N/m).
        R_m : float
            Mechanical resistance (N*s/m).
        A_d : float
            Diaphragm area (m^2).
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If any parameter is non-positive (except L_e which can be zero).
        """
        if Bl <= 0:
            raise ValueError(f"Force factor Bl must be positive, got {Bl}")
        if R_e <= 0:
            raise ValueError(f"Electrical resistance R_e must be positive, got {R_e}")
        if L_e < 0:
            raise ValueError(f"Electrical inductance L_e must be non-negative, got {L_e}")
        if m <= 0:
            raise ValueError(f"Moving mass m must be positive, got {m}")
        if k <= 0:
            raise ValueError(f"Spring constant k must be positive, got {k}")
        if R_m < 0:
            raise ValueError(f"Mechanical resistance R_m must be non-negative, got {R_m}")
        if A_d <= 0:
            raise ValueError(f"Diaphragm area A_d must be positive, got {A_d}")

        self._Bl = Bl
        self._R_e = R_e
        self._L_e = L_e
        self._m = m
        self._k = k
        self._R_m = R_m
        self._A_d = A_d

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=A_d, geometry=None)

    @property
    def Bl(self) -> float:
        """
        Force factor (Bl product).

        Returns
        -------
        float
            Force factor in T*m.
        """
        return self._Bl

    @property
    def R_e(self) -> float:
        """
        Electrical resistance of the voice coil.

        Returns
        -------
        float
            Resistance in Ohm.
        """
        return self._R_e

    @property
    def L_e(self) -> float:
        """
        Electrical inductance of the voice coil.

        Returns
        -------
        float
            Inductance in H.
        """
        return self._L_e

    @property
    def m(self) -> float:
        """
        Moving mass of the diaphragm assembly.

        Returns
        -------
        float
            Mass in kg.
        """
        return self._m

    @property
    def k(self) -> float:
        """
        Spring constant of the suspension.

        Returns
        -------
        float
            Spring constant in N/m.
        """
        return self._k

    @property
    def R_m(self) -> float:
        """
        Mechanical resistance (damping).

        Returns
        -------
        float
            Mechanical resistance in N*s/m.
        """
        return self._R_m

    @property
    def A_d(self) -> float:
        """
        Effective diaphragm area.

        Returns
        -------
        float
            Area in m^2.
        """
        return self._A_d

    def resonant_frequency(self) -> float:
        """
        Calculate the mechanical resonant frequency.

        Returns
        -------
        float
            Resonant frequency f_s in Hz.
        """
        return np.sqrt(self._k / self._m) / (2 * np.pi)

    def electrical_impedance(self, omega: float) -> complex:
        """
        Calculate the electrical impedance Z_e = R_e + j*omega*L_e.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Electrical impedance in Ohm.
        """
        return self._R_e + 1j * omega * self._L_e

    def mechanical_impedance(self, omega: float) -> complex:
        """
        Calculate the mechanical impedance Z_m = R_m + j*omega*m + k/(j*omega).

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Mechanical impedance in N*s/m.
        """
        return self._R_m + 1j * omega * self._m + self._k / (1j * omega)

    def blocked_acoustic_impedance(self, omega: float) -> complex:
        """
        Calculate the blocked acoustic impedance (electrical terminals open).

        With the electrical circuit open (infinite load impedance), no current
        flows and the acoustic impedance is purely mechanical:
            Z_a_blocked = Z_m / A_d^2

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Blocked acoustic impedance in Pa*s/m^3.
        """
        Z_m = self.mechanical_impedance(omega)
        return Z_m / (self._A_d**2)

    def free_acoustic_impedance(self, omega: float) -> complex:
        """
        Calculate the free acoustic impedance (electrical terminals shorted).

        With the electrical circuit shorted (zero load impedance), maximum
        current flows and the motional impedance contributes:
            Z_a_free = Z_m / A_d^2 + (Bl)^2 / (Z_e * A_d^2)

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Free acoustic impedance in Pa*s/m^3.
        """
        Z_e = self.electrical_impedance(omega)
        Z_m = self.mechanical_impedance(omega)

        # Motional contribution from electromagnetic coupling
        Z_motional = (self._Bl**2) / (Z_e * self._A_d**2)

        return Z_m / (self._A_d**2) + Z_motional

    def acoustic_impedance(
        self, omega: float, Z_load: complex | None = None
    ) -> complex:
        """
        Calculate the acoustic impedance with a given electrical load.

        The total acoustic impedance looking into the transducer:
            Z_a = Z_m / A_d^2 + (Bl)^2 / (A_d^2 * (Z_e + Z_load))

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        Z_load : complex, optional
            Electrical load impedance (Ohm). If None, assumes open circuit
            (blocked condition).

        Returns
        -------
        complex
            Acoustic impedance in Pa*s/m^3.
        """
        if Z_load is None:
            return self.blocked_acoustic_impedance(omega)

        Z_e = self.electrical_impedance(omega)
        Z_m = self.mechanical_impedance(omega)

        # Total electrical impedance
        Z_total = Z_e + Z_load

        # Motional contribution from electromagnetic coupling
        Z_motional = (self._Bl**2) / (Z_total * self._A_d**2)

        return Z_m / (self._A_d**2) + Z_motional

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
        Z_load: complex | None = None,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the transducer.

        The transducer acts as an acoustic impedance that relates pressure
        and velocity. For a passive transducer (no external voltage drive),
        the acoustic impedance is applied:
            p1_out = p1_in - Z_a * U1_in

        where Z_a is the total acoustic impedance including motional effects.

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
        Z_load : complex, optional
            Electrical load impedance (Ohm). If None, assumes open circuit.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
        """
        # Calculate the acoustic impedance
        Z_a = self.acoustic_impedance(omega, Z_load)

        # The transducer introduces a pressure drop proportional to velocity
        # (similar to an acoustic impedance)
        p1_out = p1_in - Z_a * U1_in

        # Volumetric velocity is continuous through the transducer
        U1_out = U1_in

        # Temperature is unchanged
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def propagate_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        I1: complex,
    ) -> tuple[complex, complex, float, complex]:
        """
        Propagate acoustic state through a current-driven transducer (IESPEAKER).

        When the transducer is driven by a current source I1, the pressure
        equation becomes (reference baseline convention):
            p1_out = p1_in - Bl * I1 / A_d + Z_m * U1_in / A_d^2

        This models the electromechanical coupling where:
        - The Lorentz force (Bl * I) creates a pressure source
        - The mechanical impedance (Z_m) provides back-pressure from motion

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
        I1 : complex
            Complex current amplitude (A).

        Returns
        -------
        tuple[complex, complex, float, complex]
            Tuple of (p1_out, U1_out, T_m_out, V1):
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
            - V1: Complex voltage amplitude (V)
        """
        # Calculate mechanical impedance
        Z_m = self.mechanical_impedance(omega)

        # Pressure source from Lorentz force (negative in reference baseline convention)
        p_source = -self._Bl * I1 / self._A_d

        # Back-pressure from mechanical impedance
        p_impedance = Z_m * U1_in / (self._A_d**2)

        # Output pressure
        p1_out = p1_in + p_source + p_impedance

        # Volumetric velocity is continuous
        U1_out = U1_in

        # Temperature is unchanged
        T_m_out = T_m

        # Calculate voltage: V = Z_e * I + Bl * v, where v = U / A_d
        Z_e = self.electrical_impedance(omega)
        v1 = U1_in / self._A_d  # Diaphragm velocity
        V1 = Z_e * I1 + self._Bl * v1

        return p1_out, U1_out, T_m_out, V1

    def propagate_voltage_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        V1: complex,
    ) -> tuple[complex, complex, float, complex]:
        """
        Propagate acoustic state through a voltage-driven transducer (VESPEAKER).

        When the transducer is driven by a voltage source V1, the current is
        computed from the electrical equation:
            V1 = Z_e * I1 + Bl * (U1 / A_d)

        Solving for I1:
            I1 = (V1 - Bl * U1 / A_d) / Z_e

        The pressure equation is then applied (same as IESPEAKER):
            p1_out = p1_in - Bl * I1 / A_d + Z_m * U1 / A_d^2

        This models reference baseline's VESPEAKER segment where the driving voltage
        is specified and the current is computed from the electromechanical
        coupling equations.

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
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        tuple[complex, complex, float, complex]
            Tuple of (p1_out, U1_out, T_m_out, I1):
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
            - I1: Complex current amplitude (A)

        Notes
        -----
        This implements reference baseline Equations  with:
        - Z_e = R_e + j*omega*L_e (electrical impedance)
        - tau = -Bl/A_d (transduction coefficient)
        - Z = Z_m/A_d^2 (acoustic impedance from mechanical side)

        The VESPEAKER is useful for modeling loudspeakers driven by voltage
        amplifiers where the voltage is controlled but current varies with
        the mechanical load.

        Examples
        --------
        >>> from openthermoacoustics.segments import Transducer
        >>> from openthermoacoustics.gas import Helium
        >>> trans = Transducer(Bl=5.0, R_e=6.0, L_e=0.5e-3, m=0.01, k=2000, R_m=1.0, A_d=0.01)
        >>> gas = Helium(mean_pressure=101325)
        >>> p1_out, U1_out, T_out, I1 = trans.propagate_voltage_driven(
        ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas, V1=10+0j
        ... )
        """
        # Calculate electrical impedance
        Z_e = self.electrical_impedance(omega)

        # Calculate mechanical impedance
        Z_m = self.mechanical_impedance(omega)

        # Diaphragm velocity
        v1 = U1_in / self._A_d

        # Solve for current: V1 = Z_e * I1 + Bl * v1
        # Therefore: I1 = (V1 - Bl * v1) / Z_e
        I1 = (V1 - self._Bl * v1) / Z_e

        # Pressure source from Lorentz force (negative in reference baseline convention)
        p_source = -self._Bl * I1 / self._A_d

        # Back-pressure from mechanical impedance
        p_impedance = Z_m * U1_in / (self._A_d**2)

        # Output pressure
        p1_out = p1_in + p_source + p_impedance

        # Volumetric velocity is continuous
        U1_out = U1_in

        # Temperature is unchanged
        T_m_out = T_m

        return p1_out, U1_out, T_m_out, I1

    def electrical_power(self, I1: complex, V1: complex) -> float:
        """
        Calculate real electrical power from voltage and current.

        Uses the amplitude convention (divide by 2 for time-averaged power).

        Parameters
        ----------
        I1 : complex
            Complex current amplitude (A).
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        float
            Time-averaged electrical power (W). Positive = power consumed.
        """
        return 0.5 * (V1 * np.conj(I1)).real

    def __repr__(self) -> str:
        """Return string representation of the transducer."""
        return (
            f"Transducer(name='{self._name}', Bl={self._Bl}, R_e={self._R_e}, "
            f"L_e={self._L_e}, m={self._m}, k={self._k}, R_m={self._R_m}, A_d={self._A_d})"
        )
