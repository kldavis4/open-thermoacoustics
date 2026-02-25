"""Side-branch transducer segments (IDUCER, VDUCER, ISPEAKER, VSPEAKER).

These transducers are connected as side branches to the main acoustic network,
unlike the enclosed (series) transducers which are in-line with the flow.

Side-branch behavior:
- Pressure is unchanged in the trunk: p1_out = p1_in
- Volume flow rate diverts: U1_out = U1_in - Ux

where Ux is the volume flow rate into the transducer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class SideBranchTransducer(Segment):
    """
    Generic side-branch transducer with frequency-independent parameters (IDUCER/VDUCER).

    This is the reference baseline *DUCER equivalent, where the transduction coefficients
    are specified directly rather than derived from physical parameters.

    The transducer obeys the equations:
        V1 = Ze * I1 + τ * Ux
        p1 = τ' * I1 + Z * Ux

    where Ux is the volume velocity into the transducer.

    Parameters
    ----------
    Ze : complex
        Electrical impedance (Ohm) when transducer is blocked.
    tau : complex
        Transduction coefficient τ (V·s/m³).
    tau_prime : complex
        Transduction coefficient τ' (Pa/A).
    Z : complex
        Acoustic impedance (Pa·s/m³) of the transducer.
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    For side-branch transducers:
    - Pressure is unchanged: p1_out = p1_in
    - Flow diverts: U1_out = U1_in - Ux

    For IDUCER (current-driven):
        Ux = (p1 - τ' * I1) / Z
        V1 = Ze * I1 + τ * Ux

    For VDUCER (voltage-driven):
        I1 = (Z * V1 - τ * p1) / (Ze * Z - τ * τ')
        Ux = (V1 - Ze * I1) / τ

    References
    ----------
    .. [1] published literature, relevant reference
    """

    def __init__(
        self,
        Ze: complex,
        tau: complex,
        tau_prime: complex,
        Z: complex,
        name: str = "",
    ) -> None:
        """
        Initialize a generic side-branch transducer.

        Parameters
        ----------
        Ze : complex
            Electrical impedance (Ohm).
        tau : complex
            Transduction coefficient τ (V·s/m³).
        tau_prime : complex
            Transduction coefficient τ' (Pa/A).
        Z : complex
            Acoustic impedance (Pa·s/m³).
        name : str, optional
            Name identifier.
        """
        self._Ze = complex(Ze)
        self._tau = complex(tau)
        self._tau_prime = complex(tau_prime)
        self._Z = complex(Z)

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=1.0, geometry=None)

    @property
    def Ze(self) -> complex:
        """Electrical impedance (Ohm)."""
        return self._Ze

    @property
    def tau(self) -> complex:
        """Transduction coefficient τ (V·s/m³)."""
        return self._tau

    @property
    def tau_prime(self) -> complex:
        """Transduction coefficient τ' (Pa/A)."""
        return self._tau_prime

    @property
    def Z(self) -> complex:
        """Acoustic impedance (Pa·s/m³)."""
        return self._Z

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Return zero derivatives for this lumped element.

        Parameters
        ----------
        x : float
            Axial position (m). Not used.
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        NDArray[np.float64]
            Zero vector since this is a lumped element.
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
        Propagate through passive side-branch transducer (no drive).

        For a passive side-branch, we need an external load impedance
        to determine the behavior. This method assumes open circuit
        (no current flows), so Ux = p1 / Z.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.

        Returns
        -------
        tuple[complex, complex, float]
            (p1_out, U1_out, T_m_out)
        """
        # Passive case: assume open circuit (I1 = 0)
        # Then Ux = p1 / Z from p1 = τ'*I1 + Z*Ux with I1=0
        if abs(self._Z) > 1e-20:
            Ux = p1_in / self._Z
        else:
            Ux = 0.0 + 0.0j

        # Pressure unchanged in trunk
        p1_out = p1_in

        # Flow diverts
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m

    def propagate_current_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        I1: complex,
    ) -> tuple[complex, complex, float, complex, complex]:
        """
        Propagate through current-driven side-branch transducer (IDUCER).

        Given current I1, computes transducer volume velocity Ux and voltage V1.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        I1 : complex
            Complex current amplitude (A).

        Returns
        -------
        tuple[complex, complex, float, complex, complex]
            (p1_out, U1_out, T_m_out, V1, Ux):
            - p1_out: Output pressure (Pa), equals p1_in
            - U1_out: Output volume velocity (m³/s)
            - T_m_out: Output temperature (K)
            - V1: Complex voltage (V)
            - Ux: Transducer volume velocity (m³/s)

        Raises
        ------
        ValueError
            If acoustic impedance Z is zero (would cause division by zero).
        """
        if abs(self._Z) < 1e-20:
            raise ValueError(
                "IDUCER cannot be used with zero acoustic impedance Z. "
                "Use VDUCER instead."
            )

        # Ux = (p1 - τ' * I1) / Z
        Ux = (p1_in - self._tau_prime * I1) / self._Z

        # V1 = Ze * I1 + τ * Ux
        V1 = self._Ze * I1 + self._tau * Ux

        # Pressure unchanged
        p1_out = p1_in

        # Flow diverts
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m, V1, Ux

    def propagate_voltage_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        V1: complex,
    ) -> tuple[complex, complex, float, complex, complex]:
        """
        Propagate through voltage-driven side-branch transducer (VDUCER).

        Given voltage V1, computes current I1 and transducer volume velocity Ux.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        tuple[complex, complex, float, complex, complex]
            (p1_out, U1_out, T_m_out, I1, Ux):
            - p1_out: Output pressure (Pa), equals p1_in
            - U1_out: Output volume velocity (m³/s)
            - T_m_out: Output temperature (K)
            - I1: Complex current (A)
            - Ux: Transducer volume velocity (m³/s)
        """
        # I1 = (Z * V1 - τ * p1) / (Ze * Z - τ * τ')
        denom = self._Ze * self._Z - self._tau * self._tau_prime
        if abs(denom) < 1e-20:
            raise ValueError(
                "Degenerate transducer parameters: Ze*Z - τ*τ' = 0"
            )

        I1 = (self._Z * V1 - self._tau * p1_in) / denom

        # Ux = (V1 - Ze * I1) / τ
        if abs(self._tau) < 1e-20:
            # If τ = 0, use the p1 equation: Ux = (p1 - τ'*I1) / Z
            if abs(self._Z) < 1e-20:
                Ux = 0.0 + 0.0j
            else:
                Ux = (p1_in - self._tau_prime * I1) / self._Z
        else:
            Ux = (V1 - self._Ze * I1) / self._tau

        # Pressure unchanged
        p1_out = p1_in

        # Flow diverts
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m, I1, Ux

    def electrical_power(self, I1: complex, V1: complex) -> float:
        """
        Calculate real electrical power.

        Parameters
        ----------
        I1 : complex
            Complex current amplitude (A).
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        float
            Time-averaged electrical power (W). Positive = power into transducer.
        """
        return 0.5 * np.real(V1 * np.conj(I1))

    def __repr__(self) -> str:
        return (
            f"SideBranchTransducer(name='{self._name}', Ze={self._Ze}, "
            f"tau={self._tau}, tau_prime={self._tau_prime}, Z={self._Z})"
        )


# reference baseline aliases
IDUCER = SideBranchTransducer
VDUCER = SideBranchTransducer


class SideBranchSpeaker(Segment):
    """
    Electrodynamic side-branch speaker (ISPEAKER/VSPEAKER).

    Similar to the enclosed Transducer but connected as a side branch.
    The transduction coefficients are derived from physical parameters.

    Parameters
    ----------
    area : float
        Surface area S of the moving element (m²).
    Bl : float
        Force factor (T·m).
    R_e : float
        Electrical resistance (Ohm).
    L_e : float
        Electrical inductance (H).
    m : float
        Moving mass (kg).
    k : float
        Spring constant (N/m).
    R_m : float
        Mechanical resistance (N·s/m).
    name : str, optional
        Name identifier.

    Notes
    -----
    For speakers:
        Ze = Re + j*ω*L
        τ = -τ' = Bl/S
        Z = Rm/S² + j*(ω*M - K/ω)/S²

    The speaker is treated as a side branch:
    - Pressure unchanged: p1_out = p1_in
    - Flow diverts: U1_out = U1_in - Ux

    References
    ----------
    .. [1] published literature, relevant reference
    """

    def __init__(
        self,
        area: float,
        Bl: float,
        R_e: float,
        L_e: float,
        m: float,
        k: float,
        R_m: float,
        name: str = "",
    ) -> None:
        """
        Initialize a side-branch electrodynamic speaker.

        Parameters
        ----------
        area : float
            Surface area of moving element (m²).
        Bl : float
            Force factor (T·m).
        R_e : float
            Electrical resistance (Ohm).
        L_e : float
            Electrical inductance (H).
        m : float
            Moving mass (kg).
        k : float
            Spring constant (N/m).
        R_m : float
            Mechanical resistance (N·s/m).
        name : str, optional
            Name identifier.
        """
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")
        if Bl <= 0:
            raise ValueError(f"Bl must be positive, got {Bl}")
        if R_e <= 0:
            raise ValueError(f"R_e must be positive, got {R_e}")
        if L_e < 0:
            raise ValueError(f"L_e must be non-negative, got {L_e}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if R_m < 0:
            raise ValueError(f"R_m must be non-negative, got {R_m}")

        self._area = area  # S
        self._Bl = Bl
        self._R_e = R_e
        self._L_e = L_e
        self._m = m
        self._k = k
        self._R_m = R_m

        super().__init__(name=name, length=0.0, area=area, geometry=None)

    @property
    def speaker_area(self) -> float:
        """Surface area of moving element (m²)."""
        return self._area

    @property
    def Bl(self) -> float:
        """Force factor (T·m)."""
        return self._Bl

    @property
    def R_e(self) -> float:
        """Electrical resistance (Ohm)."""
        return self._R_e

    @property
    def L_e(self) -> float:
        """Electrical inductance (H)."""
        return self._L_e

    @property
    def m(self) -> float:
        """Moving mass (kg)."""
        return self._m

    @property
    def k(self) -> float:
        """Spring constant (N/m)."""
        return self._k

    @property
    def R_m(self) -> float:
        """Mechanical resistance (N·s/m)."""
        return self._R_m

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Return zero derivatives for this lumped element.

        Parameters
        ----------
        x : float
            Axial position (m). Not used.
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        NDArray[np.float64]
            Zero vector since this is a lumped element.
        """
        return np.zeros(4, dtype=np.float64)

    def resonant_frequency(self) -> float:
        """Calculate mechanical resonant frequency (Hz)."""
        return np.sqrt(self._k / self._m) / (2 * np.pi)

    def electrical_impedance(self, omega: float) -> complex:
        """
        Calculate electrical impedance Ze = Re + j*ω*L.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Electrical impedance (Ohm).
        """
        return self._R_e + 1j * omega * self._L_e

    def acoustic_impedance(self, omega: float) -> complex:
        """
        Calculate acoustic impedance Z = Rm/S² + j*(ω*M - K/ω)/S².

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Acoustic impedance (Pa·s/m³).
        """
        S = self._area
        return self._R_m / S**2 + 1j * (omega * self._m - self._k / omega) / S**2

    def tau(self, omega: float) -> complex:
        """
        Calculate transduction coefficient τ = Bl/S.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s). Not used but included for consistency.

        Returns
        -------
        complex
            Transduction coefficient τ (V·s/m³).
        """
        return self._Bl / self._area

    def tau_prime(self, omega: float) -> complex:
        """
        Calculate transduction coefficient τ' = -Bl/S.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s). Not used but included for consistency.

        Returns
        -------
        complex
            Transduction coefficient τ' (Pa/A).
        """
        return -self._Bl / self._area

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate through passive side-branch speaker (open circuit).

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.

        Returns
        -------
        tuple[complex, complex, float]
            (p1_out, U1_out, T_m_out)
        """
        Z = self.acoustic_impedance(omega)

        # Passive: I1 = 0, so Ux = p1 / Z
        if abs(Z) > 1e-20:
            Ux = p1_in / Z
        else:
            Ux = 0.0 + 0.0j

        p1_out = p1_in
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m

    def propagate_current_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        I1: complex,
    ) -> tuple[complex, complex, float, complex, complex]:
        """
        Propagate through current-driven side-branch speaker (ISPEAKER).

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        I1 : complex
            Complex current amplitude (A).

        Returns
        -------
        tuple[complex, complex, float, complex, complex]
            (p1_out, U1_out, T_m_out, V1, Ux)
        """
        Ze = self.electrical_impedance(omega)
        Z = self.acoustic_impedance(omega)
        tau = self.tau(omega)
        tau_p = self.tau_prime(omega)

        if abs(Z) < 1e-20:
            raise ValueError(
                "ISPEAKER cannot be used with zero acoustic impedance. "
                "Use VSPEAKER instead for resonant or lossless transducers."
            )

        # Ux = (p1 - τ' * I1) / Z
        Ux = (p1_in - tau_p * I1) / Z

        # V1 = Ze * I1 + τ * Ux
        V1 = Ze * I1 + tau * Ux

        p1_out = p1_in
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m, V1, Ux

    def propagate_voltage_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        V1: complex,
    ) -> tuple[complex, complex, float, complex, complex]:
        """
        Propagate through voltage-driven side-branch speaker (VSPEAKER).

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        tuple[complex, complex, float, complex, complex]
            (p1_out, U1_out, T_m_out, I1, Ux)
        """
        Ze = self.electrical_impedance(omega)
        Z = self.acoustic_impedance(omega)
        tau = self.tau(omega)
        tau_p = self.tau_prime(omega)

        # I1 = (Z * V1 - τ * p1) / (Ze * Z - τ * τ')
        denom = Ze * Z - tau * tau_p
        if abs(denom) < 1e-20:
            raise ValueError("Degenerate speaker parameters: Ze*Z - τ*τ' = 0")

        I1 = (Z * V1 - tau * p1_in) / denom

        # Ux = (V1 - Ze * I1) / τ
        if abs(tau) < 1e-20:
            if abs(Z) < 1e-20:
                Ux = 0.0 + 0.0j
            else:
                Ux = (p1_in - tau_p * I1) / Z
        else:
            Ux = (V1 - Ze * I1) / tau

        p1_out = p1_in
        U1_out = U1_in - Ux

        return p1_out, U1_out, T_m, I1, Ux

    def electrical_power(self, I1: complex, V1: complex) -> float:
        """
        Calculate real electrical power.

        Parameters
        ----------
        I1 : complex
            Complex current amplitude (A).
        V1 : complex
            Complex voltage amplitude (V).

        Returns
        -------
        float
            Time-averaged electrical power (W).
        """
        return 0.5 * np.real(V1 * np.conj(I1))

    def __repr__(self) -> str:
        return (
            f"SideBranchSpeaker(name='{self._name}', area={self._area}, "
            f"Bl={self._Bl}, R_e={self._R_e}, L_e={self._L_e}, "
            f"m={self._m}, k={self._k}, R_m={self._R_m})"
        )


# reference baseline aliases
ISPEAKER = SideBranchSpeaker
VSPEAKER = SideBranchSpeaker
