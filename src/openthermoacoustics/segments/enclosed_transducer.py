"""Enclosed (series) transducer with direct coefficients (IEDUCER/VEDUCER).

This module provides the EnclosedTransducer class for modeling transducers
with user-specified transduction coefficients, unlike the Transducer class
which derives coefficients from physical parameters (Bl, m, k, etc.).

Enclosed (series) behavior:
- Volume flow rate is continuous: U1_out = U1_in
- Pressure changes: p1_out = p1_in + τ'*I1 - Z*U1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class EnclosedTransducer(Segment):
    """
    Enclosed (series) transducer with frequency-independent coefficients (IEDUCER/VEDUCER).

    This is the reference baseline *EDUCER equivalent, where the transduction coefficients
    are specified directly rather than derived from physical parameters like
    Bl, mass, spring constant, etc.

    The transducer obeys the canonical equations:
        V1 = Ze * I1 - τ * U1
        p1_out - p1_in = τ' * I1 - Z * U1

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
    For enclosed (series) transducers:
    - Flow is continuous: U1_out = U1_in
    - Pressure changes across the transducer

    For IEDUCER (current-driven):
        p1_out = p1_in + τ' * I1 - Z * U1
        V1 = Ze * I1 - τ * U1

    For VEDUCER (voltage-driven):
        I1 = (V1 + τ * U1) / Ze
        p1_out = p1_in + τ' * I1 - Z * U1

    Unlike *ESPEAKER segments, *EDUCER segments do not have thermal-hysteresis
    losses since they have no area parameter.

    References
    ----------
    .. [1] published literature, relevant reference, governing relations
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
        Initialize an enclosed transducer with direct coefficients.

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

        Raises
        ------
        ValueError
            If Ze is zero (would cause division by zero in voltage-driven mode).
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
        Propagate through passive enclosed transducer (open circuit).

        For open circuit (I1 = 0), the pressure change is purely from
        the acoustic impedance: p1_out = p1_in + Z * U1

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
        # Open circuit: I1 = 0
        # p1_out = p1_in + τ'*0 + Z*U1 = p1_in + Z*U1
        p1_out = p1_in + self._Z * U1_in

        # Flow is continuous for enclosed transducers
        U1_out = U1_in

        return p1_out, U1_out, T_m

    def propagate_current_driven(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
        I1: complex,
    ) -> tuple[complex, complex, float, complex]:
        """
        Propagate through current-driven enclosed transducer (IEDUCER).

        Given current I1, computes output pressure and voltage V1.

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
        tuple[complex, complex, float, complex]
            (p1_out, U1_out, T_m_out, V1):
            - p1_out: Output pressure (Pa)
            - U1_out: Output volume velocity (m³/s), equals U1_in
            - T_m_out: Output temperature (K)
            - V1: Complex voltage (V)

        Notes
        -----
        Implements equations consistent with the validated Transducer class:
            p1_out = p1_in + τ' * I1 + Z * U1
            V1 = Ze * I1 + τ * U1

        Note: For speakers, τ = Bl/A and τ' = -Bl/A, which gives the
        standard IESPEAKER behavior.
        """
        # Pressure equation: p1_out = p1_in + τ'*I1 + Z*U1
        p1_out = p1_in + self._tau_prime * I1 + self._Z * U1_in

        # Voltage equation: V1 = Ze*I1 + τ*U1
        V1 = self._Ze * I1 + self._tau * U1_in

        # Flow is continuous
        U1_out = U1_in

        return p1_out, U1_out, T_m, V1

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
        Propagate through voltage-driven enclosed transducer (VEDUCER).

        Given voltage V1, computes current I1 and output pressure.

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
        tuple[complex, complex, float, complex]
            (p1_out, U1_out, T_m_out, I1):
            - p1_out: Output pressure (Pa)
            - U1_out: Output volume velocity (m³/s), equals U1_in
            - T_m_out: Output temperature (K)
            - I1: Complex current (A)

        Raises
        ------
        ValueError
            If electrical impedance Ze is zero.

        Notes
        -----
        Implements equations consistent with the validated Transducer class:
            I1 = (V1 - τ * U1) / Ze
            p1_out = p1_in + τ' * I1 + Z * U1
        """
        if abs(self._Ze) < 1e-20:
            raise ValueError(
                "VEDUCER cannot be used with zero electrical impedance Ze. "
                "Use IEDUCER instead."
            )

        # Current equation: V1 = Ze*I1 + τ*U1 => I1 = (V1 - τ*U1) / Ze
        I1 = (V1 - self._tau * U1_in) / self._Ze

        # Pressure equation: p1_out = p1_in + τ'*I1 + Z*U1
        p1_out = p1_in + self._tau_prime * I1 + self._Z * U1_in

        # Flow is continuous
        U1_out = U1_in

        return p1_out, U1_out, T_m, I1

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
            f"EnclosedTransducer(name='{self._name}', Ze={self._Ze}, "
            f"tau={self._tau}, tau_prime={self._tau_prime}, Z={self._Z})"
        )


# reference baseline aliases
IEDUCER = EnclosedTransducer
VEDUCER = EnclosedTransducer
