"""Physical constants and utility functions for OpenThermoacoustics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Universal physical constants (SI units)
R_UNIVERSAL: float = 8.314462618  # J/(mol·K) - Universal gas constant
BOLTZMANN: float = 1.380649e-23  # J/K - Boltzmann constant
AVOGADRO: float = 6.02214076e23  # 1/mol - Avogadro's number

# Reference conditions
T_REF: float = 273.15  # K - Standard temperature (0°C)
P_REF: float = 101325.0  # Pa - Standard pressure (1 atm)

# Molar masses (kg/mol)
MOLAR_MASS = {
    "helium": 4.002602e-3,
    "argon": 39.948e-3,
    "nitrogen": 28.0134e-3,
    "air": 28.9647e-3,  # Effective molar mass of dry air
    "neon": 20.1797e-3,
    "xenon": 131.293e-3,
}

# Ratio of specific heats at room temperature
GAMMA = {
    "helium": 5 / 3,  # Monatomic
    "argon": 5 / 3,  # Monatomic
    "neon": 5 / 3,  # Monatomic
    "xenon": 5 / 3,  # Monatomic
    "nitrogen": 1.4,  # Diatomic
    "air": 1.4,  # Mostly diatomic
}


def specific_gas_constant(gas_name: str) -> float:
    """
    Calculate the specific gas constant for a given gas.

    Parameters
    ----------
    gas_name : str
        Name of the gas (e.g., 'helium', 'air').

    Returns
    -------
    float
        Specific gas constant R_specific in J/(kg·K).

    Raises
    ------
    ValueError
        If the gas name is not recognized.
    """
    gas_name = gas_name.lower()
    if gas_name not in MOLAR_MASS:
        raise ValueError(
            f"Unknown gas: {gas_name}. Available gases: {list(MOLAR_MASS.keys())}"
        )
    return R_UNIVERSAL / MOLAR_MASS[gas_name]


def penetration_depth_viscous(
    omega: float, rho: float, mu: float
) -> float:
    """
    Calculate the viscous penetration depth δ_ν.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    rho : float
        Gas density (kg/m³).
    mu : float
        Dynamic viscosity (Pa·s).

    Returns
    -------
    float
        Viscous penetration depth δ_ν in meters.
    """
    nu = mu / rho  # Kinematic viscosity
    return np.sqrt(2 * nu / omega)


def penetration_depth_thermal(
    omega: float, rho: float, kappa: float, cp: float
) -> float:
    """
    Calculate the thermal penetration depth δ_κ.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    rho : float
        Gas density (kg/m³).
    kappa : float
        Thermal conductivity (W/(m·K)).
    cp : float
        Specific heat at constant pressure (J/(kg·K)).

    Returns
    -------
    float
        Thermal penetration depth δ_κ in meters.
    """
    alpha = kappa / (rho * cp)  # Thermal diffusivity
    return np.sqrt(2 * alpha / omega)


def acoustic_power(
    p1: complex, U1: complex
) -> float:
    """
    Calculate the time-averaged acoustic power.

    Parameters
    ----------
    p1 : complex
        Complex pressure amplitude (Pa).
    U1 : complex
        Complex volumetric velocity amplitude (m³/s).

    Returns
    -------
    float
        Time-averaged acoustic power Ė_2 in Watts.
    """
    return 0.5 * np.real(p1 * np.conj(U1))


def complex_to_state(p1: complex, U1: complex) -> NDArray[np.float64]:
    """
    Convert complex p1 and U1 to real state vector.

    Parameters
    ----------
    p1 : complex
        Complex pressure amplitude (Pa).
    U1 : complex
        Complex volumetric velocity amplitude (m³/s).

    Returns
    -------
    NDArray[np.float64]
        State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
    """
    return np.array([p1.real, p1.imag, U1.real, U1.imag], dtype=np.float64)


def state_to_complex(y: NDArray[np.float64]) -> tuple[complex, complex]:
    """
    Convert real state vector to complex p1 and U1.

    Parameters
    ----------
    y : NDArray[np.float64]
        State vector [Re(p1), Im(p1), Re(U1), Im(U1), ...].

    Returns
    -------
    tuple[complex, complex]
        Complex pressure amplitude p1 and volumetric velocity amplitude U1.
    """
    p1 = complex(y[0], y[1])
    U1 = complex(y[2], y[3])
    return p1, U1


def wavelength(frequency: float, sound_speed: float) -> float:
    """
    Calculate the acoustic wavelength.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    sound_speed : float
        Sound speed in m/s.

    Returns
    -------
    float
        Wavelength in meters.
    """
    return sound_speed / frequency


def wavenumber(frequency: float, sound_speed: float) -> float:
    """
    Calculate the acoustic wavenumber.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    sound_speed : float
        Sound speed in m/s.

    Returns
    -------
    float
        Wavenumber k in rad/m.
    """
    return 2 * np.pi * frequency / sound_speed
