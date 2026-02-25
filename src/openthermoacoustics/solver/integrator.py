"""ODE integration through thermoacoustic network segments.

This module provides functions for integrating the acoustic wave equations
through individual segments of a thermoacoustic network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from openthermoacoustics.utils import (
    acoustic_power,
    complex_to_state,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.segments.base import Segment


def integrate_segment(
    segment: Segment,
    p1_in: complex,
    U1_in: complex,
    T_m: float,
    omega: float,
    gas: Gas,
    n_points: int = 100,
) -> dict[str, NDArray[Any]]:
    """
    Integrate the acoustic wave equations through a segment.

    This function uses scipy's solve_ivp with the RK45 method to integrate
    the coupled ODEs for complex pressure amplitude p1 and volumetric velocity
    amplitude U1 through the segment.

    Parameters
    ----------
    segment : Segment
        The acoustic segment to integrate through.
    p1_in : complex
        Complex pressure amplitude at the segment inlet (Pa).
    U1_in : complex
        Complex volumetric velocity amplitude at the segment inlet (m^3/s).
    T_m : float
        Mean temperature at the segment inlet (K).
    omega : float
        Angular frequency (rad/s).
    gas : Gas
        Gas properties object.
    n_points : int, optional
        Number of output points along the segment, by default 100.

    Returns
    -------
    dict[str, NDArray[Any]]
        Dictionary containing arrays at each integration point:
        - 'x': position along segment (m)
        - 'p1': complex pressure amplitude (Pa)
        - 'U1': complex volumetric velocity amplitude (m^3/s)
        - 'T_m': mean temperature (K)
        - 'acoustic_power': time-averaged acoustic power (W)

    Raises
    ------
    RuntimeError
        If the ODE integration fails to converge.

    Notes
    -----
    The state vector y contains:
    - y[0:4] = [Re(p1), Im(p1), Re(U1), Im(U1)] for isothermal segments
    - y[0:5] = [Re(p1), Im(p1), Re(U1), Im(U1), T_m] for segments with
      temperature gradients (e.g., stacks)

    The governing equations are the linearized acoustic wave equations
    in the thermoacoustic approximation. For a duct without temperature
    gradient, these reduce to the standard lossless wave equations with
    viscous and thermal losses included via the thermoviscous functions.
    """
    length = segment.length

    if length <= 0:
        # For zero-length segments (lumped elements), return single-point result
        p1_out, U1_out = segment.transfer(p1_in, U1_in, T_m, omega, gas)
        return {
            "x": np.array([0.0]),
            "p1": np.array([p1_out]),
            "U1": np.array([U1_out]),
            "T_m": np.array([T_m]),
            "acoustic_power": np.array([acoustic_power(p1_out, U1_out)]),
        }

    # Check if segment has temperature gradient
    has_temperature_gradient = _segment_has_temperature_gradient(segment)

    # Build initial state vector
    if has_temperature_gradient:
        y0 = np.zeros(5)
        y0[0:4] = complex_to_state(p1_in, U1_in)
        y0[4] = T_m
    else:
        y0 = complex_to_state(p1_in, U1_in)

    # Define evaluation points
    x_eval = np.linspace(0, length, n_points)

    # Define the ODE system
    def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Right-hand side of the acoustic wave ODE."""
        return _compute_derivatives(
            x, y, segment, omega, gas, has_temperature_gradient
        )

    # Integrate using RK45
    try:
        solution = solve_ivp(
            ode_func,
            t_span=(0, length),
            y0=y0,
            method="RK45",
            t_eval=x_eval,
            dense_output=False,
            rtol=1e-8,
            atol=1e-10,
        )
    except Exception as e:
        raise RuntimeError(
            f"ODE integration failed for segment {segment!r}: {e}"
        ) from e

    if not solution.success:
        raise RuntimeError(
            f"ODE integration did not converge for segment {segment!r}: "
            f"{solution.message}"
        )

    # Extract results
    x_result = solution.t
    y_result = solution.y

    # Convert state vectors to complex amplitudes
    n_pts = len(x_result)
    p1_result = np.zeros(n_pts, dtype=complex)
    U1_result = np.zeros(n_pts, dtype=complex)

    if has_temperature_gradient:
        T_m_result = y_result[4, :]
    else:
        T_m_result = np.full(n_pts, T_m)

    for i in range(n_pts):
        p1_result[i], U1_result[i] = state_to_complex(y_result[:4, i])

    # Compute acoustic power at each point
    power_result = np.array(
        [acoustic_power(p1_result[i], U1_result[i]) for i in range(n_pts)]
    )

    return {
        "x": x_result,
        "p1": p1_result,
        "U1": U1_result,
        "T_m": T_m_result,
        "acoustic_power": power_result,
    }


def _segment_has_temperature_gradient(segment: Segment) -> bool:
    """
    Check if a segment has a temperature gradient.

    Parameters
    ----------
    segment : Segment
        The segment to check.

    Returns
    -------
    bool
        True if the segment has a temperature gradient, False otherwise.
    """
    # Check for temperature gradient attribute or specific segment types
    if hasattr(segment, "has_temperature_gradient"):
        return bool(segment.has_temperature_gradient)
    if hasattr(segment, "dT_dx"):
        return segment.dT_dx != 0
    # Default: check segment type name as fallback
    segment_type = type(segment).__name__.lower()
    return segment_type in ("stack", "regenerator", "heatexchanger")


def _compute_derivatives(
    x: float,
    y: NDArray[np.float64],
    segment: Segment,
    omega: float,
    gas: Gas,
    has_temperature_gradient: bool,
) -> NDArray[np.float64]:
    """
    Compute the derivatives of the state vector for the ODE.

    Parameters
    ----------
    x : float
        Position along the segment (m).
    y : NDArray[np.float64]
        Current state vector.
    segment : Segment
        The acoustic segment.
    omega : float
        Angular frequency (rad/s).
    gas : Gas
        Gas properties object.
    has_temperature_gradient : bool
        Whether the segment has a temperature gradient.

    Returns
    -------
    NDArray[np.float64]
        Derivative of the state vector.

    Notes
    -----
    The governing equations in the low-amplitude limit are:

        dp1/dx = -i*omega*rho_m / (A * (1 - f_nu)) * U1
        dU1/dx = -i*omega*A / (rho_m * a^2) * [1 + (gamma-1)*f_kappa] * p1
                 + (f_kappa - f_nu) / ((1 - f_nu)*(1 - Pr)) * (1/T_m) * dT_m/dx * U1

    where f_nu and f_kappa are the thermoviscous functions, gamma is the
    ratio of specific heats, Pr is the Prandtl number, and a is the sound speed.

    For segments with temperature gradients, an additional equation for T_m
    may be included (typically prescribed or computed from energy balance).
    """
    # Extract complex amplitudes from state
    p1, U1 = state_to_complex(y)

    # Get local temperature
    if has_temperature_gradient:
        T_m = y[4]
    else:
        T_m = segment.T_m if hasattr(segment, "T_m") else 300.0

    # Get gas properties at local temperature
    rho_m = gas.density(T_m)
    a = gas.sound_speed(T_m)
    gamma = gas.gamma(T_m)
    Pr = gas.prandtl(T_m)

    # Get segment properties at current position
    A = segment.area(x) if callable(getattr(segment, "area", None)) else segment.area

    # Get thermoviscous functions from segment
    f_nu, f_kappa = _get_thermoviscous_functions(segment, x, omega, gas, T_m)

    # Compute derivatives of complex amplitudes
    # Momentum equation: dp1/dx = -i*omega*rho_m / (A * (1 - f_nu)) * U1
    denom_nu = 1 - f_nu
    if abs(denom_nu) < 1e-12:
        denom_nu = 1e-12 + 0j  # Avoid division by zero

    dp1_dx = -1j * omega * rho_m / (A * denom_nu) * U1

    # Continuity equation with thermal effects
    # dU1/dx = -i*omega*A / (rho_m * a^2) * [1 + (gamma-1)*f_kappa] * p1
    thermal_factor = 1 + (gamma - 1) * f_kappa
    dU1_dx = -1j * omega * A / (rho_m * a**2) * thermal_factor * p1

    # Add temperature gradient term if present
    if has_temperature_gradient:
        dT_dx = _get_temperature_gradient(segment, x)
        if abs(dT_dx) > 1e-12 and abs(1 - Pr) > 1e-12:
            gradient_term = (f_kappa - f_nu) / (denom_nu * (1 - Pr))
            gradient_term *= (1 / T_m) * dT_dx * U1
            dU1_dx += gradient_term

    # Build derivative vector
    if has_temperature_gradient:
        dydt = np.zeros(5)
        dT_dx = _get_temperature_gradient(segment, x)
        dydt[4] = dT_dx  # Temperature evolution
    else:
        dydt = np.zeros(4)

    # Convert complex derivatives to real state derivatives
    dydt[0] = dp1_dx.real
    dydt[1] = dp1_dx.imag
    dydt[2] = dU1_dx.real
    dydt[3] = dU1_dx.imag

    return dydt


def _get_thermoviscous_functions(
    segment: Segment,
    x: float,
    omega: float,
    gas: Gas,
    T_m: float,
) -> tuple[complex, complex]:
    """
    Get the thermoviscous functions f_nu and f_kappa for a segment.

    Parameters
    ----------
    segment : Segment
        The acoustic segment.
    x : float
        Position along the segment (m).
    omega : float
        Angular frequency (rad/s).
    gas : Gas
        Gas properties object.
    T_m : float
        Mean temperature (K).

    Returns
    -------
    tuple[complex, complex]
        The viscous (f_nu) and thermal (f_kappa) thermoviscous functions.
    """
    # Try to get functions from segment's geometry
    if hasattr(segment, "geometry") and segment.geometry is not None:
        geometry = segment.geometry
        if hasattr(geometry, "f_nu") and hasattr(geometry, "f_kappa"):
            # Get penetration depths
            rho = gas.density(T_m)
            mu = gas.viscosity(T_m)
            kappa = gas.thermal_conductivity(T_m)
            cp = gas.specific_heat_cp(T_m)

            delta_nu = np.sqrt(2 * mu / (rho * omega))
            delta_kappa = np.sqrt(2 * kappa / (rho * cp * omega))

            f_nu = geometry.f_nu(delta_nu)
            f_kappa = geometry.f_kappa(delta_kappa)
            return f_nu, f_kappa

    # Try segment's own thermoviscous function methods
    if hasattr(segment, "f_nu") and hasattr(segment, "f_kappa"):
        f_nu = segment.f_nu(x, omega, gas, T_m)
        f_kappa = segment.f_kappa(x, omega, gas, T_m)
        return f_nu, f_kappa

    # Default: lossless case (no boundary layer effects)
    return 0j, 0j


def _get_temperature_gradient(segment: Segment, x: float) -> float:
    """
    Get the temperature gradient dT_m/dx at position x.

    Parameters
    ----------
    segment : Segment
        The acoustic segment.
    x : float
        Position along the segment (m).

    Returns
    -------
    float
        Temperature gradient in K/m.
    """
    if hasattr(segment, "dT_dx"):
        if callable(segment.dT_dx):
            return float(segment.dT_dx(x))
        return float(segment.dT_dx)

    if hasattr(segment, "T_hot") and hasattr(segment, "T_cold"):
        # Linear temperature gradient
        if segment.length > 0:
            return (segment.T_hot - segment.T_cold) / segment.length
        return 0.0

    return 0.0
