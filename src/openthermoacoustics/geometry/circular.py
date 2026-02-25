"""Circular pore geometry for thermoacoustic systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv

from openthermoacoustics.geometry.base import (
    ComplexOrArray,
    FloatOrArray,
    Geometry,
)

# Threshold for switching between exact formula and asymptotic/small-z limits
_SMALL_Z_THRESHOLD = 1e-6
_LARGE_Z_THRESHOLD = 100.0


class CircularPore(Geometry):
    """
    Circular pore (cylindrical tube) geometry.

    For a circular pore with radius r_h, the thermoviscous function is:

        f = 2 * J1(z) / (z * J0(z))

    where z = (j - 1) * r_h / delta = (-1 + j) * r_h / delta and J0, J1
    are Bessel functions of the first kind.

    Note: reference baseline uses z = (i-1)*r0/delta for circular pores, which differs
    from the parallel plate convention of z = (1+i)*y0/delta. This sign
    convention is important for matching reference baseline's phase behavior.

    The hydraulic radius for a circular tube equals the tube radius.

    Notes
    -----
    Limiting behaviors:
    - Small |z| (wide pore or low frequency): f -> 1
    - Large |z| (narrow pore or high frequency): f -> 2/z

    References
    ----------
    .. [1] Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective for
           Some Engines and Refrigerators. ASA Press.
    .. [2] published literature, relevant reference (DUCT segment equations).
    """

    @property
    def name(self) -> str:
        """
        Return the name of this geometry type.

        Returns
        -------
        str
            The string "circular".
        """
        return "circular"

    def _compute_f(
        self,
        delta: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Compute the thermoviscous function f for given penetration depth.

        Parameters
        ----------
        delta : float or NDArray
            Penetration depth (viscous or thermal) in meters.
        hydraulic_radius : float
            Hydraulic radius (tube radius) in meters.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            The thermoviscous function f.
        """
        # Compute dimensionless argument z = (j-1) * r_h / delta
        # reference baseline uses z = (i-1)*r0/delta for circular pores (see published literature relevant equation)
        # Note: (j-1) = (-1+j) = sqrt(2) * exp(j*3*pi/4), giving argument +135°
        # This differs from parallel plates which use (1+j) with argument +45°
        delta = np.asarray(delta)
        z = hydraulic_radius * (-1 + 1j) / delta

        # Handle scalar vs array inputs
        scalar_input = delta.ndim == 0
        z = np.atleast_1d(z)
        z_abs = np.abs(z)

        # Initialize result array
        result = np.zeros_like(z, dtype=np.complex128)

        # Small |z| limit: f -> 1 - z^2/8 + O(z^4) ~ 1
        small_mask = z_abs < _SMALL_Z_THRESHOLD
        result[small_mask] = 1.0

        # Large |z| asymptotic: f -> 2/z = 2*delta / (r_h*(-1+j))
        # = 2*delta*(-1-j) / (2*r_h) = (-1-j)*delta/r_h
        large_mask = z_abs > _LARGE_Z_THRESHOLD
        z_large = z[large_mask]
        result[large_mask] = 2.0 / z_large

        # Intermediate |z|: use exact Bessel formula
        # f = 2*J1(z) / (z*J0(z))
        mid_mask = ~small_mask & ~large_mask
        z_mid = z[mid_mask]

        if np.any(mid_mask):
            j0 = jv(0, z_mid)
            j1 = jv(1, z_mid)

            # Check for numerical issues (J0 close to zero at its roots)
            # The first root of J0 is at ~2.405
            denom = z_mid * j0
            safe_mask = np.abs(denom) > 1e-15

            # For safe values, compute normally
            result_mid = np.zeros_like(z_mid, dtype=np.complex128)
            result_mid[safe_mask] = 2.0 * j1[safe_mask] / denom[safe_mask]

            # For unsafe values (near J0 roots), use asymptotic form
            # This is rare for complex z but handle it for robustness
            unsafe_mask = ~safe_mask
            if np.any(unsafe_mask):
                result_mid[unsafe_mask] = 2.0 / z_mid[unsafe_mask]

            result[mid_mask] = result_mid

        if scalar_input:
            return complex(result[0])
        return result

    def f_nu(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the viscous thermoviscous function f_nu for circular pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        hydraulic_radius : float
            Tube radius (m).

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu.

        Examples
        --------
        >>> pore = CircularPore()
        >>> f = pore.f_nu(omega=1000.0, delta_nu=1e-4, hydraulic_radius=5e-4)
        >>> print(f"f_nu = {f:.4f}")
        """
        return self._compute_f(delta_nu, hydraulic_radius)

    def f_kappa(
        self,
        omega: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the thermal thermoviscous function f_kappa for circular pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Tube radius (m).

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex thermal function f_kappa.

        Examples
        --------
        >>> pore = CircularPore()
        >>> f = pore.f_kappa(omega=1000.0, delta_kappa=1.2e-4, hydraulic_radius=5e-4)
        >>> print(f"f_kappa = {f:.4f}")
        """
        return self._compute_f(delta_kappa, hydraulic_radius)

    def compute_both(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> tuple[ComplexOrArray, ComplexOrArray]:
        """
        Compute both viscous and thermal functions for circular pores.

        Since both functions use the same formula with different penetration
        depths, this method computes them independently.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s).
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Tube radius (m).

        Returns
        -------
        tuple[ComplexOrArray, ComplexOrArray]
            Tuple of (f_nu, f_kappa).
        """
        f_nu = self._compute_f(delta_nu, hydraulic_radius)
        f_kappa = self._compute_f(delta_kappa, hydraulic_radius)
        return f_nu, f_kappa
