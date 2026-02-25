"""Parallel plate (slit) geometry for thermoacoustic systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.geometry.base import (
    ComplexOrArray,
    FloatOrArray,
    Geometry,
)

# Threshold for switching between exact formula and asymptotic/small-z limits
_SMALL_Z_THRESHOLD = 1e-6
_LARGE_Z_THRESHOLD = 50.0


class ParallelPlate(Geometry):
    """
    Parallel plate (slit) pore geometry.

    For parallel plates separated by a gap of 2*y0 (half-gap = y0), the
    thermoviscous function is:

        f = tanh(z) / z

    where z = y0 * (1 + j) / delta.

    The hydraulic radius for parallel plates is y0 (the half-gap).

    Parameters
    ----------
    None

    Notes
    -----
    Limiting behaviors:
    - Small |z| (wide gap or low frequency): f -> 1 - z^2/3 ~ 1
    - Large |z| (narrow gap or high frequency): f -> 1/z = (1-j)*delta/y0

    The full gap between plates is 2*y0. When specifying hydraulic_radius,
    use y0 (the half-gap), not the full gap.

    References
    ----------
    .. [1] Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective for
           Some Engines and Refrigerators. ASA Press.
    .. [2] Rott, N. (1969). Damped and thermally driven acoustic oscillations
           in wide and narrow tubes. Z. Angew. Math. Phys., 20, 230-243.
    """

    @property
    def name(self) -> str:
        """
        Return the name of this geometry type.

        Returns
        -------
        str
            The string "parallel_plate".
        """
        return "parallel_plate"

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
            Half-gap y0 between plates in meters.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            The thermoviscous function f.
        """
        # Compute dimensionless argument z = y0 * (1+j) / delta
        delta = np.asarray(delta)
        y0 = hydraulic_radius
        z = y0 * (1 + 1j) / delta

        # Handle scalar vs array inputs
        scalar_input = delta.ndim == 0
        z = np.atleast_1d(z)
        z_abs = np.abs(z)

        # Initialize result array
        result = np.zeros_like(z, dtype=np.complex128)

        # Small |z| limit: tanh(z)/z -> 1 - z^2/3 + 2z^4/15 - ... ~ 1
        small_mask = z_abs < _SMALL_Z_THRESHOLD
        result[small_mask] = 1.0

        # Large |z| asymptotic: tanh(z) -> 1, so f -> 1/z
        # More precisely: tanh(z) = 1 - 2*exp(-2z) + O(exp(-4z))
        # For complex z with positive real part, this converges
        large_mask = z_abs > _LARGE_Z_THRESHOLD
        z_large = z[large_mask]
        result[large_mask] = 1.0 / z_large

        # Intermediate |z|: use exact formula f = tanh(z) / z
        mid_mask = ~small_mask & ~large_mask
        z_mid = z[mid_mask]

        if np.any(mid_mask):
            # Compute tanh for complex argument
            # tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
            # For numerical stability, use np.tanh which handles complex args
            tanh_z = np.tanh(z_mid)
            result[mid_mask] = tanh_z / z_mid

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
        Calculate the viscous thermoviscous function f_nu for parallel plates.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        hydraulic_radius : float
            Half-gap y0 between plates (m). Full gap is 2*y0.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu.

        Examples
        --------
        >>> plates = ParallelPlate()
        >>> # For a 1mm full gap (y0 = 0.5mm)
        >>> f = plates.f_nu(omega=1000.0, delta_nu=1e-4, hydraulic_radius=5e-4)
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
        Calculate the thermal thermoviscous function f_kappa for parallel plates.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Half-gap y0 between plates (m). Full gap is 2*y0.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex thermal function f_kappa.

        Examples
        --------
        >>> plates = ParallelPlate()
        >>> # For a 1mm full gap (y0 = 0.5mm)
        >>> f = plates.f_kappa(omega=1000.0, delta_kappa=1.2e-4, hydraulic_radius=5e-4)
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
        Compute both viscous and thermal functions for parallel plates.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s).
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Half-gap y0 between plates (m).

        Returns
        -------
        tuple[ComplexOrArray, ComplexOrArray]
            Tuple of (f_nu, f_kappa).
        """
        f_nu = self._compute_f(delta_nu, hydraulic_radius)
        f_kappa = self._compute_f(delta_kappa, hydraulic_radius)
        return f_nu, f_kappa
