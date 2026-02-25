"""Rectangular pore geometry for thermoacoustic systems (STKRECT equivalent)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.geometry.base import (
    ComplexOrArray,
    FloatOrArray,
    Geometry,
)

# Default number of terms in the series expansion
_DEFAULT_N_TERMS = 20

# Threshold for switching to asymptotic limit
_SMALL_Z_THRESHOLD = 1e-6
_LARGE_Z_THRESHOLD = 50.0


class RectangularPore(Geometry):
    """
    Rectangular pore geometry for thermoacoustic systems.

    Models a rectangular duct with half-widths a and b. The thermoviscous
    functions are computed using a double series expansion.

    Parameters
    ----------
    half_width_a : float
        Half-width in the first direction (m). This is the 'aa' parameter
        in reference baseline STKRECT.
    half_width_b : float, optional
        Half-width in the second direction (m). This is the 'bb' parameter
        in reference baseline STKRECT. If not provided, defaults to half_width_a
        (square pore).
    n_terms : int, optional
        Number of terms in each direction of the series expansion.
        Default is 20, which provides good convergence for most cases.

    Attributes
    ----------
    half_width_a : float
        Half-width in first direction (m).
    half_width_b : float
        Half-width in second direction (m).
    hydraulic_radius : float
        Computed hydraulic radius: r_h = a*b / (a + b).

    Notes
    -----
    The thermoviscous function for a rectangular duct is:

        f = 1 - (64/pi^4) * sum_{m,n=1,3,5,...} 1/(m^2*n^2) * g_mn

    where g_mn = 1 / (1 + (z/lambda_mn)^2) and:
        - z = (j-1) * r_h / delta (reference baseline sign convention for closed pores)
        - lambda_mn = r_h * sqrt((m*pi/(2*a))^2 + (n*pi/(2*b))^2)

    Note: The sign convention z = (j-1)*r_h/delta matches reference baseline's convention
    for closed pore geometries (circular, rectangular), which differs from
    parallel plates that use z = (1+j)*y0/delta.

    The hydraulic radius for a rectangular duct is:
        r_h = a * b / (a + b)

    Special cases:
        - Square duct (a = b): r_h = a/2
        - Elongated rectangle (a << b): r_h → a (approaches parallel plates)

    References
    ----------
    .. [1] Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective for
           Some Engines and Refrigerators. ASA Press.
    .. [2] published literature, STKRECT segment.

    Examples
    --------
    >>> pore = RectangularPore(half_width_a=0.5e-3, half_width_b=1.0e-3)
    >>> print(f"Hydraulic radius: {pore.hydraulic_radius*1e6:.1f} µm")
    Hydraulic radius: 333.3 µm
    """

    def __init__(
        self,
        half_width_a: float,
        half_width_b: float | None = None,
        n_terms: int = _DEFAULT_N_TERMS,
    ) -> None:
        """
        Initialize a rectangular pore geometry.

        Parameters
        ----------
        half_width_a : float
            Half-width in the first direction (m).
        half_width_b : float, optional
            Half-width in the second direction (m). Defaults to half_width_a.
        n_terms : int, optional
            Number of terms in series expansion. Default is 20.

        Raises
        ------
        ValueError
            If half_width_a or half_width_b is not positive.
        """
        if half_width_a <= 0:
            raise ValueError(f"half_width_a must be positive, got {half_width_a}")

        if half_width_b is None:
            half_width_b = half_width_a

        if half_width_b <= 0:
            raise ValueError(f"half_width_b must be positive, got {half_width_b}")

        self._half_width_a = half_width_a
        self._half_width_b = half_width_b
        self._n_terms = n_terms

        # Compute hydraulic radius: r_h = a*b / (a + b)
        self._hydraulic_radius = (half_width_a * half_width_b) / (
            half_width_a + half_width_b
        )

        # Precompute the odd indices for the series
        self._odd_indices = np.arange(1, 2 * n_terms, 2)  # 1, 3, 5, ...

        # Precompute lambda_mn / r_h for all (m, n) combinations
        # lambda_mn = r_h * sqrt((m*pi/(2*a))^2 + (n*pi/(2*b))^2)
        # lambda_mn / r_h = sqrt((m*pi/(2*a))^2 + (n*pi/(2*b))^2)
        m_vals = self._odd_indices[:, np.newaxis]  # Shape: (n_terms, 1)
        n_vals = self._odd_indices[np.newaxis, :]  # Shape: (1, n_terms)

        term_a = (m_vals * np.pi / (2 * half_width_a)) ** 2
        term_b = (n_vals * np.pi / (2 * half_width_b)) ** 2
        self._lambda_over_rh = np.sqrt(term_a + term_b)

        # Precompute the coefficients 1/(m^2 * n^2)
        self._coefficients = 1.0 / (m_vals**2 * n_vals**2)

        # Normalization factor
        self._norm = 64.0 / np.pi**4

    @property
    def name(self) -> str:
        """
        Return the name of this geometry type.

        Returns
        -------
        str
            The string "rectangular".
        """
        return "rectangular"

    @property
    def half_width_a(self) -> float:
        """
        Half-width in the first direction.

        Returns
        -------
        float
            Half-width a in meters.
        """
        return self._half_width_a

    @property
    def half_width_b(self) -> float:
        """
        Half-width in the second direction.

        Returns
        -------
        float
            Half-width b in meters.
        """
        return self._half_width_b

    @property
    def hydraulic_radius(self) -> float:
        """
        Hydraulic radius of the rectangular pore.

        r_h = a * b / (a + b)

        Returns
        -------
        float
            Hydraulic radius in meters.
        """
        return self._hydraulic_radius

    @property
    def aspect_ratio(self) -> float:
        """
        Aspect ratio of the rectangular pore (b/a).

        Returns
        -------
        float
            Aspect ratio, dimensionless. 1.0 for square pores.
        """
        return self._half_width_b / self._half_width_a

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
            Hydraulic radius in meters. Note: this parameter is provided
            for API consistency but the internal precomputed r_h is used.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            The thermoviscous function f.
        """
        delta = np.asarray(delta)
        scalar_input = delta.ndim == 0
        delta = np.atleast_1d(delta)

        # Compute z = (j-1) * r_h / delta for each delta value
        # reference baseline uses z = (i-1)*r_h/delta for closed pore geometries (circular, rectangular)
        # This differs from parallel plates which use (1+j)*y0/delta
        r_h = self._hydraulic_radius
        z = r_h * (-1 + 1j) / delta  # Shape: (n_delta,)

        z_abs = np.abs(z)

        # Initialize result
        result = np.zeros(len(delta), dtype=np.complex128)

        # Handle different regimes
        small_mask = z_abs < _SMALL_Z_THRESHOLD
        large_mask = z_abs > _LARGE_Z_THRESHOLD
        mid_mask = ~small_mask & ~large_mask

        # Small |z| limit: f -> 1 (inviscid limit)
        result[small_mask] = 1.0

        # Large |z| asymptotic (boundary layer limit):
        # f -> (1-j) * delta / r_h = 2 / z
        # (same as other geometries in the boundary layer limit)
        result[large_mask] = 2.0 / z[large_mask]

        # Intermediate |z|: use series expansion
        if np.any(mid_mask):
            z_mid = z[mid_mask]

            # Compute f = 1 - (64/pi^4) * sum_{m,n odd} 1/(m^2*n^2) * g_mn
            # where g_mn = 1 / (1 + (z/lambda_mn)^2)
            # and z/lambda_mn = z / (r_h * lambda_over_rh) = z / (r_h * L)
            #                 = z * (1/r_h) * (1/L) = (z/r_h) / L
            # But z = (1+j) * r_h / delta, so z/r_h = (1+j)/delta
            # and z/lambda_mn = (1+j) / (delta * lambda_over_rh)

            # For each z value, compute the series sum
            f_mid = np.zeros(len(z_mid), dtype=np.complex128)

            for i, z_val in enumerate(z_mid):
                # z / lambda_mn = z_val / (r_h * lambda_over_rh)
                # But our precomputed lambda_over_rh already has r_h factored out
                # So z / lambda_mn = z_val / (r_h * (lambda_mn/r_h))
                #                  = z_val / lambda_mn
                # where lambda_mn = r_h * lambda_over_rh
                z_over_lambda = z_val / (r_h * self._lambda_over_rh)

                # g_mn = 1 / (1 + (z/lambda_mn)^2)
                g_mn = 1.0 / (1.0 + z_over_lambda**2)

                # Sum: sum_{m,n} coeff_mn * g_mn
                series_sum = np.sum(self._coefficients * g_mn)

                # f = 1 - norm * sum
                f_mid[i] = 1.0 - self._norm * series_sum

            result[mid_mask] = f_mid

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
        Calculate the viscous thermoviscous function f_nu for rectangular pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        hydraulic_radius : float
            Hydraulic radius (m). Note: the internal computed value is used.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu.

        Examples
        --------
        >>> pore = RectangularPore(half_width_a=0.5e-3, half_width_b=1.0e-3)
        >>> f = pore.f_nu(omega=1000.0, delta_nu=1e-4, hydraulic_radius=pore.hydraulic_radius)
        """
        return self._compute_f(delta_nu, hydraulic_radius)

    def f_kappa(
        self,
        omega: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the thermal thermoviscous function f_kappa for rectangular pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Hydraulic radius (m). Note: the internal computed value is used.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex thermal function f_kappa.
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
        Compute both viscous and thermal functions for rectangular pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s).
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Hydraulic radius (m).

        Returns
        -------
        tuple[ComplexOrArray, ComplexOrArray]
            Tuple of (f_nu, f_kappa).
        """
        f_nu = self._compute_f(delta_nu, hydraulic_radius)
        f_kappa = self._compute_f(delta_kappa, hydraulic_radius)
        return f_nu, f_kappa

    def __repr__(self) -> str:
        """Return string representation of the rectangular pore geometry."""
        return (
            f"RectangularPore(half_width_a={self._half_width_a}, "
            f"half_width_b={self._half_width_b}, "
            f"r_h={self._hydraulic_radius:.6e})"
        )
