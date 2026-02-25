"""Pin array geometry for thermoacoustic systems (STKPIN equivalent).

This module provides the PinArray geometry class for computing thermoviscous
functions f_nu and f_kappa for pin array stacks and regenerators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv, yv  # Bessel functions of first and second kind

from openthermoacoustics.geometry.base import (
    ComplexOrArray,
    FloatOrArray,
    Geometry,
)

# Threshold for switching between exact formula and asymptotic limits
_SMALL_Z_THRESHOLD = 1e-6
_LARGE_Z_THRESHOLD = 100.0


class PinArray(Geometry):
    """
    Pin array geometry for thermoacoustic systems (reference baseline STKPIN equivalent).

    Models an array of cylindrical pins arranged in a hexagonal lattice,
    with gas flowing parallel to the pins in the interstitial space.

    Parameters
    ----------
    pin_radius : float
        Radius of each pin (m). This is ri in reference baseline notation.
    pin_spacing : float
        Center-to-center distance between nearest-neighbor pins (m).
        This is 2y0 in reference baseline notation.

    Attributes
    ----------
    pin_radius : float
        Radius of each pin (m).
    pin_spacing : float
        Center-to-center pin spacing (m).
    outer_radius : float
        Effective outer radius (m), computed from hexagonal lattice geometry:
        ro = pin_spacing * sqrt(sqrt(3) / (2*pi))
    porosity : float
        Gas area fraction: 1 - pi*ri²/(sqrt(3)/2 * (2y0)²)

    Notes
    -----
    For a hexagonal array of pins, the thermoviscous function is given by
    reference baseline governing relations:

        f_j = -δ_j/(i-1) * 2*r_i/(r_o² - r_i²) * N/D

    where:
        N = Y1(z_o)*J1(z_i) - J1(z_o)*Y1(z_i)
        D = Y1(z_o)*J0(z_i) - J1(z_o)*Y0(z_i)
        z_o = (i-1)*r_o/δ_j
        z_i = (i-1)*r_i/δ_j

    In Python notation, (i-1) = (-1+j), so:
        z = (-1+1j) * r / delta

    The effective outer radius r_o is determined by the hexagonal lattice
    geometry such that the gas area per pin equals pi*(r_o² - r_i²):
        r_o² = (sqrt(3)/2*pi) * (2*y0)²

    References
    ----------
    .. [1] Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective.
    .. [2] published literature, relevant reference, governing relations.
    .. [3] Keolian & Swift, JASA (2004) - Pin array stack measurements.

    Examples
    --------
    >>> pore = PinArray(pin_radius=40e-6, pin_spacing=320e-6)
    >>> print(f"Porosity: {pore.porosity:.3f}")
    >>> f = pore.f_nu(omega=1000.0, delta_nu=50e-6, hydraulic_radius=pore.hydraulic_radius)
    """

    def __init__(
        self,
        pin_radius: float,
        pin_spacing: float,
    ) -> None:
        """
        Initialize a pin array geometry.

        Parameters
        ----------
        pin_radius : float
            Radius of each pin (m).
        pin_spacing : float
            Center-to-center distance between nearest-neighbor pins (m).

        Raises
        ------
        ValueError
            If pin_radius or pin_spacing is not positive, or if pins overlap.
        """
        if pin_radius <= 0:
            raise ValueError(f"pin_radius must be positive, got {pin_radius}")
        if pin_spacing <= 0:
            raise ValueError(f"pin_spacing must be positive, got {pin_spacing}")
        if pin_spacing <= 2 * pin_radius:
            raise ValueError(
                f"pin_spacing ({pin_spacing}) must be greater than 2*pin_radius "
                f"({2*pin_radius}) to avoid overlapping pins"
            )

        self._pin_radius = pin_radius  # r_i
        self._pin_spacing = pin_spacing  # 2*y0

        # Compute effective outer radius from hexagonal lattice geometry
        # r_o² = (sqrt(3)/(2*pi)) * (2*y0)²
        # From reference baseline: r_o² = (√3/2π) * (2y0)²
        self._outer_radius = pin_spacing * np.sqrt(np.sqrt(3) / (2 * np.pi))

        # Compute porosity: gas area / total area
        # For hexagonal lattice, area per pin = (sqrt(3)/2) * (2*y0)²
        # Gas area per pin = area_per_pin - pi*ri²
        area_per_pin = (np.sqrt(3) / 2) * pin_spacing**2
        pin_area = np.pi * pin_radius**2
        self._porosity = (area_per_pin - pin_area) / area_per_pin

        # Compute hydraulic radius: r_h = 2 * A_gas / perimeter
        # For annular region: A_gas = pi*(ro² - ri²), perimeter = 2*pi*(ro + ri)
        # r_h = (ro² - ri²) / (ro + ri) = ro - ri
        # But actually for pin arrays, the conventional r_h differs...
        # Swift uses r_h = (ro² - ri²) / (2*ri) for the "inner" surface
        # reference baseline seems to use the pin surface as the characteristic length
        # Let's use the formula that gives r_h = (ro² - ri²) / (2*ri)
        self._hydraulic_radius = (
            self._outer_radius**2 - pin_radius**2
        ) / (2 * pin_radius)

    @property
    def name(self) -> str:
        """Return the name of this geometry type."""
        return "pin_array"

    @property
    def pin_radius(self) -> float:
        """Radius of each pin (m)."""
        return self._pin_radius

    @property
    def pin_spacing(self) -> float:
        """Center-to-center distance between nearest-neighbor pins (m)."""
        return self._pin_spacing

    @property
    def outer_radius(self) -> float:
        """Effective outer radius from hexagonal geometry (m)."""
        return self._outer_radius

    @property
    def porosity(self) -> float:
        """Gas area fraction (dimensionless)."""
        return self._porosity

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius (m)."""
        return self._hydraulic_radius

    def _compute_f(
        self,
        delta: FloatOrArray,
        hydraulic_radius: float,  # Not used, we use internal radii
    ) -> ComplexOrArray:
        """
        Compute the thermoviscous function f for pin array geometry.

        Uses reference baseline governing relations with Bessel functions J0, J1, Y0, Y1.

        Parameters
        ----------
        delta : float or NDArray
            Penetration depth (viscous or thermal) in meters.
        hydraulic_radius : float
            Hydraulic radius (m). Note: internal radii are used instead.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            The thermoviscous function f.
        """
        ri = self._pin_radius
        ro = self._outer_radius

        delta = np.asarray(delta)
        scalar_input = delta.ndim == 0
        delta = np.atleast_1d(delta)

        # Compute z arguments: z = (i-1)*r/delta in reference baseline notation
        # In Python: (i-1) -> (-1+1j)
        z_i = (-1 + 1j) * ri / delta  # inner (pin surface)
        z_o = (-1 + 1j) * ro / delta  # outer (effective boundary)

        # Initialize result
        result = np.zeros(len(delta), dtype=np.complex128)

        z_abs = np.abs(z_i)  # Use inner z for regime determination

        # Handle different regimes
        small_mask = z_abs < _SMALL_Z_THRESHOLD
        large_mask = z_abs > _LARGE_Z_THRESHOLD
        mid_mask = ~small_mask & ~large_mask

        # Small |z| limit: f -> 1 (inviscid limit)
        # When delta >> ro, ri, the gas is essentially isothermal/inviscid
        result[small_mask] = 1.0

        # Large |z| asymptotic: boundary layer limit
        # f -> (1-j)*delta/r_h approximately, where r_h is hydraulic radius
        # More precisely for annular geometry: f -> 2*ri/(ro² - ri²) * delta * (1-j)
        # This comes from the leading term in the Bessel function expansions
        if np.any(large_mask):
            delta_large = delta[large_mask]
            # Asymptotic form: f ≈ 2*ri/(ro² - ri²) * delta * (1-j)
            # = (1-j) * delta * 2*ri / (ro² - ri²)
            # = (1-j) * delta / r_h  (using our r_h definition)
            result[large_mask] = (1 - 1j) * delta_large / self._hydraulic_radius

        # Intermediate |z|: use exact Bessel formula (reference governing relation)
        # f_j = -δ/(i-1) * 2*ri/(ro² - ri²) * [Y1(zo)*J1(zi) - J1(zo)*Y1(zi)]
        #                                    / [Y1(zo)*J0(zi) - J1(zo)*Y0(zi)]
        if np.any(mid_mask):
            z_i_mid = z_i[mid_mask]
            z_o_mid = z_o[mid_mask]
            delta_mid = delta[mid_mask]

            # Compute Bessel functions
            J0_zi = jv(0, z_i_mid)
            J1_zi = jv(1, z_i_mid)
            J1_zo = jv(1, z_o_mid)
            Y0_zi = yv(0, z_i_mid)
            Y1_zi = yv(1, z_i_mid)
            Y1_zo = yv(1, z_o_mid)

            # Numerator: Y1(zo)*J1(zi) - J1(zo)*Y1(zi)
            numer = Y1_zo * J1_zi - J1_zo * Y1_zi

            # Denominator: Y1(zo)*J0(zi) - J1(zo)*Y0(zi)
            denom = Y1_zo * J0_zi - J1_zo * Y0_zi

            # Prefactor: -delta/(i-1) * 2*ri/(ro² - ri²)
            # Note: -1/(i-1) = -1/(-1+j) = 1/(1-j) = (1+j)/2
            # So -delta/(i-1) = delta * (1+j) / 2
            # But let's compute it directly: -delta / ((-1+1j))
            prefactor = -delta_mid / (-1 + 1j) * 2 * ri / (ro**2 - ri**2)

            # Handle potential division by zero
            safe_mask = np.abs(denom) > 1e-15
            result_mid = np.zeros(len(z_i_mid), dtype=np.complex128)

            result_mid[safe_mask] = (
                prefactor[safe_mask] * numer[safe_mask] / denom[safe_mask]
            )

            # For unsafe values, use asymptotic form
            unsafe_mask = ~safe_mask
            if np.any(unsafe_mask):
                result_mid[unsafe_mask] = (
                    (1 - 1j) * delta_mid[unsafe_mask] / self._hydraulic_radius
                )

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
        Calculate the viscous thermoviscous function f_nu for pin arrays.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used.
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        hydraulic_radius : float
            Hydraulic radius (m). Internal values are used.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu.
        """
        return self._compute_f(delta_nu, hydraulic_radius)

    def f_kappa(
        self,
        omega: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the thermal thermoviscous function f_kappa for pin arrays.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used.
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float
            Hydraulic radius (m). Internal values are used.

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
        Compute both viscous and thermal functions for pin arrays.

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
        """Return string representation of the pin array geometry."""
        return (
            f"PinArray(pin_radius={self._pin_radius:.6e}, "
            f"pin_spacing={self._pin_spacing:.6e}, "
            f"porosity={self._porosity:.4f})"
        )
