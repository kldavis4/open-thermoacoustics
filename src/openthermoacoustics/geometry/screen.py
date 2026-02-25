"""Wire screen (mesh) geometry for thermoacoustic systems."""

from __future__ import annotations

from typing import Optional

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


class WireScreen(Geometry):
    """
    Wire screen (mesh) pore geometry.

    Wire screens are modeled using the parallel plate approximation with an
    effective hydraulic radius computed from the screen properties.

    The thermoviscous function is:

        f = tanh(z) / z

    where z = r_h * (1 + j) / delta.

    The hydraulic radius is computed as:

        r_h = (porosity * wire_diameter) / (4 * (1 - porosity))

    And the porosity (open area fraction) for a square mesh is:

        porosity = 1 - (pi/4) * (mesh_count * wire_diameter)^2

    Parameters
    ----------
    wire_diameter : float, optional
        Diameter of the wire (m). Required if not providing porosity and
        hydraulic_radius directly.
    mesh_count : float, optional
        Number of wires per unit length (1/m). Required with wire_diameter
        if not providing porosity and hydraulic_radius directly.
    porosity : float, optional
        Open area fraction (0 < porosity < 1). Can be provided directly
        instead of computing from wire_diameter and mesh_count.
    hydraulic_radius : float, optional
        Hydraulic radius (m). Can be provided directly instead of computing
        from other parameters.

    Raises
    ------
    ValueError
        If insufficient parameters are provided or if computed porosity
        is out of valid range.

    Notes
    -----
    Two ways to construct a WireScreen:

    1. From physical parameters:
       WireScreen(wire_diameter=d, mesh_count=n)
       - Porosity and hydraulic radius are computed automatically

    2. From derived parameters:
       WireScreen(porosity=phi, hydraulic_radius=r_h)
       - Use when porosity/r_h are known directly

    The parallel plate approximation works well when the penetration depths
    are small compared to the pore dimensions.

    References
    ----------
    .. [1] Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective for
           Some Engines and Refrigerators. ASA Press.
    .. [2] Gedeon, D. (1999). DC Gas Flows in Stirling and Pulse Tube
           Cryocoolers. Cryocoolers 9, Plenum Press.

    Examples
    --------
    >>> # From physical parameters (200 mesh, 0.05mm wire)
    >>> screen = WireScreen(wire_diameter=5e-5, mesh_count=200/0.0254)  # 200 per inch
    >>> print(f"Porosity: {screen.porosity:.3f}")
    >>> print(f"Hydraulic radius: {screen.hydraulic_radius*1e6:.1f} um")

    >>> # From derived parameters
    >>> screen = WireScreen(porosity=0.7, hydraulic_radius=5e-5)
    """

    def __init__(
        self,
        wire_diameter: Optional[float] = None,
        mesh_count: Optional[float] = None,
        porosity: Optional[float] = None,
        hydraulic_radius: Optional[float] = None,
    ) -> None:
        """
        Initialize wire screen geometry.

        Parameters
        ----------
        wire_diameter : float, optional
            Diameter of the wire (m).
        mesh_count : float, optional
            Number of wires per unit length (1/m).
        porosity : float, optional
            Open area fraction (0 < porosity < 1).
        hydraulic_radius : float, optional
            Hydraulic radius (m).
        """
        # Case 1: Both porosity and hydraulic_radius provided directly
        if porosity is not None and hydraulic_radius is not None:
            self._porosity = porosity
            self._hydraulic_radius = hydraulic_radius
            self._wire_diameter = wire_diameter  # May be None
            self._mesh_count = mesh_count  # May be None

        # Case 2: Compute from wire_diameter and mesh_count
        elif wire_diameter is not None and mesh_count is not None:
            self._wire_diameter = wire_diameter
            self._mesh_count = mesh_count

            # Compute porosity: phi = 1 - (pi/4) * (n * d)^2
            # This assumes a square mesh where wires cross at right angles
            nd = mesh_count * wire_diameter
            self._porosity = 1.0 - (np.pi / 4.0) * nd**2

            # Validate porosity
            if self._porosity <= 0:
                raise ValueError(
                    f"Computed porosity {self._porosity:.4f} is not positive. "
                    f"The mesh is too dense (mesh_count * wire_diameter = {nd:.4f} >= {2/np.sqrt(np.pi):.4f})."
                )
            if self._porosity >= 1:
                raise ValueError(
                    f"Computed porosity {self._porosity:.4f} is >= 1. "
                    "Check wire_diameter and mesh_count values."
                )

            # Compute hydraulic radius: r_h = (phi * d) / (4 * (1 - phi))
            self._hydraulic_radius = (
                self._porosity * wire_diameter / (4.0 * (1.0 - self._porosity))
            )

        else:
            raise ValueError(
                "Must provide either (wire_diameter, mesh_count) or "
                "(porosity, hydraulic_radius) to construct WireScreen."
            )

        # Validate final values
        if self._porosity <= 0 or self._porosity >= 1:
            raise ValueError(
                f"Porosity must be in range (0, 1), got {self._porosity}"
            )
        if self._hydraulic_radius <= 0:
            raise ValueError(
                f"Hydraulic radius must be positive, got {self._hydraulic_radius}"
            )

    @property
    def name(self) -> str:
        """
        Return the name of this geometry type.

        Returns
        -------
        str
            The string "wire_screen".
        """
        return "wire_screen"

    @property
    def porosity(self) -> float:
        """
        Return the porosity (open area fraction) of the screen.

        Returns
        -------
        float
            Porosity in range (0, 1).
        """
        return self._porosity

    @property
    def hydraulic_radius(self) -> float:
        """
        Return the hydraulic radius of the screen pores.

        Returns
        -------
        float
            Hydraulic radius in meters.
        """
        return self._hydraulic_radius

    @property
    def wire_diameter(self) -> Optional[float]:
        """
        Return the wire diameter if known.

        Returns
        -------
        float or None
            Wire diameter in meters, or None if not specified.
        """
        return self._wire_diameter

    @property
    def mesh_count(self) -> Optional[float]:
        """
        Return the mesh count (wires per unit length) if known.

        Returns
        -------
        float or None
            Mesh count in 1/m, or None if not specified.
        """
        return self._mesh_count

    def _compute_f(
        self,
        delta: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Compute the thermoviscous function f using parallel plate approximation.

        Parameters
        ----------
        delta : float or NDArray
            Penetration depth (viscous or thermal) in meters.
        hydraulic_radius : float
            Hydraulic radius in meters.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            The thermoviscous function f.
        """
        # Compute dimensionless argument z = r_h * (1+j) / delta
        delta = np.asarray(delta)
        z = hydraulic_radius * (1 + 1j) / delta

        # Handle scalar vs array inputs
        scalar_input = delta.ndim == 0
        z = np.atleast_1d(z)
        z_abs = np.abs(z)

        # Initialize result array
        result = np.zeros_like(z, dtype=np.complex128)

        # Small |z| limit: tanh(z)/z -> 1
        small_mask = z_abs < _SMALL_Z_THRESHOLD
        result[small_mask] = 1.0

        # Large |z| asymptotic: f -> 1/z = (1-j)*delta/r_h
        large_mask = z_abs > _LARGE_Z_THRESHOLD
        z_large = z[large_mask]
        result[large_mask] = 1.0 / z_large

        # Intermediate |z|: use exact formula f = tanh(z) / z
        mid_mask = ~small_mask & ~large_mask
        z_mid = z[mid_mask]

        if np.any(mid_mask):
            tanh_z = np.tanh(z_mid)
            result[mid_mask] = tanh_z / z_mid

        if scalar_input:
            return complex(result[0])
        return result

    def f_nu(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        hydraulic_radius: Optional[float] = None,
    ) -> ComplexOrArray:
        """
        Calculate the viscous thermoviscous function f_nu for wire screen.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        hydraulic_radius : float, optional
            Hydraulic radius (m). If None, uses the screen's computed
            hydraulic radius.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu.

        Examples
        --------
        >>> screen = WireScreen(porosity=0.7, hydraulic_radius=5e-5)
        >>> f = screen.f_nu(omega=1000.0, delta_nu=1e-4)
        >>> print(f"f_nu = {f:.4f}")
        """
        r_h = hydraulic_radius if hydraulic_radius is not None else self._hydraulic_radius
        return self._compute_f(delta_nu, r_h)

    def f_kappa(
        self,
        omega: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: Optional[float] = None,
    ) -> ComplexOrArray:
        """
        Calculate the thermal thermoviscous function f_kappa for wire screen.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Not directly used but included for
            API consistency.
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float, optional
            Hydraulic radius (m). If None, uses the screen's computed
            hydraulic radius.

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex thermal function f_kappa.

        Examples
        --------
        >>> screen = WireScreen(porosity=0.7, hydraulic_radius=5e-5)
        >>> f = screen.f_kappa(omega=1000.0, delta_kappa=1.2e-4)
        >>> print(f"f_kappa = {f:.4f}")
        """
        r_h = hydraulic_radius if hydraulic_radius is not None else self._hydraulic_radius
        return self._compute_f(delta_kappa, r_h)

    def compute_both(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: Optional[float] = None,
    ) -> tuple[ComplexOrArray, ComplexOrArray]:
        """
        Compute both viscous and thermal functions for wire screen.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s).
        delta_nu : float or NDArray
            Viscous penetration depth (m).
        delta_kappa : float or NDArray
            Thermal penetration depth (m).
        hydraulic_radius : float, optional
            Hydraulic radius (m). If None, uses the screen's computed
            hydraulic radius.

        Returns
        -------
        tuple[ComplexOrArray, ComplexOrArray]
            Tuple of (f_nu, f_kappa).
        """
        r_h = hydraulic_radius if hydraulic_radius is not None else self._hydraulic_radius
        f_nu = self._compute_f(delta_nu, r_h)
        f_kappa = self._compute_f(delta_kappa, r_h)
        return f_nu, f_kappa

    def __repr__(self) -> str:
        """Return string representation of the wire screen geometry."""
        return (
            f"WireScreen(porosity={self._porosity:.4f}, "
            f"hydraulic_radius={self._hydraulic_radius:.2e})"
        )
