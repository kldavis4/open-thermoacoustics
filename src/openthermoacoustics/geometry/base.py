"""Abstract base class for pore geometries in thermoacoustic systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray

# Type alias for scalar or array inputs
FloatOrArray = Union[float, NDArray[np.floating]]
ComplexOrArray = Union[complex, NDArray[np.complexfloating]]


class Geometry(ABC):
    """
    Abstract base class for pore geometry thermoviscous functions.

    All pore geometries must implement the viscous function f_nu and thermal
    function f_kappa, which characterize how viscous and thermal effects
    modify acoustic wave propagation in narrow channels.

    The functions f_nu and f_kappa depend on the ratio of the hydraulic radius
    to the penetration depth, and their form varies with pore geometry.

    Notes
    -----
    The thermoviscous functions f_nu and f_kappa satisfy:
    - f -> 1 as penetration depth -> 0 (inviscid/adiabatic limit)
    - f -> 0 as penetration depth -> infinity (fully viscous/isothermal limit)

    For most geometries, f can be written as a function of the dimensionless
    parameter z = r_h * (1 + j) / delta, where r_h is the hydraulic radius
    and delta is the penetration depth.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this geometry type.

        Returns
        -------
        str
            Human-readable name of the geometry.
        """
        pass

    @abstractmethod
    def f_nu(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the viscous thermoviscous function f_nu.

        The viscous function characterizes how viscous losses modify the
        momentum equation in narrow pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Can be scalar or array.
        delta_nu : float or NDArray
            Viscous penetration depth (m). Must match shape of omega.
        hydraulic_radius : float
            Hydraulic radius of the pore (m).

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex viscous function f_nu. Same shape as input.

        Notes
        -----
        The viscous penetration depth is defined as:
            delta_nu = sqrt(2 * nu / omega)
        where nu = mu/rho is the kinematic viscosity.
        """
        pass

    @abstractmethod
    def f_kappa(
        self,
        omega: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> ComplexOrArray:
        """
        Calculate the thermal thermoviscous function f_kappa.

        The thermal function characterizes how thermal conduction modifies
        the continuity equation in narrow pores.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Can be scalar or array.
        delta_kappa : float or NDArray
            Thermal penetration depth (m). Must match shape of omega.
        hydraulic_radius : float
            Hydraulic radius of the pore (m).

        Returns
        -------
        complex or NDArray[np.complexfloating]
            Complex thermal function f_kappa. Same shape as input.

        Notes
        -----
        The thermal penetration depth is defined as:
            delta_kappa = sqrt(2 * alpha / omega)
        where alpha = kappa/(rho*cp) is the thermal diffusivity.
        """
        pass

    def compute_both(
        self,
        omega: FloatOrArray,
        delta_nu: FloatOrArray,
        delta_kappa: FloatOrArray,
        hydraulic_radius: float,
    ) -> tuple[ComplexOrArray, ComplexOrArray]:
        """
        Compute both viscous and thermal functions simultaneously.

        This is a convenience method that calls both f_nu and f_kappa.
        Subclasses may override this for efficiency if both functions
        share intermediate calculations.

        Parameters
        ----------
        omega : float or NDArray
            Angular frequency (rad/s). Can be scalar or array.
        delta_nu : float or NDArray
            Viscous penetration depth (m). Must match shape of omega.
        delta_kappa : float or NDArray
            Thermal penetration depth (m). Must match shape of omega.
        hydraulic_radius : float
            Hydraulic radius of the pore (m).

        Returns
        -------
        tuple[ComplexOrArray, ComplexOrArray]
            Tuple of (f_nu, f_kappa), the viscous and thermal functions.

        Examples
        --------
        >>> from openthermoacoustics.geometry import CircularPore
        >>> geom = CircularPore()
        >>> f_nu, f_kappa = geom.compute_both(
        ...     omega=1000.0,
        ...     delta_nu=1e-4,
        ...     delta_kappa=1.2e-4,
        ...     hydraulic_radius=5e-4
        ... )
        """
        f_nu = self.f_nu(omega, delta_nu, hydraulic_radius)
        f_kappa = self.f_kappa(omega, delta_kappa, hydraulic_radius)
        return f_nu, f_kappa

    def __repr__(self) -> str:
        """Return string representation of the geometry."""
        return f"{self.__class__.__name__}()"
