"""Xenon gas properties for thermoacoustic calculations."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.gas.base import Gas
from openthermoacoustics.utils import GAMMA, MOLAR_MASS, R_UNIVERSAL


class Xenon(Gas):
    """
    Xenon gas properties.

    Xenon is a monatomic ideal gas with gamma = 5/3. This class provides
    thermophysical property calculations using ideal gas relations and
    empirical correlations for transport properties.

    Parameters
    ----------
    mean_pressure : float
        Mean operating pressure in Pa.

    Notes
    -----
    Transport property correlations are power-law fits valid for
    temperatures in the range of approximately 100-1000 K.

    Viscosity correlation:
        mu(T) = 2.30e-5 * (T/300)^0.85 Pa*s

    Thermal conductivity correlation:
        k(T) = 0.00565 * (T/300)^0.85 W/(m*K)

    Examples
    --------
    >>> xe = Xenon(mean_pressure=101325.0)
    >>> xe.density(T=300.0)  # Density at 300 K and 1 atm
    5.29...
    >>> xe.sound_speed(T=300.0)  # Sound speed at 300 K
    178.0...
    """

    # Reference values for transport property correlations
    _T_REF: float = 300.0  # Reference temperature (K)
    _MU_REF: float = 2.30e-5  # Reference viscosity at T_ref (Pa*s)
    _KAPPA_REF: float = 0.00565  # Reference thermal conductivity at T_ref (W/(m*K))
    _VISC_EXPONENT: float = 0.85  # Power law exponent for viscosity
    _KAPPA_EXPONENT: float = 0.85  # Power law exponent for thermal conductivity

    @property
    def name(self) -> str:
        """
        Name of the gas.

        Returns
        -------
        str
            'Xenon'
        """
        return "Xenon"

    @property
    def molar_mass(self) -> float:
        """
        Molar mass of xenon.

        Returns
        -------
        float
            Molar mass in kg/mol (131.293e-3).
        """
        return MOLAR_MASS["xenon"]

    def density(self, T: float, P: float | None = None) -> float:
        """
        Calculate xenon density using ideal gas law.

        rho = P / (R_specific * T)

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Density in kg/m^3.
        """
        if P is None:
            P = self.mean_pressure
        R_specific = R_UNIVERSAL / self.molar_mass
        return P / (R_specific * T)

    def sound_speed(self, T: float, P: float | None = None) -> float:
        """
        Calculate sound speed in xenon.

        a = sqrt(gamma * R_specific * T)

        For an ideal gas, the sound speed depends only on temperature
        and the gas properties, not on pressure.

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. Not used for ideal gas (included for interface
            consistency).

        Returns
        -------
        float
            Sound speed in m/s.
        """
        R_specific = R_UNIVERSAL / self.molar_mass
        gamma = GAMMA["xenon"]
        return np.sqrt(gamma * R_specific * T)

    def viscosity(self, T: float) -> float:
        """
        Calculate dynamic viscosity using power law correlation.

        mu(T) = mu_ref * (T/T_ref)^0.85

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Dynamic viscosity in Pa*s.
        """
        return self._MU_REF * (T / self._T_REF) ** self._VISC_EXPONENT

    def thermal_conductivity(self, T: float) -> float:
        """
        Calculate thermal conductivity using power law correlation.

        k(T) = k_ref * (T/T_ref)^0.85

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Thermal conductivity in W/(m*K).
        """
        return self._KAPPA_REF * (T / self._T_REF) ** self._KAPPA_EXPONENT

    def specific_heat_cp(self, T: float, P: float | None = None) -> float:
        """
        Calculate specific heat at constant pressure.

        For a monatomic ideal gas: cp = (gamma/(gamma-1)) * R_specific = (5/2) * R_specific

        Parameters
        ----------
        T : float
            Temperature in K. Not used for ideal gas (included for interface
            consistency).
        P : float, optional
            Pressure in Pa. Not used for ideal gas.

        Returns
        -------
        float
            Specific heat at constant pressure in J/(kg*K).
        """
        R_specific = R_UNIVERSAL / self.molar_mass
        gamma = GAMMA["xenon"]
        return gamma / (gamma - 1) * R_specific

    def gamma(self, T: float, P: float | None = None) -> float:
        """
        Return ratio of specific heats for xenon.

        For a monatomic ideal gas, gamma = 5/3 ~ 1.667

        Parameters
        ----------
        T : float
            Temperature in K. Not used (included for interface consistency).
        P : float, optional
            Pressure in Pa. Not used.

        Returns
        -------
        float
            Ratio of specific heats (5/3 for monatomic gas).
        """
        return GAMMA["xenon"]

    def prandtl(self, T: float) -> float:
        """
        Calculate Prandtl number.

        Pr = mu * cp / k

        For xenon, the Prandtl number is nearly constant at approximately 0.67
        across a wide temperature range.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Prandtl number (dimensionless), approximately 0.67.
        """
        mu = self.viscosity(T)
        cp = self.specific_heat_cp(T)
        kappa = self.thermal_conductivity(T)
        return mu * cp / kappa
