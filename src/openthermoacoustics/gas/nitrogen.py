"""Nitrogen gas properties for thermoacoustic calculations."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.gas.base import Gas
from openthermoacoustics.utils import GAMMA, MOLAR_MASS, R_UNIVERSAL


class Nitrogen(Gas):
    """
    Nitrogen gas properties.

    Nitrogen is a diatomic ideal gas with γ = 1.4. This class provides
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
        μ(T) = 1.79e-5 * (T/300)^0.7 Pa·s

    Thermal conductivity correlation:
        κ(T) = 0.026 * (T/300)^0.7 W/(m·K)

    Examples
    --------
    >>> n2 = Nitrogen(mean_pressure=101325.0)
    >>> n2.density(T=300.0)  # Density at 300 K and 1 atm
    1.138...
    >>> n2.sound_speed(T=300.0)  # Sound speed at 300 K
    352.0...
    """

    # Reference values for transport property correlations
    _T_REF: float = 300.0  # Reference temperature (K)
    _MU_REF: float = 1.79e-5  # Reference viscosity at T_ref (Pa·s)
    _KAPPA_REF: float = 0.026  # Reference thermal conductivity at T_ref (W/(m·K))
    _VISC_EXPONENT: float = 0.7  # Power law exponent for viscosity
    _KAPPA_EXPONENT: float = 0.7  # Power law exponent for thermal conductivity
    _PRANDTL_APPROX: float = 0.71  # Approximate Prandtl number

    @property
    def name(self) -> str:
        """
        Name of the gas.

        Returns
        -------
        str
            'Nitrogen'
        """
        return "Nitrogen"

    @property
    def molar_mass(self) -> float:
        """
        Molar mass of nitrogen.

        Returns
        -------
        float
            Molar mass in kg/mol (28.0134e-3).
        """
        return MOLAR_MASS["nitrogen"]

    def density(self, T: float, P: float | None = None) -> float:
        """
        Calculate nitrogen density using ideal gas law.

        ρ = P / (R_specific * T)

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Density in kg/m³.
        """
        if P is None:
            P = self.mean_pressure
        R_specific = R_UNIVERSAL / self.molar_mass
        return P / (R_specific * T)

    def sound_speed(self, T: float, P: float | None = None) -> float:
        """
        Calculate sound speed in nitrogen.

        a = sqrt(γ * R_specific * T)

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
        gamma = GAMMA["nitrogen"]
        return np.sqrt(gamma * R_specific * T)

    def viscosity(self, T: float) -> float:
        """
        Calculate dynamic viscosity using power law correlation.

        μ(T) = μ_ref * (T/T_ref)^0.7

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Dynamic viscosity in Pa·s.
        """
        return self._MU_REF * (T / self._T_REF) ** self._VISC_EXPONENT

    def thermal_conductivity(self, T: float) -> float:
        """
        Calculate thermal conductivity using power law correlation.

        κ(T) = κ_ref * (T/T_ref)^0.7

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Thermal conductivity in W/(m·K).
        """
        return self._KAPPA_REF * (T / self._T_REF) ** self._KAPPA_EXPONENT

    def specific_heat_cp(self, T: float, P: float | None = None) -> float:
        """
        Calculate specific heat at constant pressure.

        For a diatomic ideal gas: cp = (γ/(γ-1)) * R_specific = (7/2) * R_specific

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
            Specific heat at constant pressure in J/(kg·K).
        """
        R_specific = R_UNIVERSAL / self.molar_mass
        gamma = GAMMA["nitrogen"]
        return gamma / (gamma - 1) * R_specific

    def gamma(self, T: float, P: float | None = None) -> float:
        """
        Return ratio of specific heats for nitrogen.

        For a diatomic ideal gas at moderate temperatures, γ = 1.4

        Parameters
        ----------
        T : float
            Temperature in K. Not used (included for interface consistency).
        P : float, optional
            Pressure in Pa. Not used.

        Returns
        -------
        float
            Ratio of specific heats (1.4 for diatomic gas).
        """
        return GAMMA["nitrogen"]

    def prandtl(self, T: float) -> float:
        """
        Calculate Prandtl number.

        Pr = μ * cp / κ

        For nitrogen, the Prandtl number is approximately 0.71 across a wide
        temperature range.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Prandtl number (dimensionless), approximately 0.71.
        """
        mu = self.viscosity(T)
        cp = self.specific_heat_cp(T)
        kappa = self.thermal_conductivity(T)
        return mu * cp / kappa
