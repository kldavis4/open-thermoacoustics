"""Air gas properties for thermoacoustic calculations."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.gas.base import Gas
from openthermoacoustics.utils import GAMMA, MOLAR_MASS, R_UNIVERSAL


class Air(Gas):
    """
    Air gas properties.

    Air is treated as a diatomic ideal gas with γ = 1.4. This class provides
    thermophysical property calculations using ideal gas relations and
    empirical correlations for transport properties.

    Parameters
    ----------
    mean_pressure : float
        Mean operating pressure in Pa.

    Notes
    -----
    Viscosity uses Sutherland's law, which provides accurate values over a
    wide temperature range (approximately 170-1900 K):

        μ = μ_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

    where:
        μ_ref = 1.716e-5 Pa·s (reference viscosity)
        T_ref = 273.15 K (reference temperature)
        S = 110.4 K (Sutherland constant)

    Thermal conductivity correlation:
        κ(T) = 0.0262 * (T/300)^0.85 W/(m·K)

    Examples
    --------
    >>> air = Air(mean_pressure=101325.0)
    >>> air.density(T=300.0)  # Density at 300 K and 1 atm
    1.176...
    >>> air.sound_speed(T=300.0)  # Sound speed at 300 K
    347.2...
    >>> air.viscosity(T=300.0)  # Viscosity at 300 K
    1.846e-5...
    """

    # Sutherland's law constants for viscosity
    _MU_REF: float = 1.716e-5  # Reference viscosity (Pa·s) at T_ref
    _T_REF_SUTHERLAND: float = 273.15  # Reference temperature for Sutherland's law (K)
    _SUTHERLAND_S: float = 110.4  # Sutherland constant (K)

    # Thermal conductivity correlation constants
    _T_REF_KAPPA: float = 300.0  # Reference temperature for conductivity (K)
    _KAPPA_REF: float = 0.0262  # Reference thermal conductivity at 300 K (W/(m·K))
    _KAPPA_EXPONENT: float = 0.85  # Power law exponent for thermal conductivity

    # Approximate Prandtl number
    _PRANDTL_APPROX: float = 0.71

    @property
    def name(self) -> str:
        """
        Name of the gas.

        Returns
        -------
        str
            'Air'
        """
        return "Air"

    @property
    def molar_mass(self) -> float:
        """
        Effective molar mass of dry air.

        Returns
        -------
        float
            Molar mass in kg/mol (28.9647e-3).
        """
        return MOLAR_MASS["air"]

    def density(self, T: float, P: float | None = None) -> float:
        """
        Calculate air density using ideal gas law.

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
        Calculate sound speed in air.

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
        gamma = GAMMA["air"]
        return np.sqrt(gamma * R_specific * T)

    def viscosity(self, T: float) -> float:
        """
        Calculate dynamic viscosity using Sutherland's law.

        μ = μ_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

        Sutherland's law provides accurate viscosity values for air over
        a wide temperature range (approximately 170-1900 K).

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Dynamic viscosity in Pa·s.
        """
        T_ref = self._T_REF_SUTHERLAND
        S = self._SUTHERLAND_S
        return self._MU_REF * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

    def thermal_conductivity(self, T: float) -> float:
        """
        Calculate thermal conductivity using power law correlation.

        κ(T) = κ_ref * (T/T_ref)^0.85

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Thermal conductivity in W/(m·K).
        """
        return self._KAPPA_REF * (T / self._T_REF_KAPPA) ** self._KAPPA_EXPONENT

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
        gamma = GAMMA["air"]
        return gamma / (gamma - 1) * R_specific

    def gamma(self, T: float, P: float | None = None) -> float:
        """
        Return ratio of specific heats for air.

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
        return GAMMA["air"]

    def prandtl(self, T: float) -> float:
        """
        Calculate Prandtl number.

        Pr = μ * cp / κ

        For air, the Prandtl number is approximately 0.71 across a wide
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
