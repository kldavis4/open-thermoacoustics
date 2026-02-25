"""Abstract base class for gas properties in thermoacoustic calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Gas(ABC):
    """
    Abstract base class for gas thermophysical properties.

    This class defines the interface for gas property calculations used in
    thermoacoustic simulations. All derived classes must implement methods
    for calculating density, sound speed, viscosity, thermal conductivity,
    specific heat, ratio of specific heats, and Prandtl number.

    All methods use SI units:
    - Temperature: K (Kelvin)
    - Pressure: Pa (Pascal)
    - Density: kg/m³
    - Sound speed: m/s
    - Viscosity: Pa·s
    - Thermal conductivity: W/(m·K)
    - Specific heat: J/(kg·K)

    Parameters
    ----------
    mean_pressure : float
        Mean operating pressure in Pa.

    Attributes
    ----------
    mean_pressure : float
        Mean operating pressure in Pa.
    """

    def __init__(self, mean_pressure: float) -> None:
        """
        Initialize the gas with a mean operating pressure.

        Parameters
        ----------
        mean_pressure : float
            Mean operating pressure in Pa.
        """
        self._mean_pressure = mean_pressure

    @property
    def mean_pressure(self) -> float:
        """
        Mean operating pressure in Pa.

        Returns
        -------
        float
            Mean operating pressure in Pa.
        """
        return self._mean_pressure

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the gas.

        Returns
        -------
        str
            Human-readable name of the gas (e.g., 'Helium', 'Air').
        """
        ...

    @property
    @abstractmethod
    def molar_mass(self) -> float:
        """
        Molar mass of the gas.

        Returns
        -------
        float
            Molar mass in kg/mol.
        """
        ...

    @abstractmethod
    def density(self, T: float, P: float | None = None) -> float:
        """
        Calculate gas density at given temperature and pressure.

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
        ...

    @abstractmethod
    def sound_speed(self, T: float, P: float | None = None) -> float:
        """
        Calculate sound speed at given temperature and pressure.

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Sound speed in m/s.
        """
        ...

    @abstractmethod
    def viscosity(self, T: float) -> float:
        """
        Calculate dynamic viscosity at given temperature.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Dynamic viscosity in Pa·s.
        """
        ...

    @abstractmethod
    def thermal_conductivity(self, T: float) -> float:
        """
        Calculate thermal conductivity at given temperature.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Thermal conductivity in W/(m·K).
        """
        ...

    @abstractmethod
    def specific_heat_cp(self, T: float, P: float | None = None) -> float:
        """
        Calculate specific heat at constant pressure.

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Specific heat at constant pressure in J/(kg·K).
        """
        ...

    @abstractmethod
    def gamma(self, T: float, P: float | None = None) -> float:
        """
        Calculate ratio of specific heats (cp/cv).

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Ratio of specific heats (dimensionless).
        """
        ...

    @abstractmethod
    def prandtl(self, T: float) -> float:
        """
        Calculate Prandtl number at given temperature.

        The Prandtl number is defined as Pr = μ·cp / κ, where μ is the
        dynamic viscosity, cp is the specific heat at constant pressure,
        and κ is the thermal conductivity.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Prandtl number (dimensionless).
        """
        ...

    def specific_gas_constant(self) -> float:
        """
        Calculate the specific gas constant R_specific = R_universal / M.

        Returns
        -------
        float
            Specific gas constant in J/(kg·K).
        """
        from openthermoacoustics.utils import R_UNIVERSAL

        return R_UNIVERSAL / self.molar_mass
