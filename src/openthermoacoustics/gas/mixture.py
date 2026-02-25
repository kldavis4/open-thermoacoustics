"""Gas mixture support for thermoacoustic calculations.

This module provides classes and functions for calculating thermophysical
properties of gas mixtures using established mixing rules.

References
----------
Poling, B. E., Prausnitz, J. M., & O'Connell, J. P. (2001).
    The Properties of Gases and Liquids (5th ed.). McGraw-Hill.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from openthermoacoustics.gas.base import Gas
from openthermoacoustics.utils import R_UNIVERSAL


class GasMixture(Gas):
    """
    Gas mixture with support for binary and multi-component mixtures.

    This class calculates mixture properties using established mixing rules:
    - Molar mass: Mole-fraction weighted average
    - Density: Ideal gas law with mixture molar mass
    - Sound speed: Ideal gas formula with mixture gamma and R
    - Specific heat cp: Mass-fraction weighted average
    - Viscosity: Wilke's mixing rule
    - Thermal conductivity: Wassiljewa equation with Mason-Saxena modification

    Parameters
    ----------
    components : Sequence[Gas]
        List of Gas instances representing the mixture components.
    mole_fractions : Sequence[float]
        List of mole fractions corresponding to each component.
        Must sum to 1.0 (within tolerance).
    mean_pressure : float, optional
        Mean operating pressure in Pa. If not provided, uses the mean pressure
        of the first component.

    Attributes
    ----------
    components : tuple[Gas, ...]
        Tuple of Gas instances in the mixture.
    mole_fractions : tuple[float, ...]
        Tuple of mole fractions for each component.

    Examples
    --------
    >>> from openthermoacoustics.gas import Helium, Argon
    >>> # 70% Helium, 30% Argon mixture at 1 atm
    >>> he = Helium(mean_pressure=101325.0)
    >>> ar = Argon(mean_pressure=101325.0)
    >>> mix = GasMixture([he, ar], [0.7, 0.3])
    >>> mix.molar_mass  # Mixture molar mass
    0.01479...
    >>> mix.sound_speed(300.0)  # Sound speed at 300 K
    748.3...

    Notes
    -----
    The mixture rules implemented here assume ideal gas behavior. For real gas
    effects at high pressures, more sophisticated equations of state would be
    needed.

    He-Ar mixtures are common in thermoacoustics because:
    - Helium provides high sound speed and thermal conductivity
    - Argon provides higher density and acoustic impedance
    - Mixing allows tuning of acoustic properties and Prandtl number
    """

    _MOLE_FRACTION_TOLERANCE: float = 1e-6

    def __init__(
        self,
        components: Sequence[Gas],
        mole_fractions: Sequence[float],
        mean_pressure: float | None = None,
    ) -> None:
        """
        Initialize a gas mixture.

        Parameters
        ----------
        components : Sequence[Gas]
            List of Gas instances representing the mixture components.
        mole_fractions : Sequence[float]
            List of mole fractions corresponding to each component.
            Must sum to 1.0 (within tolerance).
        mean_pressure : float, optional
            Mean operating pressure in Pa. If not provided, uses the mean
            pressure of the first component.

        Raises
        ------
        ValueError
            If components and mole_fractions have different lengths.
        ValueError
            If mole fractions don't sum to 1.0 (within tolerance).
        ValueError
            If any mole fraction is negative.
        ValueError
            If components list is empty.
        """
        if len(components) == 0:
            raise ValueError("At least one component is required.")

        if len(components) != len(mole_fractions):
            raise ValueError(
                f"Number of components ({len(components)}) must match "
                f"number of mole fractions ({len(mole_fractions)})."
            )

        mole_fractions_array = np.array(mole_fractions, dtype=np.float64)

        if np.any(mole_fractions_array < 0):
            raise ValueError("All mole fractions must be non-negative.")

        fraction_sum = np.sum(mole_fractions_array)
        if not np.isclose(fraction_sum, 1.0, atol=self._MOLE_FRACTION_TOLERANCE):
            raise ValueError(
                f"Mole fractions must sum to 1.0, got {fraction_sum:.6f}."
            )

        # Normalize mole fractions to exactly 1.0
        mole_fractions_array = mole_fractions_array / fraction_sum

        self._components = tuple(components)
        self._mole_fractions = tuple(mole_fractions_array.tolist())

        # Use provided mean_pressure or get from first component
        if mean_pressure is None:
            mean_pressure = components[0].mean_pressure

        super().__init__(mean_pressure)

    @property
    def components(self) -> tuple[Gas, ...]:
        """
        Gas components in the mixture.

        Returns
        -------
        tuple[Gas, ...]
            Tuple of Gas instances.
        """
        return self._components

    @property
    def mole_fractions(self) -> tuple[float, ...]:
        """
        Mole fractions of each component.

        Returns
        -------
        tuple[float, ...]
            Tuple of mole fractions, summing to 1.0.
        """
        return self._mole_fractions

    @property
    def name(self) -> str:
        """
        Name of the gas mixture.

        Returns
        -------
        str
            Descriptive name showing components and fractions.
        """
        parts = []
        for gas, x in zip(self._components, self._mole_fractions):
            parts.append(f"{x*100:.1f}% {gas.name}")
        return " + ".join(parts)

    @property
    def molar_mass(self) -> float:
        """
        Mixture molar mass using mole-fraction weighted average.

        M_mix = sum(x_i * M_i)

        Returns
        -------
        float
            Mixture molar mass in kg/mol.
        """
        M_mix = sum(
            x * gas.molar_mass
            for x, gas in zip(self._mole_fractions, self._components)
        )
        return M_mix

    def _mass_fractions(self) -> tuple[float, ...]:
        """
        Calculate mass fractions from mole fractions.

        w_i = x_i * M_i / M_mix

        Returns
        -------
        tuple[float, ...]
            Mass fractions for each component.
        """
        M_mix = self.molar_mass
        mass_fracs = tuple(
            x * gas.molar_mass / M_mix
            for x, gas in zip(self._mole_fractions, self._components)
        )
        return mass_fracs

    def density(self, T: float, P: float | None = None) -> float:
        """
        Calculate mixture density using ideal gas law.

        rho = P / (R_specific * T) where R_specific = R_universal / M_mix

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
        R_specific = self.specific_gas_constant()
        return P / (R_specific * T)

    def sound_speed(self, T: float, P: float | None = None) -> float:
        """
        Calculate mixture sound speed.

        a = sqrt(gamma_mix * R_mix * T)

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
        R_specific = self.specific_gas_constant()
        gamma_mix = self.gamma(T, P)
        return np.sqrt(gamma_mix * R_specific * T)

    def specific_heat_cp(self, T: float, P: float | None = None) -> float:
        """
        Calculate mixture specific heat at constant pressure.

        Uses mass-fraction weighted average:
        cp_mix = sum(w_i * cp_i)

        where w_i is the mass fraction of component i.

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Specific heat at constant pressure in J/(kg*K).
        """
        mass_fracs = self._mass_fractions()
        cp_mix = sum(
            w * gas.specific_heat_cp(T, P)
            for w, gas in zip(mass_fracs, self._components)
        )
        return cp_mix

    def _specific_heat_cv(self, T: float, P: float | None = None) -> float:
        """
        Calculate mixture specific heat at constant volume.

        Uses mass-fraction weighted average:
        cv_mix = sum(w_i * cv_i)

        where cv_i = cp_i / gamma_i

        Parameters
        ----------
        T : float
            Temperature in K.
        P : float, optional
            Pressure in Pa. If None, uses mean_pressure.

        Returns
        -------
        float
            Specific heat at constant volume in J/(kg*K).
        """
        mass_fracs = self._mass_fractions()
        cv_mix = sum(
            w * gas.specific_heat_cp(T, P) / gas.gamma(T, P)
            for w, gas in zip(mass_fracs, self._components)
        )
        return cv_mix

    def gamma(self, T: float, P: float | None = None) -> float:
        """
        Calculate mixture ratio of specific heats.

        gamma_mix = cp_mix / cv_mix

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
        cp_mix = self.specific_heat_cp(T, P)
        cv_mix = self._specific_heat_cv(T, P)
        return cp_mix / cv_mix

    def _wilke_phi(
        self, mu_i: float, mu_j: float, M_i: float, M_j: float
    ) -> float:
        """
        Calculate Wilke's interaction parameter phi_ij.

        phi_ij = (1/sqrt(8)) * (1 + M_i/M_j)^(-0.5) *
                 (1 + (mu_i/mu_j)^0.5 * (M_j/M_i)^0.25)^2

        Parameters
        ----------
        mu_i : float
            Viscosity of component i in Pa*s.
        mu_j : float
            Viscosity of component j in Pa*s.
        M_i : float
            Molar mass of component i in kg/mol.
        M_j : float
            Molar mass of component j in kg/mol.

        Returns
        -------
        float
            Wilke's interaction parameter phi_ij (dimensionless).

        References
        ----------
        Wilke, C. R. (1950). A viscosity equation for gas mixtures.
            J. Chem. Phys., 18(4), 517-519.
        """
        mu_ratio = np.sqrt(mu_i / mu_j)
        M_ratio_quarter = (M_j / M_i) ** 0.25

        numerator = (1 + mu_ratio * M_ratio_quarter) ** 2
        denominator = np.sqrt(8 * (1 + M_i / M_j))

        return numerator / denominator

    def viscosity(self, T: float) -> float:
        """
        Calculate mixture dynamic viscosity using Wilke's mixing rule.

        mu_mix = sum(x_i * mu_i / sum(x_j * phi_ij))

        where phi_ij is Wilke's interaction parameter:
        phi_ij = (1/sqrt(8)) * (1 + M_i/M_j)^(-0.5) *
                 (1 + (mu_i/mu_j)^0.5 * (M_j/M_i)^0.25)^2

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Dynamic viscosity in Pa*s.

        References
        ----------
        Wilke, C. R. (1950). A viscosity equation for gas mixtures.
            J. Chem. Phys., 18(4), 517-519.
        """
        n = len(self._components)
        mu = [gas.viscosity(T) for gas in self._components]
        M = [gas.molar_mass for gas in self._components]
        x = self._mole_fractions

        mu_mix = 0.0
        for i in range(n):
            phi_sum = 0.0
            for j in range(n):
                phi_ij = self._wilke_phi(mu[i], mu[j], M[i], M[j])
                phi_sum += x[j] * phi_ij
            mu_mix += x[i] * mu[i] / phi_sum

        return mu_mix

    def _mason_saxena_A(
        self, k_i: float, k_j: float, M_i: float, M_j: float
    ) -> float:
        """
        Calculate Mason-Saxena interaction parameter A_ij.

        Uses a similar form to Wilke's phi_ij:
        A_ij = (1/sqrt(8)) * (1 + M_i/M_j)^(-0.5) *
               (1 + (k_i/k_j)^0.5 * (M_j/M_i)^0.25)^2

        Parameters
        ----------
        k_i : float
            Thermal conductivity of component i in W/(m*K).
        k_j : float
            Thermal conductivity of component j in W/(m*K).
        M_i : float
            Molar mass of component i in kg/mol.
        M_j : float
            Molar mass of component j in kg/mol.

        Returns
        -------
        float
            Mason-Saxena interaction parameter A_ij (dimensionless).

        References
        ----------
        Mason, E. A., & Saxena, S. C. (1958). Approximate formula for the
            thermal conductivity of gas mixtures. Physics of Fluids, 1(5), 361.
        """
        k_ratio = np.sqrt(k_i / k_j)
        M_ratio_quarter = (M_j / M_i) ** 0.25

        numerator = (1 + k_ratio * M_ratio_quarter) ** 2
        denominator = np.sqrt(8 * (1 + M_i / M_j))

        return numerator / denominator

    def thermal_conductivity(self, T: float) -> float:
        """
        Calculate mixture thermal conductivity.

        Uses Wassiljewa equation with Mason-Saxena modification:
        k_mix = sum(x_i * k_i / sum(x_j * A_ij))

        where A_ij is the Mason-Saxena interaction parameter, similar in form
        to Wilke's phi_ij for viscosity.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Thermal conductivity in W/(m*K).

        References
        ----------
        Mason, E. A., & Saxena, S. C. (1958). Approximate formula for the
            thermal conductivity of gas mixtures. Physics of Fluids, 1(5), 361.
        """
        n = len(self._components)
        k = [gas.thermal_conductivity(T) for gas in self._components]
        M = [gas.molar_mass for gas in self._components]
        x = self._mole_fractions

        k_mix = 0.0
        for i in range(n):
            A_sum = 0.0
            for j in range(n):
                A_ij = self._mason_saxena_A(k[i], k[j], M[i], M[j])
                A_sum += x[j] * A_ij
            k_mix += x[i] * k[i] / A_sum

        return k_mix

    def prandtl(self, T: float) -> float:
        """
        Calculate mixture Prandtl number.

        Pr = mu_mix * cp_mix / k_mix

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        float
            Prandtl number (dimensionless).
        """
        mu = self.viscosity(T)
        cp = self.specific_heat_cp(T)
        k = self.thermal_conductivity(T)
        return mu * cp / k


def helium_argon(he_fraction: float, mean_pressure: float) -> GasMixture:
    """
    Create a helium-argon gas mixture.

    He-Ar mixtures are commonly used in thermoacoustic devices because:
    - Helium provides high sound speed, low density, and high thermal conductivity
    - Argon provides higher density and acoustic impedance
    - Mixing allows tuning of the acoustic impedance and Prandtl number

    Parameters
    ----------
    he_fraction : float
        Mole fraction of helium (0 to 1). The argon fraction is (1 - he_fraction).
    mean_pressure : float
        Mean operating pressure in Pa.

    Returns
    -------
    GasMixture
        A GasMixture instance with helium and argon.

    Raises
    ------
    ValueError
        If he_fraction is not in the range [0, 1].

    Examples
    --------
    >>> # 70% helium, 30% argon at 1 atm
    >>> mix = helium_argon(0.7, 101325.0)
    >>> mix.sound_speed(300.0)
    748.3...
    """
    if not 0.0 <= he_fraction <= 1.0:
        raise ValueError(
            f"Helium fraction must be between 0 and 1, got {he_fraction}."
        )

    from openthermoacoustics.gas.argon import Argon
    from openthermoacoustics.gas.helium import Helium

    he = Helium(mean_pressure)
    ar = Argon(mean_pressure)

    ar_fraction = 1.0 - he_fraction

    return GasMixture([he, ar], [he_fraction, ar_fraction], mean_pressure)


def helium_xenon(he_fraction: float, mean_pressure: float) -> GasMixture:
    """
    Create a helium-xenon gas mixture.

    He-Xe mixtures offer a wide range of tunable acoustic properties due to
    the large difference in molar masses between helium and xenon.

    Parameters
    ----------
    he_fraction : float
        Mole fraction of helium (0 to 1). The xenon fraction is (1 - he_fraction).
    mean_pressure : float
        Mean operating pressure in Pa.

    Returns
    -------
    GasMixture
        A GasMixture instance with helium and xenon.

    Raises
    ------
    ValueError
        If he_fraction is not in the range [0, 1].

    Examples
    --------
    >>> # 80% helium, 20% xenon at 1 atm
    >>> mix = helium_xenon(0.8, 101325.0)
    >>> mix.molar_mass  # Much heavier than pure He
    0.02946...
    """
    if not 0.0 <= he_fraction <= 1.0:
        raise ValueError(
            f"Helium fraction must be between 0 and 1, got {he_fraction}."
        )

    from openthermoacoustics.gas.helium import Helium
    from openthermoacoustics.gas.xenon import Xenon

    he = Helium(mean_pressure)
    xe = Xenon(mean_pressure)

    xe_fraction = 1.0 - he_fraction

    return GasMixture([he, xe], [he_fraction, xe_fraction], mean_pressure)
