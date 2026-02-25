"""Variable temperature heat exchanger segments (VXT1, VXT2, VXQ1, VXQ2).

This module implements the VXT1, VXT2, VXQ1, and VXQ2 segments from reference baseline,
which model heat exchangers where the gas temperature varies with x.

VXT1: Single-pass heat exchanger with fixed solid temperature
VXT2: Two-pass heat exchanger with potentially different solid temperatures
VXQ1: Single-pass heat exchanger with fixed heat flux per unit length
VXQ2: Two-pass heat exchanger with fixed heat flux per unit length

VXT segments are useful for long heat exchangers with vigorous shell-side cooling
(e.g., water-cooled heat exchangers) where the solid surfaces maintain a
nearly constant temperature.

VXQ segments are useful for heat exchangers where the heat input per unit length
is controlled externally (e.g., electrically heated or with uniform shell-side
heat transfer).

References
----------
[1] published literature, relevant reference, governing relations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    penetration_depth_viscous,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class VariableTemperatureHeatExchanger(Segment):
    """
    VXT1 segment - Variable gas temperature heat exchanger with fixed solid temperature.

    Models a heat exchanger where the solid temperature is fixed (user input)
    but the gas mean temperature varies with x. This is appropriate when the
    shell-side heat transfer coefficient is much higher than the tube-side
    (thermoacoustic) coefficient.

    The segment consists of three regions:
    - Tubesheet 1 (inlet): Viscous/thermal losses, no heat transfer
    - Heat transfer region: Gas temperature changes toward solid temperature
    - Tubesheet 2 (outlet): Viscous/thermal losses, no heat transfer

    Parameters
    ----------
    area : float
        Total cross-sectional area A = Agas + Asolid + Ashell-side (m²).
    gas_area_fraction : float
        Fraction of total area available to gas, Agas/A.
    solid_area_fraction : float
        Fraction of total area for solid thermal conduction, Asolid/A.
    hydraulic_radius : float
        Hydraulic radius of gas channels (m).
    length_tubesheet1 : float
        Length of inlet tubesheet (m). No heat transfer in this region.
    length_heat_transfer : float
        Length of heat transfer region (m).
    length_tubesheet2 : float
        Length of outlet tubesheet (m). No heat transfer in this region.
    solid_temperature : float
        Fixed solid temperature (K).
    solid_thermal_conductivity : float, optional
        Thermal conductivity of solid (W/(m·K)). Default 15.0 (stainless steel).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    In VXT1, the solid temperature Tsolid is fixed (independent of x).
    The gas temperature Tm(x) varies according to heat transfer.

    This uses a simplified model where:
    - Acoustic propagation uses boundary-layer approximation
    - Heat transfer uses convective model with Nu = 3.7 (laminar circular tubes)
    - Temperature evolves toward solid temperature exponentially
    """

    def __init__(
        self,
        area: float,
        gas_area_fraction: float,
        solid_area_fraction: float,
        hydraulic_radius: float,
        length_tubesheet1: float,
        length_heat_transfer: float,
        length_tubesheet2: float,
        solid_temperature: float,
        solid_thermal_conductivity: float = 15.0,
        name: str = "",
    ) -> None:
        if not 0 < gas_area_fraction < 1:
            raise ValueError(f"gas_area_fraction must be in (0, 1), got {gas_area_fraction}")
        if solid_area_fraction < 0 or solid_area_fraction >= 1:
            raise ValueError(f"solid_area_fraction must be in [0, 1), got {solid_area_fraction}")
        if solid_temperature <= 0:
            raise ValueError(f"solid_temperature must be positive, got {solid_temperature}")

        total_length = length_tubesheet1 + length_heat_transfer + length_tubesheet2

        self._gas_area_fraction = gas_area_fraction
        self._solid_area_fraction = solid_area_fraction
        self._hydraulic_radius = hydraulic_radius
        self._length_tubesheet1 = length_tubesheet1
        self._length_heat_transfer = length_heat_transfer
        self._length_tubesheet2 = length_tubesheet2
        self._solid_temperature = solid_temperature
        self._solid_thermal_conductivity = solid_thermal_conductivity

        super().__init__(name=name, length=total_length, area=area, geometry=None)

    @property
    def gas_area_fraction(self) -> float:
        """Fraction of area available to gas."""
        return self._gas_area_fraction

    @property
    def solid_area_fraction(self) -> float:
        """Fraction of area for solid conduction."""
        return self._solid_area_fraction

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of gas channels (m)."""
        return self._hydraulic_radius

    @property
    def solid_temperature(self) -> float:
        """Fixed solid temperature (K)."""
        return self._solid_temperature

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Parameters
        ----------
        x : float
            Axial position within the segment (m).
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1), T_m].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K) - not used, taken from state.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1 = y[0] + 1j * y[1]
        U1 = y[2] + 1j * y[3]
        T_local = y[4]

        # Ensure temperature stays physical
        T_local = max(T_local, 10.0)  # Minimum 10 K
        T_local = min(T_local, 10000.0)  # Maximum 10000 K

        A = self._area
        A_gas = self._gas_area_fraction * A
        rh = self._hydraulic_radius
        T_solid = self._solid_temperature

        # Gas properties at local temperature
        rho_m = gas.density(T_local)
        a = gas.sound_speed(T_local)
        gamma = gas.gamma(T_local)
        mu = gas.viscosity(T_local)
        k_gas = gas.thermal_conductivity(T_local)
        cp = gas.specific_heat_cp(T_local)

        # Penetration depths
        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        # Boundary layer f functions (approximate for large channels)
        f_nu = (1 - 1j) * delta_nu / rh
        f_kappa = (1 - 1j) * delta_kappa / rh

        # Determine region
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_heat_transfer

        in_heat_transfer_region = x1 <= x < x2

        # Momentum equation: dp1/dx = -iωρm/(1-fν)/Agas * U1
        denom_mom = (1 - f_nu) * A_gas
        if np.abs(denom_mom) > 1e-20:
            dp1_dx = -1j * omega * rho_m / denom_mom * U1
        else:
            dp1_dx = 0.0 + 0.0j

        # Continuity equation: dU1/dx = -iωAgas/(ρm*a²) * [1 + (γ-1)*fκ] * p1
        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

        # Temperature equation
        if in_heat_transfer_region:
            # Heat transfer from solid to gas
            # Using convective model: dTm/dx = h*P*(Tsolid - Tm) / (mdot * cp)
            # where h = k*Nu/(4*rh), P = perimeter, mdot = rho*u*A
            #
            # Simplified: dTm/dx ≈ k*Nu/(4*rh²) * (Tsolid - Tm) / (rho*|u1|*cp)

            u1_mag = np.abs(U1) / A_gas if A_gas > 1e-20 else 0.0

            if u1_mag > 1e-10:
                # Nusselt number for laminar flow in tubes
                Nu = 3.7

                # Heat transfer coefficient
                h = k_gas * Nu / (4.0 * rh)

                # Perimeter per unit area = 4/rh for circular tubes
                perimeter_per_area = 1.0 / rh

                # Temperature change rate
                # Using thermal time constant approach
                # dTm/dx = (Tsolid - Tm) / L_char
                # where L_char ~ rho*cp*u*rh / (k*Nu)

                L_char = rho_m * cp * u1_mag * rh / (k_gas * Nu) if k_gas * Nu > 1e-15 else 1e10

                # Limit L_char to reasonable values
                L_char = max(L_char, 0.001)  # At least 1 mm
                L_char = min(L_char, 100.0)  # At most 100 m

                dTm_dx = (T_solid - T_local) / L_char
            else:
                # No flow - temperature relaxes more slowly
                # Use a conduction-based estimate
                L_char = 0.1  # Default 10 cm thermal length scale
                dTm_dx = (T_solid - T_local) / L_char
        else:
            # In tubesheet regions: no heat transfer
            dTm_dx = 0.0

        # Limit temperature change rate
        max_dT_per_m = 1000.0  # Maximum 1000 K/m
        dTm_dx = max(min(dTm_dx, max_dT_per_m), -max_dT_per_m)

        return np.array([
            dp1_dx.real, dp1_dx.imag,
            dU1_dx.real, dU1_dx.imag,
            dTm_dx
        ])

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the VXT1 heat exchanger.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature at input (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        if self._length == 0:
            return p1_in, U1_in, T_m

        # Ensure input temperature is physical
        T_m = max(T_m, 10.0)
        T_m = min(T_m, 10000.0)

        # Initial state vector: [Re(p1), Im(p1), Re(U1), Im(U1), T_m]
        y0 = np.array([
            p1_in.real, p1_in.imag,
            U1_in.real, U1_in.imag,
            T_m
        ])

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, y[4])

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
            max_step=self._length / 10,  # Ensure reasonable stepping
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        y_out = sol.y[:, -1]
        p1_out = y_out[0] + 1j * y_out[1]
        U1_out = y_out[2] + 1j * y_out[3]
        T_m_out = y_out[4]

        # Ensure output temperature is physical
        T_m_out = max(T_m_out, 10.0)
        T_m_out = min(T_m_out, 10000.0)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return (
            f"VariableTemperatureHeatExchanger(name='{self._name}', "
            f"length={self._length}, T_solid={self._solid_temperature})"
        )


class VariableTemperatureHeatExchanger2Pass(Segment):
    """
    VXT2 segment - Two-pass variable gas temperature heat exchanger.

    Models a heat exchanger with two separate heat transfer regions, each
    with its own fixed solid temperature. This is useful for shell-and-tube
    heat exchangers with two passes on the shell side.

    The segment consists of five regions:
    - Tubesheet 1 (inlet): Viscous/thermal losses, no heat transfer
    - Heat transfer region 1: Gas temperature changes toward solid_temperature_1
    - Heat transfer region 2: Gas temperature changes toward solid_temperature_2
    - Tubesheet 2 (outlet): Viscous/thermal losses, no heat transfer

    Parameters
    ----------
    area : float
        Total cross-sectional area A = Agas + Asolid + Ashell-side (m²).
    gas_area_fraction : float
        Fraction of total area available to gas, Agas/A.
    solid_area_fraction : float
        Fraction of total area for solid thermal conduction, Asolid/A.
    hydraulic_radius : float
        Hydraulic radius of gas channels (m).
    length_tubesheet1 : float
        Length of inlet tubesheet (m). No heat transfer in this region.
    length_pass1 : float
        Length of first heat transfer region (m).
    length_pass2 : float
        Length of second heat transfer region (m).
    length_tubesheet2 : float
        Length of outlet tubesheet (m). No heat transfer in this region.
    solid_temperature_1 : float
        Fixed solid temperature in first pass (K).
    solid_temperature_2 : float
        Fixed solid temperature in second pass (K).
    solid_thermal_conductivity : float, optional
        Thermal conductivity of solid (W/(m·K)). Default 15.0 (stainless steel).
    name : str, optional
        Name identifier for the segment.
    """

    def __init__(
        self,
        area: float,
        gas_area_fraction: float,
        solid_area_fraction: float,
        hydraulic_radius: float,
        length_tubesheet1: float,
        length_pass1: float,
        length_pass2: float,
        length_tubesheet2: float,
        solid_temperature_1: float,
        solid_temperature_2: float,
        solid_thermal_conductivity: float = 15.0,
        name: str = "",
    ) -> None:
        if not 0 < gas_area_fraction < 1:
            raise ValueError(f"gas_area_fraction must be in (0, 1), got {gas_area_fraction}")
        if solid_area_fraction < 0 or solid_area_fraction >= 1:
            raise ValueError(f"solid_area_fraction must be in [0, 1), got {solid_area_fraction}")
        if solid_temperature_1 <= 0:
            raise ValueError(f"solid_temperature_1 must be positive, got {solid_temperature_1}")
        if solid_temperature_2 <= 0:
            raise ValueError(f"solid_temperature_2 must be positive, got {solid_temperature_2}")

        total_length = length_tubesheet1 + length_pass1 + length_pass2 + length_tubesheet2

        self._gas_area_fraction = gas_area_fraction
        self._solid_area_fraction = solid_area_fraction
        self._hydraulic_radius = hydraulic_radius
        self._length_tubesheet1 = length_tubesheet1
        self._length_pass1 = length_pass1
        self._length_pass2 = length_pass2
        self._length_tubesheet2 = length_tubesheet2
        self._solid_temperature_1 = solid_temperature_1
        self._solid_temperature_2 = solid_temperature_2
        self._solid_thermal_conductivity = solid_thermal_conductivity

        super().__init__(name=name, length=total_length, area=area, geometry=None)

    @property
    def gas_area_fraction(self) -> float:
        """Fraction of area available to gas."""
        return self._gas_area_fraction

    @property
    def solid_area_fraction(self) -> float:
        """Fraction of area for solid conduction."""
        return self._solid_area_fraction

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of gas channels (m)."""
        return self._hydraulic_radius

    @property
    def solid_temperature_1(self) -> float:
        """Fixed solid temperature in first pass (K)."""
        return self._solid_temperature_1

    @property
    def solid_temperature_2(self) -> float:
        """Fixed solid temperature in second pass (K)."""
        return self._solid_temperature_2

    def _get_solid_temperature(self, x: float) -> float:
        """Get the solid temperature at position x."""
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_pass1
        x3 = x2 + self._length_pass2

        if x < x1:
            # Tubesheet 1 - no heat transfer
            return self._solid_temperature_1
        elif x < x2:
            # Pass 1
            return self._solid_temperature_1
        elif x < x3:
            # Pass 2
            return self._solid_temperature_2
        else:
            # Tubesheet 2 - no heat transfer
            return self._solid_temperature_2

    def _in_heat_transfer_region(self, x: float) -> bool:
        """Check if position x is in a heat transfer region."""
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_pass1
        x3 = x2 + self._length_pass2

        return x1 <= x < x3

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Parameters
        ----------
        x : float
            Axial position within the segment (m).
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1), T_m].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K) - not used, taken from state.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1 = y[0] + 1j * y[1]
        U1 = y[2] + 1j * y[3]
        T_local = y[4]

        # Ensure temperature stays physical
        T_local = max(T_local, 10.0)
        T_local = min(T_local, 10000.0)

        A = self._area
        A_gas = self._gas_area_fraction * A
        rh = self._hydraulic_radius
        T_solid = self._get_solid_temperature(x)

        # Gas properties at local temperature
        rho_m = gas.density(T_local)
        a = gas.sound_speed(T_local)
        gamma = gas.gamma(T_local)
        mu = gas.viscosity(T_local)
        k_gas = gas.thermal_conductivity(T_local)
        cp = gas.specific_heat_cp(T_local)

        # Penetration depths
        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        # Boundary layer f functions
        f_nu = (1 - 1j) * delta_nu / rh
        f_kappa = (1 - 1j) * delta_kappa / rh

        in_heat_transfer = self._in_heat_transfer_region(x)

        # Momentum equation
        denom_mom = (1 - f_nu) * A_gas
        if np.abs(denom_mom) > 1e-20:
            dp1_dx = -1j * omega * rho_m / denom_mom * U1
        else:
            dp1_dx = 0.0 + 0.0j

        # Continuity equation
        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

        # Temperature equation
        if in_heat_transfer:
            u1_mag = np.abs(U1) / A_gas if A_gas > 1e-20 else 0.0

            if u1_mag > 1e-10:
                Nu = 3.7
                L_char = rho_m * cp * u1_mag * rh / (k_gas * Nu) if k_gas * Nu > 1e-15 else 1e10
                L_char = max(L_char, 0.001)
                L_char = min(L_char, 100.0)
                dTm_dx = (T_solid - T_local) / L_char
            else:
                L_char = 0.1
                dTm_dx = (T_solid - T_local) / L_char
        else:
            dTm_dx = 0.0

        # Limit temperature change rate
        max_dT_per_m = 1000.0
        dTm_dx = max(min(dTm_dx, max_dT_per_m), -max_dT_per_m)

        return np.array([
            dp1_dx.real, dp1_dx.imag,
            dU1_dx.real, dU1_dx.imag,
            dTm_dx
        ])

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the VXT2 heat exchanger.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature at input (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        if self._length == 0:
            return p1_in, U1_in, T_m

        T_m = max(T_m, 10.0)
        T_m = min(T_m, 10000.0)

        y0 = np.array([
            p1_in.real, p1_in.imag,
            U1_in.real, U1_in.imag,
            T_m
        ])

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, y[4])

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
            max_step=self._length / 20,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        y_out = sol.y[:, -1]
        p1_out = y_out[0] + 1j * y_out[1]
        U1_out = y_out[2] + 1j * y_out[3]
        T_m_out = y_out[4]

        T_m_out = max(T_m_out, 10.0)
        T_m_out = min(T_m_out, 10000.0)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return (
            f"VariableTemperatureHeatExchanger2Pass(name='{self._name}', "
            f"length={self._length}, T_solid_1={self._solid_temperature_1}, "
            f"T_solid_2={self._solid_temperature_2})"
        )


class VariableHeatFluxHeatExchanger(Segment):
    """
    VXQ1 segment - Variable gas temperature heat exchanger with fixed heat flux.

    Models a heat exchanger where the heat input per unit length is fixed (user input)
    but the gas and solid temperatures vary with x. This is appropriate for
    externally heated heat exchangers (e.g., electric heaters) or heat exchangers
    with uniform shell-side heat transfer.

    The segment consists of three regions:
    - Tubesheet 1 (inlet): Viscous/thermal losses, no heat transfer
    - Heat transfer region: Heat flux q̇ = HeatP / length, temperature changes
    - Tubesheet 2 (outlet): Viscous/thermal losses, no heat transfer

    Parameters
    ----------
    area : float
        Total cross-sectional area A = Agas + Asolid + Ashell-side (m²).
    gas_area_fraction : float
        Fraction of total area available to gas, Agas/A.
    solid_area_fraction : float
        Fraction of total area for solid thermal conduction, Asolid/A.
    hydraulic_radius : float
        Hydraulic radius of gas channels (m).
    length_tubesheet1 : float
        Length of inlet tubesheet (m). No heat transfer in this region.
    length_heat_transfer : float
        Length of heat transfer region (m).
    length_tubesheet2 : float
        Length of outlet tubesheet (m). No heat transfer in this region.
    heat_power : float
        Total heat input to the heat transfer region (W). Positive = heating.
    solid_thermal_conductivity : float, optional
        Thermal conductivity of solid (W/(m·K)). Default 15.0 (stainless steel).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    In VXQ1, the heat flux per unit length q̇ = HeatP / L_heat_transfer is fixed.
    The gas temperature Tm(x) evolves as: dTm/dx = q̇ / (ṁ * cp)
    The solid temperature Tsolid(x) is computed from heat transfer balance.

    For an ideal gas with constant cp and constant q̇, Tm varies linearly with x.
    """

    def __init__(
        self,
        area: float,
        gas_area_fraction: float,
        solid_area_fraction: float,
        hydraulic_radius: float,
        length_tubesheet1: float,
        length_heat_transfer: float,
        length_tubesheet2: float,
        heat_power: float,
        solid_thermal_conductivity: float = 15.0,
        name: str = "",
    ) -> None:
        if not 0 < gas_area_fraction < 1:
            raise ValueError(f"gas_area_fraction must be in (0, 1), got {gas_area_fraction}")
        if solid_area_fraction < 0 or solid_area_fraction >= 1:
            raise ValueError(f"solid_area_fraction must be in [0, 1), got {solid_area_fraction}")
        if length_heat_transfer <= 0:
            raise ValueError(f"length_heat_transfer must be positive, got {length_heat_transfer}")

        total_length = length_tubesheet1 + length_heat_transfer + length_tubesheet2

        self._gas_area_fraction = gas_area_fraction
        self._solid_area_fraction = solid_area_fraction
        self._hydraulic_radius = hydraulic_radius
        self._length_tubesheet1 = length_tubesheet1
        self._length_heat_transfer = length_heat_transfer
        self._length_tubesheet2 = length_tubesheet2
        self._heat_power = heat_power
        self._solid_thermal_conductivity = solid_thermal_conductivity

        # Heat flux per unit length (W/m)
        self._heat_flux_per_length = heat_power / length_heat_transfer

        super().__init__(name=name, length=total_length, area=area, geometry=None)

    @property
    def gas_area_fraction(self) -> float:
        """Fraction of area available to gas."""
        return self._gas_area_fraction

    @property
    def solid_area_fraction(self) -> float:
        """Fraction of area for solid conduction."""
        return self._solid_area_fraction

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of gas channels (m)."""
        return self._hydraulic_radius

    @property
    def heat_power(self) -> float:
        """Total heat input to heat transfer region (W)."""
        return self._heat_power

    @property
    def heat_flux_per_length(self) -> float:
        """Heat flux per unit length (W/m)."""
        return self._heat_flux_per_length

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Parameters
        ----------
        x : float
            Axial position within the segment (m).
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1), T_m].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K) - not used, taken from state.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1 = y[0] + 1j * y[1]
        U1 = y[2] + 1j * y[3]
        T_local = y[4]

        # Ensure temperature stays physical
        T_local = max(T_local, 10.0)
        T_local = min(T_local, 10000.0)

        A = self._area
        A_gas = self._gas_area_fraction * A
        rh = self._hydraulic_radius

        # Gas properties at local temperature
        rho_m = gas.density(T_local)
        a = gas.sound_speed(T_local)
        gamma = gas.gamma(T_local)
        mu = gas.viscosity(T_local)
        k_gas = gas.thermal_conductivity(T_local)
        cp = gas.specific_heat_cp(T_local)

        # Penetration depths
        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        # Boundary layer f functions (approximate for large channels)
        f_nu = (1 - 1j) * delta_nu / rh
        f_kappa = (1 - 1j) * delta_kappa / rh

        # Determine region
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_heat_transfer

        in_heat_transfer_region = x1 <= x < x2

        # Momentum equation: dp1/dx = -iωρm/(1-fν)/Agas * U1
        denom_mom = (1 - f_nu) * A_gas
        if np.abs(denom_mom) > 1e-20:
            dp1_dx = -1j * omega * rho_m / denom_mom * U1
        else:
            dp1_dx = 0.0 + 0.0j

        # Continuity equation: dU1/dx = -iωAgas/(ρm*a²) * [1 + (γ-1)*fκ] * p1
        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

        # Temperature equation
        if in_heat_transfer_region:
            # Fixed heat flux per unit length
            q_dot = self._heat_flux_per_length

            # For oscillating flow, effective mass flow rate relates to velocity amplitude
            # ṁ_eff ~ ρ * |u1| * A_gas (time-averaged sense)
            u1_mag = np.abs(U1) / A_gas if A_gas > 1e-20 else 0.0

            if u1_mag > 1e-10:
                # dTm/dx = q̇ / (ṁ * cp) where ṁ = ρ * u * A_gas
                # For oscillating flow, use magnitude
                m_dot_eff = rho_m * u1_mag * A_gas

                # Protect against division by zero
                if m_dot_eff * cp > 1e-20:
                    dTm_dx = q_dot / (m_dot_eff * cp)
                else:
                    dTm_dx = 0.0
            else:
                # Very low velocity - use conduction-dominated estimate
                # Heat accumulates locally, temperature changes gradually
                if np.abs(q_dot) > 1e-10:
                    # Use thermal diffusivity time scale
                    # dTm/dx ~ q̇ / (k * A) but limited
                    dTm_dx = q_dot / (rho_m * cp * A_gas * 0.01)  # 1 cm/s effective
                else:
                    dTm_dx = 0.0
        else:
            # In tubesheet regions: no heat transfer
            dTm_dx = 0.0

        # Limit temperature change rate
        max_dT_per_m = 10000.0  # Maximum 10000 K/m for VXQ
        dTm_dx = max(min(dTm_dx, max_dT_per_m), -max_dT_per_m)

        return np.array([
            dp1_dx.real, dp1_dx.imag,
            dU1_dx.real, dU1_dx.imag,
            dTm_dx
        ])

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the VXQ1 heat exchanger.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature at input (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        if self._length == 0:
            return p1_in, U1_in, T_m

        # Ensure input temperature is physical
        T_m = max(T_m, 10.0)
        T_m = min(T_m, 10000.0)

        # Initial state vector: [Re(p1), Im(p1), Re(U1), Im(U1), T_m]
        y0 = np.array([
            p1_in.real, p1_in.imag,
            U1_in.real, U1_in.imag,
            T_m
        ])

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, y[4])

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
            max_step=self._length / 10,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        y_out = sol.y[:, -1]
        p1_out = y_out[0] + 1j * y_out[1]
        U1_out = y_out[2] + 1j * y_out[3]
        T_m_out = y_out[4]

        # Ensure output temperature is physical
        T_m_out = max(T_m_out, 10.0)
        T_m_out = min(T_m_out, 10000.0)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return (
            f"VariableHeatFluxHeatExchanger(name='{self._name}', "
            f"length={self._length}, heat_power={self._heat_power})"
        )


class VariableHeatFluxHeatExchanger2Pass(Segment):
    """
    VXQ2 segment - Two-pass variable gas temperature heat exchanger with fixed heat flux.

    Models a heat exchanger with two separate heat transfer regions, each
    with its own fixed heat input. This is useful for shell-and-tube
    heat exchangers with two passes on the shell side.

    The segment consists of four regions:
    - Tubesheet 1 (inlet): Viscous/thermal losses, no heat transfer
    - Heat transfer region 1: Heat flux q̇_1 = HeatP1 / L_pass1
    - Heat transfer region 2: Heat flux q̇_2 = HeatP2 / L_pass2
    - Tubesheet 2 (outlet): Viscous/thermal losses, no heat transfer

    Parameters
    ----------
    area : float
        Total cross-sectional area A = Agas + Asolid + Ashell-side (m²).
    gas_area_fraction : float
        Fraction of total area available to gas, Agas/A.
    solid_area_fraction : float
        Fraction of total area for solid thermal conduction, Asolid/A.
    hydraulic_radius : float
        Hydraulic radius of gas channels (m).
    length_tubesheet1 : float
        Length of inlet tubesheet (m). No heat transfer in this region.
    length_pass1 : float
        Length of first heat transfer region (m).
    length_pass2 : float
        Length of second heat transfer region (m).
    length_tubesheet2 : float
        Length of outlet tubesheet (m). No heat transfer in this region.
    heat_power_1 : float
        Total heat input to pass 1 (W). Positive = heating.
    heat_power_2 : float
        Total heat input to pass 2 (W). Positive = heating.
    solid_thermal_conductivity : float, optional
        Thermal conductivity of solid (W/(m·K)). Default 15.0 (stainless steel).
    name : str, optional
        Name identifier for the segment.
    """

    def __init__(
        self,
        area: float,
        gas_area_fraction: float,
        solid_area_fraction: float,
        hydraulic_radius: float,
        length_tubesheet1: float,
        length_pass1: float,
        length_pass2: float,
        length_tubesheet2: float,
        heat_power_1: float,
        heat_power_2: float,
        solid_thermal_conductivity: float = 15.0,
        name: str = "",
    ) -> None:
        if not 0 < gas_area_fraction < 1:
            raise ValueError(f"gas_area_fraction must be in (0, 1), got {gas_area_fraction}")
        if solid_area_fraction < 0 or solid_area_fraction >= 1:
            raise ValueError(f"solid_area_fraction must be in [0, 1), got {solid_area_fraction}")
        if length_pass1 <= 0:
            raise ValueError(f"length_pass1 must be positive, got {length_pass1}")
        if length_pass2 <= 0:
            raise ValueError(f"length_pass2 must be positive, got {length_pass2}")

        total_length = length_tubesheet1 + length_pass1 + length_pass2 + length_tubesheet2

        self._gas_area_fraction = gas_area_fraction
        self._solid_area_fraction = solid_area_fraction
        self._hydraulic_radius = hydraulic_radius
        self._length_tubesheet1 = length_tubesheet1
        self._length_pass1 = length_pass1
        self._length_pass2 = length_pass2
        self._length_tubesheet2 = length_tubesheet2
        self._heat_power_1 = heat_power_1
        self._heat_power_2 = heat_power_2
        self._solid_thermal_conductivity = solid_thermal_conductivity

        # Heat flux per unit length for each pass (W/m)
        self._heat_flux_1 = heat_power_1 / length_pass1
        self._heat_flux_2 = heat_power_2 / length_pass2

        super().__init__(name=name, length=total_length, area=area, geometry=None)

    @property
    def gas_area_fraction(self) -> float:
        """Fraction of area available to gas."""
        return self._gas_area_fraction

    @property
    def solid_area_fraction(self) -> float:
        """Fraction of area for solid conduction."""
        return self._solid_area_fraction

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of gas channels (m)."""
        return self._hydraulic_radius

    @property
    def heat_power_1(self) -> float:
        """Total heat input to pass 1 (W)."""
        return self._heat_power_1

    @property
    def heat_power_2(self) -> float:
        """Total heat input to pass 2 (W)."""
        return self._heat_power_2

    def _get_heat_flux(self, x: float) -> float:
        """Get the heat flux per unit length at position x."""
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_pass1
        x3 = x2 + self._length_pass2

        if x < x1:
            # Tubesheet 1 - no heat transfer
            return 0.0
        elif x < x2:
            # Pass 1
            return self._heat_flux_1
        elif x < x3:
            # Pass 2
            return self._heat_flux_2
        else:
            # Tubesheet 2 - no heat transfer
            return 0.0

    def _in_heat_transfer_region(self, x: float) -> bool:
        """Check if position x is in a heat transfer region."""
        x1 = self._length_tubesheet1
        x2 = x1 + self._length_pass1
        x3 = x2 + self._length_pass2

        return x1 <= x < x3

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Parameters
        ----------
        x : float
            Axial position within the segment (m).
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1), T_m].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K) - not used, taken from state.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1 = y[0] + 1j * y[1]
        U1 = y[2] + 1j * y[3]
        T_local = y[4]

        # Ensure temperature stays physical
        T_local = max(T_local, 10.0)
        T_local = min(T_local, 10000.0)

        A = self._area
        A_gas = self._gas_area_fraction * A
        rh = self._hydraulic_radius

        # Gas properties at local temperature
        rho_m = gas.density(T_local)
        a = gas.sound_speed(T_local)
        gamma = gas.gamma(T_local)
        mu = gas.viscosity(T_local)
        k_gas = gas.thermal_conductivity(T_local)
        cp = gas.specific_heat_cp(T_local)

        # Penetration depths
        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        # Boundary layer f functions
        f_nu = (1 - 1j) * delta_nu / rh
        f_kappa = (1 - 1j) * delta_kappa / rh

        in_heat_transfer = self._in_heat_transfer_region(x)
        q_dot = self._get_heat_flux(x)

        # Momentum equation
        denom_mom = (1 - f_nu) * A_gas
        if np.abs(denom_mom) > 1e-20:
            dp1_dx = -1j * omega * rho_m / denom_mom * U1
        else:
            dp1_dx = 0.0 + 0.0j

        # Continuity equation
        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

        # Temperature equation
        if in_heat_transfer and np.abs(q_dot) > 1e-20:
            u1_mag = np.abs(U1) / A_gas if A_gas > 1e-20 else 0.0

            if u1_mag > 1e-10:
                m_dot_eff = rho_m * u1_mag * A_gas
                if m_dot_eff * cp > 1e-20:
                    dTm_dx = q_dot / (m_dot_eff * cp)
                else:
                    dTm_dx = 0.0
            else:
                if np.abs(q_dot) > 1e-10:
                    dTm_dx = q_dot / (rho_m * cp * A_gas * 0.01)
                else:
                    dTm_dx = 0.0
        else:
            dTm_dx = 0.0

        # Limit temperature change rate
        max_dT_per_m = 10000.0
        dTm_dx = max(min(dTm_dx, max_dT_per_m), -max_dT_per_m)

        return np.array([
            dp1_dx.real, dp1_dx.imag,
            dU1_dx.real, dU1_dx.imag,
            dTm_dx
        ])

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the VXQ2 heat exchanger.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature at input (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        if self._length == 0:
            return p1_in, U1_in, T_m

        T_m = max(T_m, 10.0)
        T_m = min(T_m, 10000.0)

        y0 = np.array([
            p1_in.real, p1_in.imag,
            U1_in.real, U1_in.imag,
            T_m
        ])

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, y[4])

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
            max_step=self._length / 20,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        y_out = sol.y[:, -1]
        p1_out = y_out[0] + 1j * y_out[1]
        U1_out = y_out[2] + 1j * y_out[3]
        T_m_out = y_out[4]

        T_m_out = max(T_m_out, 10.0)
        T_m_out = min(T_m_out, 10000.0)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return (
            f"VariableHeatFluxHeatExchanger2Pass(name='{self._name}', "
            f"length={self._length}, heat_power_1={self._heat_power_1}, "
            f"heat_power_2={self._heat_power_2})"
        )


# reference baseline aliases
VXT1 = VariableTemperatureHeatExchanger
VXT2 = VariableTemperatureHeatExchanger2Pass
VXQ1 = VariableHeatFluxHeatExchanger
VXQ2 = VariableHeatFluxHeatExchanger2Pass
