"""Shooting method solver for thermoacoustic eigenvalue problems.

This module provides classes for solving thermoacoustic network problems
using the shooting method, where initial guesses are iteratively refined
until boundary conditions at the end of the network are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from openthermoacoustics.solver.network import NetworkTopology

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


@dataclass
class SolverResult:
    """
    Results from the shooting method solver.

    Attributes
    ----------
    frequency : float
        Resonance frequency (Hz).
    omega : float
        Angular frequency (rad/s).
    p1_profile : NDArray[np.complex128]
        Complex pressure amplitude along the network (Pa).
    U1_profile : NDArray[np.complex128]
        Complex volumetric velocity amplitude along the network (m^3/s).
    T_m_profile : NDArray[np.float64]
        Mean temperature along the network (K).
    x_profile : NDArray[np.float64]
        Position along the network (m).
    acoustic_power : NDArray[np.float64]
        Time-averaged acoustic power along the network (W).
    converged : bool
        Whether the solver converged.
    message : str
        Status message from the solver.
    n_iterations : int
        Number of iterations taken.
    residual_norm : float
        Final residual norm.
    guesses_final : dict[str, float]
        Final values of the shooting parameters.
    """

    frequency: float
    omega: float
    p1_profile: NDArray[np.complex128]
    U1_profile: NDArray[np.complex128]
    T_m_profile: NDArray[np.float64]
    x_profile: NDArray[np.float64]
    acoustic_power: NDArray[np.float64]
    converged: bool
    message: str
    n_iterations: int
    residual_norm: float
    guesses_final: dict[str, float]

    def __repr__(self) -> str:
        """Return a string representation of the result."""
        status = "converged" if self.converged else "NOT converged"
        return (
            f"SolverResult({status}, f={self.frequency:.2f} Hz, "
            f"iterations={self.n_iterations}, residual={self.residual_norm:.2e})"
        )

    def plot_profiles(
        self,
        figsize: tuple[float, float] = (10, 8),
        show: bool = True,
    ) -> Any:
        """
        Plot the acoustic profiles along the network.

        Creates a figure with four subplots showing pressure amplitude,
        velocity amplitude, acoustic power, and temperature.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (10, 8).
        show : bool, optional
            Whether to display the figure, by default True.

        Returns
        -------
        Any
            Matplotlib figure object, or None if matplotlib is not available.

        Notes
        -----
        Requires matplotlib to be installed. If matplotlib is not available,
        a warning is printed and None is returned.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Warning: matplotlib not available. Cannot plot profiles. "
                "Install with: pip install matplotlib"
            )
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        x_mm = self.x_profile * 1000  # Convert to mm

        # Pressure amplitude
        ax = axes[0, 0]
        ax.plot(x_mm, np.abs(self.p1_profile), "b-", linewidth=1.5)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("|p1| (Pa)")
        ax.set_title("Pressure Amplitude")
        ax.grid(True, alpha=0.3)

        # Velocity amplitude
        ax = axes[0, 1]
        ax.plot(x_mm, np.abs(self.U1_profile) * 1e6, "r-", linewidth=1.5)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("|U1| (mm^3/s)")
        ax.set_title("Volumetric Velocity Amplitude")
        ax.grid(True, alpha=0.3)

        # Acoustic power
        ax = axes[1, 0]
        ax.plot(x_mm, self.acoustic_power, "g-", linewidth=1.5)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Acoustic Power (W)")
        ax.set_title("Time-Averaged Acoustic Power")
        ax.grid(True, alpha=0.3)

        # Temperature
        ax = axes[1, 1]
        ax.plot(x_mm, self.T_m_profile, "m-", linewidth=1.5)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Mean Temperature")
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Acoustic Profiles at f = {self.frequency:.2f} Hz "
            f"({'Converged' if self.converged else 'Not Converged'})",
            fontsize=12,
        )
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_complex_profiles(
        self,
        figsize: tuple[float, float] = (12, 6),
        show: bool = True,
    ) -> Any:
        """
        Plot the real and imaginary parts of pressure and velocity.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (12, 6).
        show : bool, optional
            Whether to display the figure, by default True.

        Returns
        -------
        Any
            Matplotlib figure object, or None if matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Warning: matplotlib not available. Cannot plot profiles. "
                "Install with: pip install matplotlib"
            )
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        x_mm = self.x_profile * 1000

        # Pressure
        ax = axes[0]
        ax.plot(x_mm, self.p1_profile.real, "b-", label="Re(p1)", linewidth=1.5)
        ax.plot(
            x_mm, self.p1_profile.imag, "b--", label="Im(p1)", linewidth=1.5
        )
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("p1 (Pa)")
        ax.set_title("Complex Pressure Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes[1]
        ax.plot(
            x_mm,
            self.U1_profile.real * 1e6,
            "r-",
            label="Re(U1)",
            linewidth=1.5,
        )
        ax.plot(
            x_mm,
            self.U1_profile.imag * 1e6,
            "r--",
            label="Im(U1)",
            linewidth=1.5,
        )
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("U1 (mm^3/s)")
        ax.set_title("Complex Volumetric Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"f = {self.frequency:.2f} Hz", fontsize=12)
        plt.tight_layout()

        if show:
            plt.show()

        return fig


class ShootingSolver:
    """
    Shooting method solver for thermoacoustic network eigenvalue problems.

    The shooting method works by:
    1. Making initial guesses for unknown parameters (e.g., frequency,
       pressure amplitude)
    2. Propagating through the network with these guesses
    3. Computing the residual (difference between computed and target values)
    4. Using an optimization algorithm to adjust guesses until residual is zero

    Parameters
    ----------
    network : NetworkTopology
        The thermoacoustic network to solve.
    gas : Gas
        Gas properties object.

    Attributes
    ----------
    network : NetworkTopology
        The network being solved.
    gas : Gas
        The gas properties.

    Examples
    --------
    >>> from openthermoacoustics.segments import Duct
    >>> from openthermoacoustics.gas import Helium
    >>> from openthermoacoustics.solver import NetworkTopology, ShootingSolver
    >>> # Create a simple resonator
    >>> network = NetworkTopology()
    >>> network.add_segment(Duct(length=0.34, diameter=0.05))
    >>> gas = Helium(mean_pressure=1e5)
    >>> solver = ShootingSolver(network, gas)
    >>> # Solve for closed-closed resonance (U1=0 at both ends)
    >>> result = solver.solve(
    ...     guesses={'p1_amplitude': 1000, 'frequency': 500},
    ...     targets={'U1_end_real': 0, 'U1_end_imag': 0},
    ... )
    """

    # Default solver options
    DEFAULT_OPTIONS: dict[str, Any] = {
        "method": "hybr",  # scipy.optimize.root method
        "tol": 1e-10,  # Tolerance for convergence
        "maxiter": 100,  # Maximum iterations
        "n_points_per_segment": 100,  # Resolution for integration
        "T_m_start": 300.0,  # Default inlet temperature
        "verbose": False,  # Print progress
    }

    def __init__(self, network: NetworkTopology, gas: Gas) -> None:
        """
        Initialize the shooting method solver.

        Parameters
        ----------
        network : NetworkTopology
            The thermoacoustic network to solve.
        gas : Gas
            Gas properties object.
        """
        self.network = network
        self.gas = gas

    def solve(
        self,
        guesses: dict[str, float],
        targets: dict[str, float],
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """
        Solve the thermoacoustic network problem using the shooting method.

        Parameters
        ----------
        guesses : dict[str, float]
            Initial guesses for the shooting parameters. Supported keys:
            - 'p1_amplitude': Pressure amplitude at inlet (Pa)
            - 'p1_phase': Pressure phase at inlet (radians)
            - 'U1_amplitude': Velocity amplitude at inlet (m^3/s)
            - 'U1_phase': Velocity phase at inlet (radians)
            - 'frequency': Oscillation frequency (Hz)
            - 'p1_real': Real part of p1 at inlet (Pa)
            - 'p1_imag': Imaginary part of p1 at inlet (Pa)
            - 'U1_real': Real part of U1 at inlet (m^3/s)
            - 'U1_imag': Imaginary part of U1 at inlet (m^3/s)
        targets : dict[str, float]
            Target values to match at the network outlet. Supported keys:
            - 'U1_end_real': Real part of U1 at outlet (closed end: 0)
            - 'U1_end_imag': Imaginary part of U1 at outlet (closed end: 0)
            - 'p1_end_real': Real part of p1 at outlet (open end: 0)
            - 'p1_end_imag': Imaginary part of p1 at outlet (open end: 0)
            - 'impedance_real': Real part of Z = p1/U1 at outlet
            - 'impedance_imag': Imaginary part of Z = p1/U1 at outlet
        options : dict[str, Any], optional
            Solver options. Supported keys:
            - 'method': scipy.optimize.root method (default: 'hybr')
            - 'tol': Convergence tolerance (default: 1e-10)
            - 'maxiter': Maximum iterations (default: 100)
            - 'n_points_per_segment': Integration resolution (default: 100)
            - 'T_m_start': Inlet temperature in K (default: 300)
            - 'verbose': Print progress (default: False)

        Returns
        -------
        SolverResult
            Results of the solve, including profiles and convergence info.

        Raises
        ------
        ValueError
            If guesses or targets are invalid or incompatible.

        Notes
        -----
        The number of guesses must equal the number of targets for the
        system to be well-posed.

        Common configurations:
        - Closed-open resonator: Fix p1 at inlet, target U1=0 at inlet
          and p1=0 at outlet. Guess frequency.
        - Closed-closed resonator: Fix p1 at inlet, target U1=0 at inlet
          and U1=0 at outlet. Guess frequency.
        """
        # Merge options with defaults
        opts = self.DEFAULT_OPTIONS.copy()
        if options is not None:
            opts.update(options)

        # Validate inputs
        self._validate_guesses(guesses)
        self._validate_targets(targets)

        if len(guesses) != len(targets):
            raise ValueError(
                f"Number of guesses ({len(guesses)}) must equal "
                f"number of targets ({len(targets)})"
            )

        # Build the shooting vector and bounds
        guess_keys = list(guesses.keys())
        target_keys = list(targets.keys())
        x0 = np.array([guesses[k] for k in guess_keys])
        target_values = np.array([targets[k] for k in target_keys])

        # Count iterations
        iteration_count = [0]

        def residual_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute residuals for the shooting method."""
            iteration_count[0] += 1

            # Unpack current guesses
            current_guesses = dict(zip(guess_keys, x))

            # Construct initial conditions
            p1_start, U1_start, omega = self._construct_initial_conditions(
                current_guesses, opts
            )

            # Propagate through network
            try:
                self.network.propagate_all(
                    p1_start=p1_start,
                    U1_start=U1_start,
                    T_m_start=opts["T_m_start"],
                    omega=omega,
                    gas=self.gas,
                    n_points_per_segment=opts["n_points_per_segment"],
                )
            except RuntimeError as e:
                # Integration failed - return large residual
                if opts["verbose"]:
                    print(f"  Integration failed: {e}")
                return np.full(len(target_keys), 1e10)

            # Get endpoint values
            endpoints = self.network.get_endpoint_values()

            # Compute residuals
            residuals = self._compute_residuals(
                endpoints, target_keys, target_values, opts
            )

            if opts["verbose"]:
                print(
                    f"  Iteration {iteration_count[0]}: "
                    f"residual norm = {np.linalg.norm(residuals):.2e}"
                )

            return residuals

        # Solve using scipy.optimize.root
        if opts["verbose"]:
            print("Starting shooting method solver...")
            print(f"  Guesses: {guess_keys}")
            print(f"  Targets: {target_keys}")

        root_options: dict[str, Any] = {}
        if opts["method"] == "hybr":
            root_options["maxfev"] = opts["maxiter"] * len(x0)
        elif opts["method"] == "lm":
            root_options["maxiter"] = opts["maxiter"] * len(x0)
        else:
            root_options["maxiter"] = opts["maxiter"] * len(x0)

        result = root(
            residual_func,
            x0,
            method=opts["method"],
            tol=opts["tol"],
            options=root_options,
        )

        # Extract final values
        final_guesses = dict(zip(guess_keys, result.x))

        # Get final profiles
        p1_start, U1_start, omega = self._construct_initial_conditions(
            final_guesses, opts
        )

        self.network.propagate_all(
            p1_start=p1_start,
            U1_start=U1_start,
            T_m_start=opts["T_m_start"],
            omega=omega,
            gas=self.gas,
            n_points_per_segment=opts["n_points_per_segment"],
        )

        profiles = self.network.get_global_profiles()

        # Extract frequency
        frequency = final_guesses.get("frequency", omega / (2 * np.pi))

        return SolverResult(
            frequency=frequency,
            omega=omega,
            p1_profile=profiles["p1"],
            U1_profile=profiles["U1"],
            T_m_profile=profiles["T_m"],
            x_profile=profiles["x"],
            acoustic_power=profiles["acoustic_power"],
            converged=result.success,
            message=result.message,
            n_iterations=iteration_count[0],
            residual_norm=float(np.linalg.norm(result.fun)),
            guesses_final=final_guesses,
        )

    def _validate_guesses(self, guesses: dict[str, float]) -> None:
        """
        Validate the guesses dictionary.

        Parameters
        ----------
        guesses : dict[str, float]
            Guesses to validate.

        Raises
        ------
        ValueError
            If guesses are invalid.
        """
        valid_keys = {
            "p1_amplitude",
            "p1_phase",
            "U1_amplitude",
            "U1_phase",
            "frequency",
            "p1_real",
            "p1_imag",
            "U1_real",
            "U1_imag",
        }

        for key in guesses:
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown guess key '{key}'. Valid keys: {valid_keys}"
                )

        # Check for conflicting specifications
        if "p1_amplitude" in guesses and "p1_real" in guesses:
            raise ValueError(
                "Cannot specify both 'p1_amplitude' and 'p1_real'"
            )
        if "U1_amplitude" in guesses and "U1_real" in guesses:
            raise ValueError(
                "Cannot specify both 'U1_amplitude' and 'U1_real'"
            )

    def _validate_targets(self, targets: dict[str, float]) -> None:
        """
        Validate the targets dictionary.

        Parameters
        ----------
        targets : dict[str, float]
            Targets to validate.

        Raises
        ------
        ValueError
            If targets are invalid.
        """
        valid_keys = {
            "U1_end_real",
            "U1_end_imag",
            "p1_end_real",
            "p1_end_imag",
            "impedance_real",
            "impedance_imag",
        }

        for key in targets:
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown target key '{key}'. Valid keys: {valid_keys}"
                )

    def _construct_initial_conditions(
        self,
        guesses: dict[str, float],
        opts: dict[str, Any],
    ) -> tuple[complex, complex, float]:
        """
        Construct initial p1, U1, and omega from guesses.

        Parameters
        ----------
        guesses : dict[str, float]
            Current guess values.
        opts : dict[str, Any]
            Solver options.

        Returns
        -------
        tuple[complex, complex, float]
            (p1_start, U1_start, omega)
        """
        # Default values
        p1_amplitude = guesses.get("p1_amplitude", 1000.0)
        p1_phase = guesses.get("p1_phase", 0.0)
        U1_amplitude = guesses.get("U1_amplitude", 0.0)
        U1_phase = guesses.get("U1_phase", 0.0)
        frequency = guesses.get("frequency", 100.0)

        # Build complex amplitudes
        if "p1_real" in guesses or "p1_imag" in guesses:
            p1_real = guesses.get("p1_real", 0.0)
            p1_imag = guesses.get("p1_imag", 0.0)
            p1_start = complex(p1_real, p1_imag)
        else:
            p1_start = p1_amplitude * np.exp(1j * p1_phase)

        if "U1_real" in guesses or "U1_imag" in guesses:
            U1_real = guesses.get("U1_real", 0.0)
            U1_imag = guesses.get("U1_imag", 0.0)
            U1_start = complex(U1_real, U1_imag)
        else:
            U1_start = U1_amplitude * np.exp(1j * U1_phase)

        omega = 2 * np.pi * frequency

        return p1_start, U1_start, omega

    def _compute_residuals(
        self,
        endpoints: dict[str, complex | float],
        target_keys: list[str],
        target_values: NDArray[np.float64],
        opts: dict[str, Any],
    ) -> NDArray[np.float64]:
        """
        Compute residuals between computed and target values.

        Parameters
        ----------
        endpoints : dict[str, complex | float]
            Endpoint values from network propagation.
        target_keys : list[str]
            Keys of target values.
        target_values : NDArray[np.float64]
            Target values to match.
        opts : dict[str, Any]
            Solver options.

        Returns
        -------
        NDArray[np.float64]
            Residual vector.
        """
        residuals = np.zeros(len(target_keys))

        p1_end = endpoints["p1_end"]
        U1_end = endpoints["U1_end"]

        # Compute impedance if needed
        if abs(U1_end) > 1e-20:
            Z_end = p1_end / U1_end
        else:
            Z_end = complex(1e20, 0)  # Large impedance for closed end

        for i, key in enumerate(target_keys):
            if key == "U1_end_real":
                computed = U1_end.real
            elif key == "U1_end_imag":
                computed = U1_end.imag
            elif key == "p1_end_real":
                computed = p1_end.real
            elif key == "p1_end_imag":
                computed = p1_end.imag
            elif key == "impedance_real":
                computed = Z_end.real
            elif key == "impedance_imag":
                computed = Z_end.imag
            else:
                computed = 0.0

            residuals[i] = computed - target_values[i]

        return residuals

    def solve_closed_closed(
        self,
        p1_amplitude: float = 1000.0,
        frequency_guess: float = 100.0,
        T_m_start: float = 300.0,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """
        Solve for a closed-closed resonator (U1=0 at both ends).

        This is a convenience method for the common case of finding
        standing wave modes in a resonator with rigid ends.

        Parameters
        ----------
        p1_amplitude : float, optional
            Pressure amplitude to fix at inlet (Pa), by default 1000.
        frequency_guess : float, optional
            Initial guess for frequency (Hz), by default 100.
        T_m_start : float, optional
            Inlet temperature (K), by default 300.
        options : dict[str, Any], optional
            Additional solver options.

        Returns
        -------
        SolverResult
            Solution results.
        """
        opts = options.copy() if options else {}
        opts["T_m_start"] = T_m_start

        # For closed-closed: U1=0 at inlet and outlet
        # We fix p1 amplitude, guess frequency
        # Two constraints: U1_end_real=0, U1_end_imag=0
        # But we also need U1_start=0, so only frequency is free

        # Actually for closed-closed starting from a pressure antinode,
        # we have U1_start = 0 automatically (it's a node).
        # Then we shoot for U1_end = 0.

        return self.solve(
            guesses={"frequency": frequency_guess},
            targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
            options=opts,
        )

    def solve_closed_open(
        self,
        p1_amplitude: float = 1000.0,
        frequency_guess: float = 100.0,
        T_m_start: float = 300.0,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """
        Solve for a closed-open resonator (U1=0 at inlet, p1=0 at outlet).

        This is a convenience method for quarter-wave resonators.

        Parameters
        ----------
        p1_amplitude : float, optional
            Pressure amplitude to fix at inlet (Pa), by default 1000.
        frequency_guess : float, optional
            Initial guess for frequency (Hz), by default 100.
        T_m_start : float, optional
            Inlet temperature (K), by default 300.
        options : dict[str, Any], optional
            Additional solver options.

        Returns
        -------
        SolverResult
            Solution results.
        """
        opts = options.copy() if options else {}
        opts["T_m_start"] = T_m_start

        return self.solve(
            guesses={"frequency": frequency_guess},
            targets={"p1_end_real": 0.0, "p1_end_imag": 0.0},
            options=opts,
        )

    def __repr__(self) -> str:
        """Return a string representation of the solver."""
        return f"ShootingSolver(network={self.network!r}, gas={self.gas!r})"
