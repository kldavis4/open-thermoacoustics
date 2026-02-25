"""High-level API for thermoacoustic engine and refrigerator analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openthermoacoustics.solver.network import NetworkTopology
from openthermoacoustics.solver.shooting import ShootingSolver, SolverResult

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.segments.base import Segment


class Network:
    """
    High-level interface for building and solving thermoacoustic networks.

    This class provides a user-friendly API for constructing thermoacoustic
    engine or refrigerator models and solving for their operating conditions.

    Parameters
    ----------
    gas : Gas
        Working gas instance with thermophysical properties.
    frequency_guess : float, optional
        Initial guess for operating frequency in Hz. Default is 100.0.

    Examples
    --------
    >>> import openthermoacoustics as ota
    >>>
    >>> # Create network with helium at 30 bar
    >>> gas = ota.gas.Helium(mean_pressure=3e6)
    >>> engine = ota.Network(gas=gas, frequency_guess=84.0)
    >>>
    >>> # Add segments
    >>> engine.add(ota.segments.HardEnd())
    >>> engine.add(ota.segments.Duct(length=0.5, radius=0.05))
    >>> engine.add(ota.segments.Duct(length=1.0, radius=0.05))
    >>> engine.add(ota.segments.HardEnd())
    >>>
    >>> # Solve for resonance
    >>> result = engine.solve(
    ...     p1_amplitude=1e4,
    ...     targets={'U1_end': 0}
    ... )
    >>> print(f"Frequency: {result.frequency:.2f} Hz")
    """

    def __init__(self, gas: Gas, frequency_guess: float = 100.0) -> None:
        """
        Initialize a thermoacoustic network.

        Parameters
        ----------
        gas : Gas
            Working gas instance.
        frequency_guess : float, optional
            Initial guess for frequency in Hz.
        """
        self.gas = gas
        self.frequency_guess = frequency_guess
        self._topology = NetworkTopology()
        self._segments: list[Segment] = []

    def add(self, segment: Segment) -> "Network":
        """
        Add a segment to the network.

        Parameters
        ----------
        segment : Segment
            Acoustic network segment to add.

        Returns
        -------
        Network
            Self, for method chaining.

        Examples
        --------
        >>> engine.add(ota.segments.Duct(length=0.5, radius=0.05))
        >>> engine.add(ota.segments.Stack(...)).add(ota.segments.Duct(...))
        """
        self._segments.append(segment)
        self._topology.add_segment(segment)
        return self

    @property
    def segments(self) -> list[Segment]:
        """List of segments in the network."""
        return self._segments.copy()

    @property
    def total_length(self) -> float:
        """Total length of the network in meters."""
        return self._topology.total_length

    def solve(
        self,
        p1_amplitude: float = 1e4,
        p1_phase: float = 0.0,
        frequency: float | None = None,
        T_m_start: float = 300.0,
        targets: dict[str, float] | None = None,
        solve_frequency: bool = True,
        method: str = "hybr",
        tol: float = 1e-9,
        maxiter: int = 100,
        verbose: bool = False,
    ) -> SolverResult:
        """
        Solve for the operating conditions of the thermoacoustic network.

        Uses a shooting method to find the pressure amplitude, phase, and
        optionally frequency that satisfy the specified target conditions.

        Parameters
        ----------
        p1_amplitude : float, optional
            Initial guess for pressure amplitude in Pa. Default is 1e4.
        p1_phase : float, optional
            Initial guess for pressure phase in radians. Default is 0.0.
        frequency : float, optional
            Initial guess for frequency in Hz. If None, uses frequency_guess.
        T_m_start : float, optional
            Mean temperature at start of network in K. Default is 300.0.
        targets : dict[str, float], optional
            Target conditions to satisfy. Keys can be:
            - 'U1_end': Target U1 magnitude at end (use 0 for closed end)
            - 'U1_end_real': Target real part of U1 at end
            - 'U1_end_imag': Target imaginary part of U1 at end
            - 'p1_end': Target p1 magnitude at end (use 0 for open end)
            - 'p1_end_real': Target real part of p1 at end
            - 'p1_end_imag': Target imaginary part of p1 at end
            Default is {'U1_end_real': 0, 'U1_end_imag': 0} (closed end).
        solve_frequency : bool, optional
            Whether to solve for frequency. Default is True.
        method : str, optional
            Optimization method for scipy.optimize.root. Default is 'hybr'.
        tol : float, optional
            Tolerance for convergence. Default is 1e-9.
        maxiter : int, optional
            Maximum iterations. Default is 100.
        verbose : bool, optional
            Print convergence information. Default is False.

        Returns
        -------
        SolverResult
            Solution containing frequency, profiles, and convergence info.

        Raises
        ------
        ValueError
            If the network has no segments or solver fails to converge.

        Examples
        --------
        >>> # Solve closed-closed resonator
        >>> result = engine.solve(p1_amplitude=1e4, targets={'U1_end': 0})
        >>>
        >>> # Solve with fixed frequency
        >>> result = engine.solve(
        ...     p1_amplitude=1e4,
        ...     frequency=100.0,
        ...     solve_frequency=False,
        ...     targets={'U1_end_real': 0, 'U1_end_imag': 0}
        ... )
        """
        if not self._segments:
            raise ValueError("Network has no segments. Add segments with add().")

        # Set up guesses
        freq = frequency if frequency is not None else self.frequency_guess
        guesses: dict[str, float] = {
            "p1_amplitude": p1_amplitude,
            "p1_phase": p1_phase,
        }
        if solve_frequency:
            guesses["frequency"] = freq

        # Set up targets
        if targets is None:
            targets = {"U1_end_real": 0.0, "U1_end_imag": 0.0}
        else:
            # Convert simplified targets
            targets = self._expand_targets(targets)

        # Create solver and solve
        solver = ShootingSolver(self._topology, self.gas)
        options = {
            "method": method,
            "tol": tol,
            "maxiter": maxiter,
            "T_m_start": T_m_start,
            "verbose": verbose,
        }

        # If not solving for frequency, we need to set it
        if not solve_frequency:
            options["fixed_frequency"] = freq

        return solver.solve(guesses, targets, options)

    def _expand_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Expand simplified target specifications to full form."""
        expanded: dict[str, float] = {}

        for key, value in targets.items():
            if key == "U1_end":
                # Magnitude target: set both real and imag to 0
                if value == 0:
                    expanded["U1_end_real"] = 0.0
                    expanded["U1_end_imag"] = 0.0
                else:
                    raise ValueError(
                        "Non-zero U1_end magnitude target not supported. "
                        "Use U1_end_real and U1_end_imag instead."
                    )
            elif key == "p1_end":
                if value == 0:
                    expanded["p1_end_real"] = 0.0
                    expanded["p1_end_imag"] = 0.0
                else:
                    raise ValueError(
                        "Non-zero p1_end magnitude target not supported. "
                        "Use p1_end_real and p1_end_imag instead."
                    )
            else:
                expanded[key] = value

        return expanded

    def solve_closed_closed(
        self,
        p1_amplitude: float = 1e4,
        T_m_start: float = 300.0,
        **kwargs,
    ) -> SolverResult:
        """
        Solve for a closed-closed resonator (U1=0 at both ends).

        This is a convenience method for the common case of a resonator
        with rigid terminations at both ends.

        Parameters
        ----------
        p1_amplitude : float, optional
            Initial guess for pressure amplitude in Pa.
        T_m_start : float, optional
            Mean temperature at start in K.
        **kwargs
            Additional arguments passed to solve().

        Returns
        -------
        SolverResult
            Solution for the resonator.
        """
        return self.solve(
            p1_amplitude=p1_amplitude,
            T_m_start=T_m_start,
            targets={"U1_end": 0},
            **kwargs,
        )

    def solve_closed_open(
        self,
        p1_amplitude: float = 1e4,
        T_m_start: float = 300.0,
        **kwargs,
    ) -> SolverResult:
        """
        Solve for a closed-open resonator (U1=0 at start, p1=0 at end).

        This is a convenience method for the common case of a quarter-wave
        resonator.

        Parameters
        ----------
        p1_amplitude : float, optional
            Initial guess for pressure amplitude in Pa.
        T_m_start : float, optional
            Mean temperature at start in K.
        **kwargs
            Additional arguments passed to solve().

        Returns
        -------
        SolverResult
            Solution for the resonator.
        """
        return self.solve(
            p1_amplitude=p1_amplitude,
            T_m_start=T_m_start,
            targets={"p1_end": 0},
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Network(gas={self.gas.name}, "
            f"n_segments={len(self._segments)}, "
            f"total_length={self.total_length:.4f} m)"
        )
