"""Shooting solver for TBRANCH/UNION loop networks.

This module provides a solver for acoustic networks with TBRANCH/UNION
topology, as used in reference baseline's lrc1.out and similar examples.

The solver iterates on:
1. Input volumetric velocity U1 (magnitude and phase)
2. Branch impedance Zb (real and imaginary parts)

To satisfy:
1. Pressure continuity at UNION (branch and trunk pressures match)
2. Boundary condition at HARDEND (U1 = 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root, OptimizeResult

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


@dataclass
class TBranchLoopResult:
    """
    Results from the TBRANCH/UNION loop solver.

    Attributes
    ----------
    converged : bool
        Whether the solver converged.
    message : str
        Status message from the solver.
    n_iterations : int
        Number of iterations taken.
    residual_norm : float
        Final residual norm.
    U1_magnitude : float
        Solved input volumetric velocity magnitude (m³/s).
    U1_phase : float
        Solved input volumetric velocity phase (degrees).
    Zb_real : float
        Solved branch impedance real part (Pa·s/m³).
    Zb_imag : float
        Solved branch impedance imaginary part (Pa·s/m³).
    p1_input : complex
        Input pressure (Pa).
    U1_input : complex
        Solved input volumetric velocity (m³/s).
    Zb : complex
        Solved branch impedance (Pa·s/m³).
    p1_union : complex
        Pressure at UNION (Pa).
    U1_hardend : complex
        Velocity at HARDEND (m³/s) - should be ~0.
    pressure_mismatch : float
        Pressure mismatch at UNION (Pa).
    """

    converged: bool
    message: str
    n_iterations: int
    residual_norm: float
    U1_magnitude: float
    U1_phase: float
    Zb_real: float
    Zb_imag: float
    p1_input: complex
    U1_input: complex
    Zb: complex
    p1_union: complex
    U1_hardend: complex
    pressure_mismatch: float

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"TBranchLoopResult({status}, "
            f"|U1|={self.U1_magnitude:.4e} m³/s, "
            f"Zb={self.Zb_real:.2e}+{self.Zb_imag:.2e}j Pa·s/m³)"
        )


class TBranchLoopSolver:
    """
    Shooting solver for TBRANCH/UNION loop networks.

    This solver handles the topology:
        INPUT → TBRANCH → [TRUNK path] → UNION → HARDEND
                    ↓                      ↑
                [BRANCH path] ────────────┘

    The solver iterates on [|U1|, Ph(U1), Re(Zb), Im(Zb)] to satisfy
    the 4 conditions: Re(p_mismatch)=0, Im(p_mismatch)=0,
    Re(U1_hardend)=0, Im(U1_hardend)=0.

    Parameters
    ----------
    propagate_func : Callable
        Function that propagates through the network given (p1, U1, Zb)
        and returns (p1_union, U1_union, pressure_mismatch).
    gas : Gas
        Gas properties object.
    omega : float
        Angular frequency (rad/s).
    T_m : float
        Mean temperature (K).
    p1_input : complex
        Fixed input pressure amplitude (Pa).

    Examples
    --------
    >>> def propagate(p1, U1, Zb):
    ...     # Your network propagation logic
    ...     return p1_union, U1_union, pressure_mismatch
    >>> solver = TBranchLoopSolver(propagate, gas, omega, T_m, p1_input=2000)
    >>> result = solver.solve(
    ...     U1_mag_guess=6e-3,
    ...     U1_phase_guess=80.0,
    ...     Zb_real_guess=3.7e5,
    ...     Zb_imag_guess=-2.7e5,
    ... )
    """

    def __init__(
        self,
        propagate_func: Callable[
            [complex, complex, complex],
            tuple[complex, complex, float],
        ],
        gas: Gas,
        omega: float,
        T_m: float,
        p1_input: complex,
    ) -> None:
        self.propagate = propagate_func
        self.gas = gas
        self.omega = omega
        self.T_m = T_m
        self.p1_input = p1_input

    def solve(
        self,
        U1_mag_guess: float,
        U1_phase_guess: float,
        Zb_real_guess: float,
        Zb_imag_guess: float,
        method: str = "hybr",
        tol: float = 1e-8,
        maxiter: int = 100,
        verbose: bool = False,
    ) -> TBranchLoopResult:
        """
        Solve for loop closure.

        Parameters
        ----------
        U1_mag_guess : float
            Initial guess for input |U1| (m³/s).
        U1_phase_guess : float
            Initial guess for input Ph(U1) (degrees).
        Zb_real_guess : float
            Initial guess for Re(Zb) (Pa·s/m³).
        Zb_imag_guess : float
            Initial guess for Im(Zb) (Pa·s/m³).
        method : str, optional
            scipy.optimize.root method, by default "hybr".
        tol : float, optional
            Convergence tolerance, by default 1e-8.
        maxiter : int, optional
            Maximum iterations, by default 100.
        verbose : bool, optional
            Print progress, by default False.

        Returns
        -------
        TBranchLoopResult
            Solution results.
        """
        iteration_count = [0]

        # Scale factors for better conditioning
        # U1 magnitudes are ~1e-3, Zb values are ~1e5
        U1_scale = max(U1_mag_guess, 1e-6)
        Zb_scale = max(abs(Zb_real_guess), abs(Zb_imag_guess), 1e3)

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute residuals for the shooting method."""
            iteration_count[0] += 1

            # Unpack scaled variables
            U1_mag = x[0] * U1_scale
            U1_phase_deg = x[1]
            Zb_real = x[2] * Zb_scale
            Zb_imag = x[3] * Zb_scale

            # Build complex values
            U1 = U1_mag * np.exp(1j * np.radians(U1_phase_deg))
            Zb = complex(Zb_real, Zb_imag)

            # Propagate through network
            try:
                p1_union, U1_union, p_mismatch = self.propagate(
                    self.p1_input, U1, Zb
                )
            except Exception as e:
                if verbose:
                    print(f"  Propagation failed: {e}")
                return np.array([1e10, 1e10, 1e10, 1e10])

            # Residuals:
            # 1-2: Pressure mismatch at UNION (real and imag)
            # 3-4: U1 at HARDEND should be 0 (real and imag)
            residuals = np.array([
                p_mismatch.real / 1000,  # Scale pressure to ~1
                p_mismatch.imag / 1000,
                U1_union.real / U1_scale,  # Scale to ~1
                U1_union.imag / U1_scale,
            ])

            if verbose:
                print(
                    f"  Iter {iteration_count[0]}: "
                    f"|U1|={U1_mag:.4e}, Ph={U1_phase_deg:.1f}°, "
                    f"Zb={Zb_real:.2e}+{Zb_imag:.2e}j, "
                    f"residual={np.linalg.norm(residuals):.2e}"
                )

            return residuals

        # Initial guess (scaled)
        x0 = np.array([
            U1_mag_guess / U1_scale,
            U1_phase_guess,
            Zb_real_guess / Zb_scale,
            Zb_imag_guess / Zb_scale,
        ])

        if verbose:
            print("Starting TBRANCH/UNION loop solver...")
            print(f"  Initial: |U1|={U1_mag_guess:.4e}, Ph={U1_phase_guess:.1f}°")
            print(f"  Initial: Zb={Zb_real_guess:.2e}+{Zb_imag_guess:.2e}j")

        # Solve - try multiple methods if needed
        result = root(
            residual,
            x0,
            method=method,
            tol=tol,
            options={"maxfev": maxiter * 10},
        )

        # If hybr fails, try lm (Levenberg-Marquardt)
        if not result.success and method == "hybr":
            if verbose:
                print("  hybr failed, trying Levenberg-Marquardt...")
            iteration_count[0] = 0
            result = root(
                residual,
                x0,
                method="lm",
                tol=tol,
                options={"maxiter": maxiter * 10},
            )

        # Extract final values (unscale)
        U1_mag_final = result.x[0] * U1_scale
        U1_phase_final = result.x[1]
        Zb_real_final = result.x[2] * Zb_scale
        Zb_imag_final = result.x[3] * Zb_scale

        U1_final = U1_mag_final * np.exp(1j * np.radians(U1_phase_final))
        Zb_final = complex(Zb_real_final, Zb_imag_final)

        # Get final state
        p1_union, U1_hardend, p_mismatch = self.propagate(
            self.p1_input, U1_final, Zb_final
        )

        return TBranchLoopResult(
            converged=result.success,
            message=result.message,
            n_iterations=iteration_count[0],
            residual_norm=float(np.linalg.norm(result.fun)),
            U1_magnitude=U1_mag_final,
            U1_phase=U1_phase_final,
            Zb_real=Zb_real_final,
            Zb_imag=Zb_imag_final,
            p1_input=self.p1_input,
            U1_input=U1_final,
            Zb=Zb_final,
            p1_union=p1_union,
            U1_hardend=U1_hardend,
            pressure_mismatch=abs(p_mismatch),
        )


def solve_lrc1_loop(
    gas: Gas,
    omega: float,
    T_m: float,
    p1_input: complex,
    Z_inertance: complex,
    C_compliance: float,
    Z_resistance: complex,
    U1_mag_guess: float = 6e-3,
    U1_phase_guess: float = 80.0,
    Zb_real_guess: float = 3.7e5,
    Zb_imag_guess: float = -2.7e5,
    verbose: bool = False,
) -> TBranchLoopResult:
    """
    Solve the lrc1.out loop network.

    This is a convenience function for solving the specific lrc1 topology:

        BEGIN(p1) → TBRANCH
                       ↓
            [BRANCH]: IMPEDANCE(L) → COMPLIANCE → SOFTEND
                       ↓
            [TRUNK]:  IMPEDANCE(R) → UNION ← (branch rejoins)
                                       ↓
                                    HARDEND (U1=0)

    Parameters
    ----------
    gas : Gas
        Gas properties.
    omega : float
        Angular frequency (rad/s).
    T_m : float
        Mean temperature (K).
    p1_input : complex
        Input pressure amplitude (Pa).
    Z_inertance : complex
        Inertance impedance (Pa·s/m³), typically j*omega*L.
    C_compliance : float
        Acoustic compliance (m³/Pa).
    Z_resistance : complex
        Resistance impedance (Pa·s/m³), typically R.
    U1_mag_guess : float
        Initial guess for |U1| (m³/s).
    U1_phase_guess : float
        Initial guess for Ph(U1) (degrees).
    Zb_real_guess : float
        Initial guess for Re(Zb) (Pa·s/m³).
    Zb_imag_guess : float
        Initial guess for Im(Zb) (Pa·s/m³).
    verbose : bool
        Print progress.

    Returns
    -------
    TBranchLoopResult
        Solution results.
    """
    # Compliance admittance
    Y_compliance = 1j * omega * C_compliance

    def propagate(p1: complex, U1: complex, Zb: complex) -> tuple[complex, complex, complex]:
        """Propagate through lrc1 network."""
        # TBRANCH: Split flow based on Zb
        # U1_branch = p1 / Zb (branch takes flow based on its impedance)
        U1_branch = p1 / Zb
        U1_trunk = U1 - U1_branch

        # === BRANCH PATH ===
        # IMPEDANCE (inertance): p drops, U unchanged
        p1_after_L = p1 - Z_inertance * U1_branch

        # COMPLIANCE: p unchanged, U reduced (some flows into capacitor)
        p1_after_C = p1_after_L
        U1_after_C = U1_branch - Y_compliance * p1_after_L

        # SOFTEND: just marks end, no change
        p1_branch_end = p1_after_C
        U1_branch_end = U1_after_C

        # === TRUNK PATH ===
        # IMPEDANCE (resistance): p drops, U unchanged
        p1_after_R = p1 - Z_resistance * U1_trunk
        U1_after_R = U1_trunk

        # === UNION ===
        # Pressure should match, velocities add
        p1_union = p1_after_R  # Trunk determines pressure
        U1_union = U1_after_R + U1_branch_end  # Velocities combine

        # Pressure mismatch
        p_mismatch = p1_branch_end - p1_after_R

        # === HARDEND ===
        # U1 should be 0
        U1_hardend = U1_union

        return p1_union, U1_hardend, p_mismatch

    solver = TBranchLoopSolver(propagate, gas, omega, T_m, p1_input)

    return solver.solve(
        U1_mag_guess=U1_mag_guess,
        U1_phase_guess=U1_phase_guess,
        Zb_real_guess=Zb_real_guess,
        Zb_imag_guess=Zb_imag_guess,
        verbose=verbose,
    )
