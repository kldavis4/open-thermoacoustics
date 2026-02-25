"""Visualization utilities for thermoacoustic results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.solver.network import SegmentResult
from openthermoacoustics.solver.shooting import SolverResult


@dataclass(frozen=True)
class SegmentBoundary:
    """Boundary metadata for plotting segment transitions."""

    position: float
    label: str


def plot_profiles(
    result: SolverResult,
    *,
    segment_results: Sequence[SegmentResult] | None = None,
    units: str = "mm",
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 8.0),
    include_segment_labels: bool = False,
) -> tuple[object, NDArray]:
    """
    Plot magnitude profiles for pressure, velocity, power, and temperature.

    Parameters
    ----------
    result : SolverResult
        Solver result containing global profiles.
    segment_results : Sequence[SegmentResult], optional
        Segment results for boundary overlays. If provided, vertical lines
        are drawn at segment boundaries.
    units : str, optional
        Position units for x-axis. Supported: "m", "cm", "mm". Default "mm".
    show : bool, optional
        Whether to display the figure.
    save_path : str | Path, optional
        If provided, save the figure to this path (PNG recommended).
    dpi : int, optional
        DPI for saved figures.
    figsize : tuple[float, float], optional
        Figure size in inches.
    include_segment_labels : bool, optional
        If True, label segment boundaries by segment class name.

    Returns
    -------
    tuple[object, NDArray]
        Matplotlib figure and axes array.
    """
    fig, axes = _subplots(2, 2, figsize=figsize)
    x = _scale_positions(result.x_profile, units)
    boundary_lines = _build_boundaries(segment_results, units)

    _plot_with_boundaries(
        axes[0, 0],
        x,
        np.abs(result.p1_profile),
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[0, 0].set_ylabel("|p1| (Pa)")
    axes[0, 0].set_title("Pressure Amplitude")

    _plot_with_boundaries(
        axes[0, 1],
        x,
        np.abs(result.U1_profile) * 1e6,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[0, 1].set_ylabel("|U1| (mm^3/s)")
    axes[0, 1].set_title("Volumetric Velocity Amplitude")

    _plot_with_boundaries(
        axes[1, 0],
        x,
        result.acoustic_power,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[1, 0].set_ylabel("Acoustic Power (W)")
    axes[1, 0].set_title("Time-Averaged Acoustic Power")

    _plot_with_boundaries(
        axes[1, 1],
        x,
        result.T_m_profile,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[1, 1].set_ylabel("Temperature (K)")
    axes[1, 1].set_title("Mean Temperature")

    for ax in axes.flat:
        ax.set_xlabel(f"Position ({units})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Acoustic Profiles at f = {result.frequency:.2f} Hz", fontsize=12
    )
    fig.tight_layout()

    _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi)
    return fig, axes


def plot_phasor_profiles(
    result: SolverResult,
    *,
    segment_results: Sequence[SegmentResult] | None = None,
    units: str = "mm",
    phase_units: str = "deg",
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 8.0),
    include_segment_labels: bool = False,
) -> tuple[object, NDArray]:
    """
    Plot magnitude and phase for pressure and velocity.

    Parameters
    ----------
    result : SolverResult
        Solver result containing global profiles.
    segment_results : Sequence[SegmentResult], optional
        Segment results for boundary overlays.
    units : str, optional
        Position units for x-axis. Supported: "m", "cm", "mm". Default "mm".
    phase_units : str, optional
        Phase units: "deg" or "rad".
    show : bool, optional
        Whether to display the figure.
    save_path : str | Path, optional
        If provided, save the figure to this path (PNG recommended).
    dpi : int, optional
        DPI for saved figures.
    figsize : tuple[float, float], optional
        Figure size in inches.
    include_segment_labels : bool, optional
        If True, label segment boundaries by segment class name.

    Returns
    -------
    tuple[object, NDArray]
        Matplotlib figure and axes array.
    """
    fig, axes = _subplots(2, 2, figsize=figsize)
    x = _scale_positions(result.x_profile, units)
    boundary_lines = _build_boundaries(segment_results, units)
    phase_scale = 180.0 / np.pi if phase_units == "deg" else 1.0

    _plot_with_boundaries(
        axes[0, 0],
        x,
        np.abs(result.p1_profile),
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[0, 0].set_ylabel("|p1| (Pa)")
    axes[0, 0].set_title("Pressure Magnitude")

    _plot_with_boundaries(
        axes[0, 1],
        x,
        np.angle(result.p1_profile) * phase_scale,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[0, 1].set_ylabel(f"∠p1 ({phase_units})")
    axes[0, 1].set_title("Pressure Phase")

    _plot_with_boundaries(
        axes[1, 0],
        x,
        np.abs(result.U1_profile) * 1e6,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[1, 0].set_ylabel("|U1| (mm^3/s)")
    axes[1, 0].set_title("Velocity Magnitude")

    _plot_with_boundaries(
        axes[1, 1],
        x,
        np.angle(result.U1_profile) * phase_scale,
        boundary_lines,
        include_segment_labels=include_segment_labels,
    )
    axes[1, 1].set_ylabel(f"∠U1 ({phase_units})")
    axes[1, 1].set_title("Velocity Phase")

    for ax in axes.flat:
        ax.set_xlabel(f"Position ({units})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Phasor Profiles at f = {result.frequency:.2f} Hz", fontsize=12
    )
    fig.tight_layout()

    _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi)
    return fig, axes


def plot_frequency_sweep(
    frequencies: Iterable[float],
    values: Iterable[float],
    *,
    ylabel: str,
    xlabel: str = "Frequency (Hz)",
    title: str = "Frequency Sweep",
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> tuple[object, object]:
    """
    Plot a generic frequency sweep (e.g., residual norm vs frequency).

    Parameters
    ----------
    frequencies : Iterable[float]
        Sweep frequency values (Hz).
    values : Iterable[float]
        Values to plot against frequency.
    ylabel : str
        Label for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.
    show : bool, optional
        Whether to display the figure.
    save_path : str | Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        DPI for saved figures.
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    tuple[object, object]
        Matplotlib figure and axis.
    """
    fig, ax = _subplots(1, 1, figsize=figsize)
    ax = np.ravel(ax)[0]
    ax.plot(list(frequencies), list(values), "b-", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi)
    return fig, ax


def _build_boundaries(
    segment_results: Sequence[SegmentResult] | None,
    units: str,
) -> list[SegmentBoundary]:
    if not segment_results:
        return []
    boundaries: list[SegmentBoundary] = []
    for result in segment_results:
        position = result.x_global[-1]
        label = result.segment.__class__.__name__
        boundaries.append(SegmentBoundary(position=position, label=label))
    return [
        SegmentBoundary(
            position=_scale_positions(np.array([b.position]), units)[0],
            label=b.label,
        )
        for b in boundaries
    ]


def _plot_with_boundaries(
    ax: object,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    boundaries: Sequence[SegmentBoundary],
    *,
    include_segment_labels: bool,
) -> None:
    ax.plot(x, y, linewidth=1.5)
    for boundary in boundaries:
        ax.axvline(boundary.position, color="k", alpha=0.15, linewidth=1.0)
        if include_segment_labels:
            ax.text(
                boundary.position,
                0.98,
                boundary.label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
                color="0.3",
            )


def _scale_positions(
    x: NDArray[np.float64],
    units: str,
) -> NDArray[np.float64]:
    if units == "m":
        return x
    if units == "cm":
        return x * 100.0
    if units == "mm":
        return x * 1000.0
    raise ValueError(f"Unsupported units '{units}'. Use 'm', 'cm', or 'mm'.")


def _subplots(rows: int, cols: int, *, figsize: tuple[float, float]):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install openthermoacoustics[viz]"
        ) from exc
    return plt.subplots(rows, cols, figsize=figsize)


def _finalize_figure(
    fig: object,
    *,
    show: bool,
    save_path: str | Path | None,
    dpi: int,
) -> None:
    if save_path is not None:
        path = Path(save_path)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
