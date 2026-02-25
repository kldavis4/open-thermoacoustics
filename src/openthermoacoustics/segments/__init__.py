"""Acoustic network segment modules."""

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.segments.duct import Duct
from openthermoacoustics.segments.cone import Cone
from openthermoacoustics.segments.stack import Stack
from openthermoacoustics.segments.stack_energy import StackEnergy
from openthermoacoustics.segments.heat_exchanger import HeatExchanger
from openthermoacoustics.segments.compliance import Compliance
from openthermoacoustics.segments.inertance import Inertance
from openthermoacoustics.segments.boundary import HardEnd, SoftEnd
from openthermoacoustics.segments.transducer import Transducer

# reference baseline aliases for transducers
IESPEAKER = Transducer  # Current-driven enclosed speaker (use propagate_driven)
VESPEAKER = Transducer  # Voltage-driven enclosed speaker (use propagate_voltage_driven)
from openthermoacoustics.segments.open_end import OpenEnd
from openthermoacoustics.segments.join import Join, JOIN
from openthermoacoustics.segments.impedance import Impedance
from openthermoacoustics.segments.branch import TBranch, Return, SideBranch
from openthermoacoustics.segments.tbranch import TBranchImpedance, Union, SoftEndWithState
from openthermoacoustics.segments.stkscreen import StackScreen
from openthermoacoustics.segments.sx import ScreenHeatExchanger, SX
from openthermoacoustics.segments.minor import Minor, MINOR
from openthermoacoustics.segments.stkduct import StackDuct, STKDUCT
from openthermoacoustics.segments.stkcone import StackCone, STKCONE
from openthermoacoustics.segments.tx import TubeHeatExchanger, TX
from openthermoacoustics.segments.surface import Surface, SURFACE
from openthermoacoustics.segments.impedance_branch import ImpedanceBranch, BRANCH
from openthermoacoustics.segments.radiation_branch import (
    OpenBranch,
    OPNBRANCH,
    PistonBranch,
    PISTBRANCH,
)
from openthermoacoustics.segments.side_branch_transducer import (
    SideBranchTransducer,
    IDUCER,
    VDUCER,
    SideBranchSpeaker,
    ISPEAKER,
    VSPEAKER,
)
from openthermoacoustics.segments.enclosed_transducer import (
    EnclosedTransducer,
    IEDUCER,
    VEDUCER,
)
from openthermoacoustics.segments.stkpowerlw import StackPowerLaw, STKPOWERLW
from openthermoacoustics.segments.px import PowerLawHeatExchanger, PX
from openthermoacoustics.segments.anchor import Anchor, ANCHOR, Insulate, INSULATE, ThermalMode
from openthermoacoustics.segments.vxt import (
    VariableTemperatureHeatExchanger,
    VXT1,
    VariableTemperatureHeatExchanger2Pass,
    VXT2,
    VariableHeatFluxHeatExchanger,
    VXQ1,
    VariableHeatFluxHeatExchanger2Pass,
    VXQ2,
)

__all__ = [
    "Segment",
    "Duct",
    "Cone",
    "Stack",
    "StackEnergy",
    "HeatExchanger",
    "Compliance",
    "Inertance",
    "HardEnd",
    "SoftEnd",
    "Transducer",
    "IESPEAKER",
    "VESPEAKER",
    "OpenEnd",
    "Join",
    "JOIN",
    "Impedance",
    "TBranch",
    "Return",
    "SideBranch",
    # reference baseline-compatible loop segments
    "TBranchImpedance",
    "Union",
    "SoftEndWithState",
    # reference baseline-compatible stack/regenerator segments
    "StackScreen",
    # reference baseline-compatible heat exchanger segments
    "ScreenHeatExchanger",
    "SX",
    # reference baseline-compatible minor loss segments
    "Minor",
    "MINOR",
    # reference baseline-compatible pulse tube segments
    "StackDuct",
    "STKDUCT",
    "StackCone",
    "STKCONE",
    # reference baseline-compatible tube heat exchanger segments
    "TubeHeatExchanger",
    "TX",
    # reference baseline-compatible surface segments
    "Surface",
    "SURFACE",
    # reference baseline-compatible impedance branch segments
    "ImpedanceBranch",
    "BRANCH",
    # reference baseline-compatible radiation branch segments
    "OpenBranch",
    "OPNBRANCH",
    "PistonBranch",
    "PISTBRANCH",
    # reference baseline-compatible side-branch transducer segments
    "SideBranchTransducer",
    "IDUCER",
    "VDUCER",
    "SideBranchSpeaker",
    "ISPEAKER",
    "VSPEAKER",
    # reference baseline-compatible enclosed transducer segments (generic)
    "EnclosedTransducer",
    "IEDUCER",
    "VEDUCER",
    # reference baseline-compatible power-law regenerator segments
    "StackPowerLaw",
    "STKPOWERLW",
    # reference baseline-compatible power-law heat exchanger segments
    "PowerLawHeatExchanger",
    "PX",
    # reference baseline-compatible thermal mode control segments
    "Anchor",
    "ANCHOR",
    "Insulate",
    "INSULATE",
    "ThermalMode",
    # reference baseline-compatible variable temperature heat exchangers
    "VariableTemperatureHeatExchanger",
    "VXT1",
    "VariableTemperatureHeatExchanger2Pass",
    "VXT2",
    # reference baseline-compatible variable heat flux heat exchangers
    "VariableHeatFluxHeatExchanger",
    "VXQ1",
    "VariableHeatFluxHeatExchanger2Pass",
    "VXQ2",
]
