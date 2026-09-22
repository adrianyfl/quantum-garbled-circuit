"""Records and registers: what Enc writes down and what Dec is handed."""
from dataclasses import dataclass, field

from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction


@dataclass
class Correction:
    """A PX-group element X^x Z^z P^p, the R of Equation (6.1)."""
    x: bool
    z: bool
    p: bool


@dataclass
class WireSegement:
    """One wire w: its EPR pair and its ancilla register a^w = (z, x, b)."""
    segment_id: int
    logical_qubit: int
    epr: QuantumRegister                # (e^w_1, e^w_2)
    z: QuantumRegister                  # holds l_{z,d}
    b: list[AncillaRegister]            # (kappa+1)^2, the Proposition 6.3 ancillas
    x: QuantumRegister                  # holds l_{x,e}


@dataclass
class TeleportationRecord:
    """Randomness for one gate output wire: labels, and the s, t bits."""
    from_id: int
    to_id: int
    l_x: tuple[int, ...]
    l_z: tuple[int, ...]
    s_x: bool
    s_z: bool
    t_x: bool
    t_z: bool


@dataclass
class InjectionRecord:
    """The same, for an input wire (Protocol 5)."""
    from_qubit: int
    to_id: int
    l_x: tuple[int, ...]
    l_z: tuple[int, ...]
    s_x: bool
    s_z: bool
    t_x: bool
    t_z: bool


@dataclass
class GateRecord:
    gate_id: int
    operation: Instruction
    logical_qubit: tuple[int, ...]
    input_segments: tuple[int, ...]
    output_segments: tuple[int, ...]


@dataclass
class WireParams:
    """Everything the correction function needs about one output wire."""
    A: QuantumCircuit
    l_z: tuple
    l_x: tuple
    s_x: bool
    s_z: bool
    t_x: bool
    t_z: bool
    kappa: int


@dataclass
class QGC:
    """The encoding, the decoder, and the bookkeeping that produced them.

    encoded_circuit and decoder are separate circuits over one shared register
    set; decode_output composes them. Dec never reads the records -- it takes
    its (d, e) bits from the d^w dictionaries in output_registers.
    """
    original_circuit: QuantumCircuit
    encoded_circuit: QuantumCircuit          # Enc  -- Protocols 4 and 5
    decoder: QuantumCircuit | None = None    # Dec  -- Protocols 6 and 7

    segments: dict[int, WireSegement] = field(default_factory=dict)
    gates: list[GateRecord] = field(default_factory=list)
    current_segment: dict[int, int] = field(default_factory=dict)
    teleportation: dict[int, list[TeleportationRecord]] = field(default_factory=dict)
    injection: dict[int, InjectionRecord] = field(default_factory=dict)
    segment_record: dict[int, object] = field(default_factory=dict)

    cre: dict = field(default_factory=dict)          # gate_id -> GateCRE
    registers: dict = field(default_factory=dict)    # gate_id -> cg, desc, anc, blk
    # segment_id -> dict, scratch, bits  for each output wire w in O \ T
    output_registers: dict = field(default_factory=dict)

    traced_out: frozenset = field(default_factory=frozenset)       # the set T
    classical_inputs: frozenset = field(default_factory=frozenset)
    gate_set: dict | None = None    # what to_gate_set did to the input circuit


class SegmentFactory:
    def __init__(self, kappa: int) -> None:
        self.next_segment_id = 0
        self.kappa = kappa

    def create(self, circuits, logical_qubit: int) -> WireSegement:
        """Create a wire segment and declare its registers on every circuit.

        Enc and Dec are separate circuits over the same registers, so both must
        receive them in the same order for the two to compose.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        segment_id = self.next_segment_id
        seg = WireSegement(
            segment_id,
            logical_qubit,
            QuantumRegister(2, name="epr{}".format(segment_id)),
            QuantumRegister(self.kappa, name="z{}".format(segment_id)),
            [AncillaRegister(self.kappa + 1, name="b{}-{}".format(segment_id, i))
             for i in range(self.kappa + 1)],
            QuantumRegister(self.kappa, name="x{}".format(segment_id)),
        )
        for circuit in circuits:
            circuit.add_register(seg.epr, seg.z, *seg.b, seg.x)
        self.next_segment_id += 1
        return seg
