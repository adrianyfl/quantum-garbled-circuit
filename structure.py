from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from dataclasses import dataclass, field
from qiskit.circuit import Instruction

@dataclass
class Correction:
    x: bool
    z: bool
    p: bool

@dataclass
class WireSegement:
    segment_id: int
    logical_qubit: int
    # physical qubits
    epr: QuantumRegister
    # ancillas
    z: QuantumRegister
    b: list[AncillaRegister]
    x: QuantumRegister

@dataclass
class TeleportationRecord:
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
class QGC:
    original_circuit: QuantumCircuit
    encoded_circuit: QuantumCircuit          # Enc  -- Protocols 4 and 5
    decoder: QuantumCircuit | None = None    # Dec  -- Protocols 6 and 7, same registers

    segments: dict[int, WireSegement] = field(default_factory=dict)
    gates: list[GateRecord] = field(default_factory=list)
    current_segment: dict[int, int] = field(default_factory=dict)
    teleportation: dict[int, list[TeleportationRecord]] = field(default_factory=dict)
    injection: dict[int, InjectionRecord] = field(default_factory=dict)
    segment_record: dict[int, object] = field(default_factory=dict)
    cre: dict = field(default_factory=dict)          # gate_id -> GateCRE (CRE mode only)
    registers: dict = field(default_factory=dict)    # gate_id -> {'cg','desc','anc'} (CRE mode)
    # segment_id -> {'dict','scratch','bits'} for each output wire w in O \ T.
    # 'dict' is the paper's d^w label dictionary (4*kappa qubits).
    output_registers: dict = field(default_factory=dict)

@dataclass
class WireParams:
    A: QuantumCircuit
    l_z: tuple
    l_x: tuple
    s_x: bool
    s_z: bool
    t_x: bool
    t_z: bool
    kappa:int

class SegmentFactory:
    def __init__(self, kappa: int, start_physical_qubits: int = 0) -> None:
        self.next_segment_id = 0
        self.next_physical_qubit = start_physical_qubits
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
            QuantumRegister(2, name = "epr{}".format(segment_id)),
            QuantumRegister(self.kappa, name = "z{}".format(segment_id)),
            [AncillaRegister(self.kappa + 1, name="b{}-{}".format(segment_id, i))
             for i in range(self.kappa + 1)],
            QuantumRegister(self.kappa, name = "x{}".format(segment_id))
        )
        for circuit in circuits:
            circuit.add_register(seg.epr, seg.z, *seg.b, seg.x)
        self.next_physical_qubit += 2 + (self.kappa + 1) ** 2 + 2 * self.kappa
        self.next_segment_id += 1
        return seg