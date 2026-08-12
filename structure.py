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
    d: bool
    e: bool

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
    d: bool
    e: bool

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
    encoded_circuit: QuantumCircuit

    segments: dict[int, WireSegement] = field(default_factory=dict)
    gates: list[GateRecord] = field(default_factory=list)
    current_segment: dict[int, int] = field(default_factory=dict)
    teleportation: dict[int, list[TeleportationRecord]] = field(default_factory=dict)
    injection: dict[int, InjectionRecord] = field(default_factory=dict)

class SegmentFactory:
    def __init__(self, kappa: int, start_physical_qubits: int = 0) -> None:
        self.next_segment_id = 0
        self.next_physical_qubit = start_physical_qubits
        self.kappa = kappa

    def create(self, circuit: QuantumCircuit, logical_qubit: int) -> WireSegement:
        segment_id = self.next_segment_id
        seg = WireSegement(
            segment_id,
            logical_qubit,
            QuantumRegister(2, name = "epr{}".format(segment_id)),
            QuantumRegister(self.kappa, name = "z{}".format(segment_id)),
            [],
            QuantumRegister(self.kappa, name = "x{}".format(segment_id))
        )
        circuit.add_register(seg.epr)
        circuit.add_register(seg.z)
        for i in range(self.kappa + 1):
            seg.b.append(AncillaRegister(self.kappa + 1, name="b{}-{}".format(self.next_segment_id, i)))
            circuit.add_register(seg.b[i])
        circuit.add_register(seg.x)
        self.next_physical_qubit += 2 + (self.kappa + 1) ** 2 + 2 * self.kappa
        self.next_segment_id += 1
        return seg