from qiskit import QuantumCircuit
from dataclasses import dataclass, field

from qiskit.circuit import Instruction


@dataclass
class WireSegement:
    segment_id: int
    logical_qubit: int
    #physical qubits
    epr_left: int
    epr_right: int

@dataclass
class GateRecord:
    gate_id: int
    operation: Instruction
    logical_qubit: tuple[int, ...]
    input_segments: tuple[int, ...]
    output_segments: tuple[int, ...]

@dataclass
class EPRWireCircuit:
    original_circuit: QuantumCircuit
    encoded_circuit: QuantumCircuit

    segments: dict[int, WireSegement] = field(default_factory=dict)
    gates: list[GateRecord] = field(default_factory=list)
    current_segment: dict[int, int] = field(default_factory=dict)

class SegmentFactory:
    def __init__(self, start_physical_qubits: int = 0) -> None:
        self.next_segment_id = 0
        self.next_physical_qubit = start_physical_qubits

    def create(self, logical_qubit: int) -> WireSegement:
        segment_id = self.next_segment_id
        left = self.next_physical_qubit
        right = left + 1
        self.next_segment_id += 1
        self.next_physical_qubit += 2
        return WireSegement(segment_id, logical_qubit, left, right)

# create EPR pair between q1 and q2
def prep_epr_pair(circuit: QuantumCircuit, q1, q2):
    circuit.h(q1)
    circuit.cx(q1, q2)

# teleportation between two wires
def teleport(circuit: QuantumCircuit, segment1: WireSegement, segment2: WireSegement):
    circuit.cx(segment1.epr_right, segment2.epr_left)
    circuit.h(segment1.epr_right)
    circuit.cx(segment2.epr_left, segment2.epr_right)
    circuit.cz(segment1.epr_right, segment2.epr_right)

# input injection
def inject_input(circuit: QuantumCircuit, input_qubit: int, initial_segment: WireSegement):
    # bell pair
    circuit.cx(input_qubit, initial_segment.epr_left)
    circuit.h(input_qubit)

    # deferred corrections
    circuit.cx(initial_segment.epr_left, initial_segment.epr_right)
    circuit.cz(input_qubit, initial_segment.epr_left)

# convert circuit to use epr pair per wire
def convert_wire_to_epr(circuit: QuantumCircuit, input_prep: QuantumCircuit | None = None) -> QuantumCircuit:
    input_qubits = tuple(range(circuit.num_qubits))
    factory = SegmentFactory(start_physical_qubits=circuit.num_qubits)
    segments:dict[int, WireSegement] = {}
    initial_segment: dict[int, int] = {}
    current_segment: dict[int, int] = {}
    gate_records: list[GateRecord] = []

    # initialize input segments
    for q in range(circuit.num_qubits):
        segment = factory.create(q)
        segments[segment.segment_id] = segment
        initial_segment[q] = segment.segment_id
        current_segment[q] = segment.segment_id

    # process gates
    for gate_id, instruction in enumerate(circuit.data):
        operation = instruction.operation
        logical_qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        input_segment_ids = tuple(current_segment[q] for q in logical_qubits)
        output_segment_ids = []
        for q in logical_qubits:
            segment = factory.create(q)
            segments[segment.segment_id] = segment
            output_segment_ids.append(segment.segment_id)
        gate_records.append(GateRecord(gate_id, operation, logical_qubits, input_segment_ids, tuple(output_segment_ids)))
        for q, segment_id in zip(logical_qubits, output_segment_ids):
            current_segment[q] = segment_id

    converted_circuit = QuantumCircuit(factory.next_physical_qubit)

    # compose with input prep circuit
    if input_prep is not None:
        converted_circuit.compose(input_prep, qubits=input_qubits, inplace=True)

    # prepare all epr paris
    for s_id, s in segments.items():
        prep_epr_pair(converted_circuit, s.epr_left, s.epr_right)

    # inject original input
    for q in range(circuit.num_qubits):
        segment_id = initial_segment[q]
        inject_input(converted_circuit, input_qubits[q], segments[segment_id])

    # add gate + teleportation
    for g in gate_records:
        converted_circuit.append(g.operation, [segments[s_id].epr_right for s_id in g.input_segments])
        for i in range(len(g.input_segments)):
            teleport(converted_circuit, segments[g.input_segments[i]], segments[g.output_segments[i]])

    return converted_circuit