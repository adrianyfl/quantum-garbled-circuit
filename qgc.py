from qiskit import AncillaRegister
from qiskit.quantum_info import Statevector

from correction import propagate_correction
from structure import *
from helper import *
from encoder import sample_a, construct_lambda1
from decoder import construct_lambda2, construct_lambda3

# create EPR pair between q1 and q2
def prep_epr_pair(circuit: QuantumCircuit, q1, q2):
    circuit.h(q1)
    circuit.cx(q1, q2)

# convert circuit to use epr pair per wire
def garble_circuit(circuit: QuantumCircuit, kappa: int, input_prep: QuantumCircuit | None = None) -> QGC:
    input_qubits = tuple(range(circuit.num_qubits))
    factory = SegmentFactory(kappa, start_physical_qubits=circuit.num_qubits)
    segments:dict[int, WireSegement] = {}
    initial_segment: dict[int, int] = {}
    current_segment: dict[int, int] = {}
    gate_records: list[GateRecord] = []
    injection_records: dict[int, InjectionRecord] = {}
    teleportation_records: dict[int, list[TeleportationRecord]] = {}
    converted_circuit = QuantumCircuit(circuit.num_qubits)
    pauli_frame: dict[int, tuple[bool, bool]] = {}

    # initialize input segments
    for q in range(circuit.num_qubits):
        segment = factory.create(converted_circuit, q)
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
            segment = factory.create(converted_circuit, q)
            segments[segment.segment_id] = segment
            output_segment_ids.append(segment.segment_id)
        gate_records.append(GateRecord(gate_id, operation, logical_qubits, input_segment_ids, tuple(output_segment_ids)))
        for q, segment_id in zip(logical_qubits, output_segment_ids):
            current_segment[q] = segment_id

    # compose with input prep circuit
    if input_prep is not None:
        converted_circuit.compose(input_prep, qubits=input_qubits, inplace=True)

    # prepare all epr paris
    for s_id, s in segments.items():
        prep_epr_pair(converted_circuit, s.epr[0], s.epr[1])

    # inject original input
    for q in range(circuit.num_qubits):
        injection_records[q] = ...
        segment2 = segments[initial_segment[q]]

        # sample key
        l_x = sample_label(kappa)
        l_z = sample_label(kappa)
        record = InjectionRecord(
            q,
            segment2.segment_id,
            tuple(l_x),
            tuple(l_z),
            bool(random.getrandbits(1)),
            bool(random.getrandbits(1)),
            bool(random.getrandbits(1)),
            bool(random.getrandbits(1)),
            bool(random.getrandbits(1)),
            bool(random.getrandbits(1)),
        )
        injection_records[q] = record
        pauli_frame[segment2.segment_id] = (record.d, record.e)

        # mapping for compose
        mapping = [q, segment2.epr[0], *segment2.z]
        for j in range(kappa + 1):
            mapping.extend(segment2.b[j])
        mapping.extend(segment2.x)

        # construct lambda 1 sub circuit
        lambda1_circuit = construct_lambda1(l_z, l_x, kappa)
        converted_circuit.compose(lambda1_circuit, qubits=mapping, inplace=True)
        # sample and apply A
        a_circ = sample_a(kappa)
        converted_circuit.compose(a_circ, qubits=mapping, inplace=True)
        # apply A inverse
        a_inverse_circ = a_circ.inverse()
        converted_circuit.compose(a_inverse_circ, qubits=mapping, inplace=True)
        # construct lambda2 sub circuit
        corr = Correction(False, False, False)  # no correction needed for input
        lambda2_circuit = construct_lambda2(corr, l_z, l_x, record.s_x, record.s_z, record.t_x, record.t_z, kappa)
        converted_circuit.compose(lambda2_circuit, qubits=mapping, inplace=True)
        # apply lambda3 sub circuit
        lambda3_circuit = construct_lambda3(kappa)
        converted_circuit.compose(lambda3_circuit, qubits=mapping, inplace=True)

    # add gate + teleportation
    for g in gate_records:
        converted_circuit.append(g.operation, [segments[s_id].epr[1] for s_id in g.input_segments])
        input_frames = [pauli_frame[s_id] for s_id in g.input_segments]
        corrections = propagate_correction(g.operation, input_frames)
        for i in range(len(g.input_segments)):
            segment1 = segments[g.input_segments[i]]
            segment2 = segments[g.output_segments[i]]

            # sample key
            l_x = sample_label(kappa)
            l_z = sample_label(kappa)
            d_out = bool(random.getrandbits(1))
            e_out = bool(random.getrandbits(1))
            record = TeleportationRecord(
                segment1.segment_id,
                segment2.segment_id,
                tuple(l_x),
                tuple(l_z),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                d_out,
                e_out,
            )
            if g.gate_id not in teleportation_records:
                teleportation_records[g.gate_id] = []
            teleportation_records[g.gate_id].append(record)
            pauli_frame[segment2.segment_id] = (record.d, record.e)

            # mapping for compose
            mapping = [segment1.epr[1], segment2.epr[0], *segment2.z]
            for j in range(kappa + 1):
                mapping.extend(segment2.b[j])
            mapping.extend(segment2.x)

            # construct lambda 1 sub circuit
            lambda1_circuit = construct_lambda1(l_z, l_x, kappa)
            converted_circuit.compose(lambda1_circuit, qubits=mapping, inplace=True)
            # sample and apply A
            a_circ = sample_a(kappa)
            converted_circuit.compose(a_circ, qubits=mapping, inplace=True)
            # apply A inverse
            a_inverse_circ = a_circ.inverse()
            converted_circuit.compose(a_inverse_circ, qubits=mapping, inplace=True)
            # construct lambda2 sub circuit
            corr = corrections[i]
            lambda2_circuit = construct_lambda2(corr, l_z, l_x, record.s_x, record.s_z, record.t_x, record.t_z, kappa)
            converted_circuit.compose(lambda2_circuit, qubits=mapping, inplace=True)
            # apply lambda3 sub circuit
            lambda3_circuit = construct_lambda3(kappa)
            converted_circuit.compose(lambda3_circuit, qubits=mapping, inplace=True)
            # frame for next segment2 consumer
            pauli_frame[segment2.segment_id] = (d_out, e_out)

    # final construction
    qgc = QGC(
        original_circuit=circuit,
        encoded_circuit=converted_circuit,
        segments = segments,
        gates=gate_records,
        current_segment=current_segment,
        teleportation=teleportation_records,
        injection=injection_records
    )
    return qgc