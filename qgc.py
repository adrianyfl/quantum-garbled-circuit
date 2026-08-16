import random

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

from correction import propagate_correction, get_witnesses, apply_coherent_correction
from structure import *
from helper import *
from encoder import sample_a, construct_lambda1
from evaluator import construct_lambda2, construct_lambda3
from randomization_group import slot_bit_offsets, desc_bits_len
from gate_words import add_controlled_word
from cre import build_gate_cre, wire_params
from cdec import add_cdec, cg_layout, prepare_cg


def prep_epr_pair(circuit: QuantumCircuit, q1, q2):
    circuit.h(q1)
    circuit.cx(q1, q2)


def _mapping(segment_in_epr1, segment2, kappa):
    """The gadget register order: u, v, z, b_0..b_kappa, x."""
    mapping = [segment_in_epr1, segment2.epr[0], *segment2.z]
    for j in range(kappa + 1):
        mapping.extend(segment2.b[j])
    mapping.extend(segment2.x)
    return mapping


def garble_circuit(circuit: QuantumCircuit, kappa: int,
                   input_prep: QuantumCircuit | None = None,
                   mode: str = "direct", prg: str = "xor") -> QGC:
    assert mode in ("direct", "cre")
    input_qubits = tuple(range(circuit.num_qubits))
    factory = SegmentFactory(kappa, start_physical_qubits=circuit.num_qubits)
    segments: dict[int, WireSegement] = {}
    initial_segment: dict[int, int] = {}
    current_segment: dict[int, int] = {}
    gate_records: list[GateRecord] = []
    injection_records: dict[int, InjectionRecord] = {}
    teleportation_records: dict[int, list[TeleportationRecord]] = {}
    converted_circuit = QuantumCircuit(circuit.num_qubits)
    segment_record: dict[int, object] = {}
    cre_records: dict[int, object] = {}
    registers: dict[int, dict] = {}

    # ---- segments ----
    for q in range(circuit.num_qubits):
        segment = factory.create(converted_circuit, q)
        segments[segment.segment_id] = segment
        initial_segment[q] = segment.segment_id
        current_segment[q] = segment.segment_id

    for gate_id, instruction in enumerate(circuit.data):
        operation = instruction.operation
        logical_qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        input_segment_ids = tuple(current_segment[q] for q in logical_qubits)
        output_segment_ids = []
        for q in logical_qubits:
            segment = factory.create(converted_circuit, q)
            segments[segment.segment_id] = segment
            output_segment_ids.append(segment.segment_id)
        gate_records.append(GateRecord(gate_id, operation, logical_qubits,
                                       input_segment_ids, tuple(output_segment_ids)))
        for q, segment_id in zip(logical_qubits, output_segment_ids):
            current_segment[q] = segment_id

    if input_prep is not None:
        converted_circuit.compose(input_prep, qubits=input_qubits, inplace=True)

    for s_id, s in segments.items():
        prep_epr_pair(converted_circuit, s.epr[0], s.epr[1])

    # ---- inject the inputs (correction is trivial and known, so no garbling) ----
    for q in range(circuit.num_qubits):
        segment2 = segments[initial_segment[q]]
        l_x = sample_label(kappa)
        l_z = sample_label(kappa)
        record = InjectionRecord(
            q, segment2.segment_id, tuple(l_x), tuple(l_z),
            bool(random.getrandbits(1)), bool(random.getrandbits(1)),
            bool(random.getrandbits(1)), bool(random.getrandbits(1)),
        )
        injection_records[q] = record
        segment_record[segment2.segment_id] = record

        mapping = _mapping(q, segment2, kappa)
        converted_circuit.compose(construct_lambda1(l_z, l_x, kappa), qubits=mapping, inplace=True)
        a_circ = sample_a(kappa)
        converted_circuit.compose(a_circ, qubits=mapping, inplace=True)
        converted_circuit.compose(a_circ.inverse(), qubits=mapping, inplace=True)
        converted_circuit.compose(
            construct_lambda2(Correction(False, False, False), l_z, l_x,
                              record.s_x, record.s_z, record.t_x, record.t_z, kappa),
            qubits=mapping, inplace=True)
        converted_circuit.compose(construct_lambda3(kappa), qubits=mapping, inplace=True)

    # ---- gates ----
    for g in gate_records:
        input_segments = [segments[s_id] for s_id in g.input_segments]
        converted_circuit.append(g.operation, [s.epr[1] for s in input_segments])

        if mode == "direct":
            witnesses = [get_witnesses(converted_circuit, s, segment_record[s.segment_id], kappa)
                         for s in input_segments]
            apply_coherent_correction(converted_circuit, g.operation, witnesses,
                                      output_qubits=[s.epr[1] for s in input_segments])

        # ---- PASS 1: sample all output-wire randomness before building f ----
        pending = []
        for i in range(len(g.input_segments)):
            segment1 = segments[g.input_segments[i]]
            segment2 = segments[g.output_segments[i]]
            l_x = sample_label(kappa)
            l_z = sample_label(kappa)
            record = TeleportationRecord(
                segment1.segment_id, segment2.segment_id, tuple(l_x), tuple(l_z),
                bool(random.getrandbits(1)), bool(random.getrandbits(1)),
                bool(random.getrandbits(1)), bool(random.getrandbits(1)),
            )
            teleportation_records.setdefault(g.gate_id, []).append(record)
            segment_record[segment2.segment_id] = record
            pending.append((segment1, segment2, record, sample_a(kappa)))

        # ---- the classical randomized encoding for this gate ----
        if mode == "cre":
            out_params = [wire_params(rec, a, kappa) for (_, _, rec, a) in pending]
            in_records = [segment_record[s.segment_id] for s in input_segments]
            gc = build_gate_cre(g.operation, out_params, in_records, kappa, prg=prg)
            cre_records[g.gate_id] = gc

            n_cg, _, _ = cg_layout(gc.garbled)
            cg = QuantumRegister(max(n_cg, 1), name=f"cg{g.gate_id}")
            desc = QuantumRegister(gc.garbled.out_bits, name=f"d{g.gate_id}")
            anc = AncillaRegister(1, name=f"anc{g.gate_id}")
            for r in (cg, desc, anc):
                converted_circuit.add_register(r)
            registers[g.gate_id] = {"cg": cg, "desc": desc, "anc": anc}
            prepare_cg(converted_circuit, gc.garbled, list(cg))

        # ---- encoder side: Lambda1 then A, per output wire ----
        for (segment1, segment2, record, a_circ) in pending:
            mapping = _mapping(segment1.epr[1], segment2, kappa)
            converted_circuit.compose(construct_lambda1(record.l_z, record.l_x, kappa),
                                      qubits=mapping, inplace=True)
            converted_circuit.compose(a_circ, qubits=mapping, inplace=True)

        if mode == "direct":
            # A^dagger and Lambda2(trivial) applied openly
            for (segment1, segment2, record, a_circ) in pending:
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                converted_circuit.compose(a_circ.inverse(), qubits=mapping, inplace=True)
                converted_circuit.compose(
                    construct_lambda2(Correction(False, False, False), record.l_z, record.l_x,
                                      record.s_x, record.s_z, record.t_x, record.t_z, kappa),
                    qubits=mapping, inplace=True)
        else:
            # ---- evaluator side: decode Corr coherently, then apply it ----
            label_qubits = []
            for s in input_segments:
                label_qubits.append(list(s.z))     # labels of d
                label_qubits.append(list(s.x))     # labels of e
            add_cdec(converted_circuit, gc.garbled, label_qubits, list(desc),
                     anc[0], kappa, cg_qubits=list(cg), mode="register")

            for j, (segment1, segment2, record, a_circ) in enumerate(pending):
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                start, _ = gc.chunk_offsets[j]
                for slot, off, width in slot_bit_offsets(kappa):
                    word_q = [desc[start + off + b] for b in range(width)]
                    targets = [mapping[q] for q in slot[1]]
                    add_controlled_word(converted_circuit, word_q, targets, slot[0], anc[0])

        # ---- Lambda3, both modes ----
        for (segment1, segment2, record, a_circ) in pending:
            mapping = _mapping(segment1.epr[1], segment2, kappa)
            converted_circuit.compose(construct_lambda3(kappa), qubits=mapping, inplace=True)

    return QGC(
        original_circuit=circuit,
        encoded_circuit=converted_circuit,
        segments=segments,
        gates=gate_records,
        current_segment=current_segment,
        teleportation=teleportation_records,
        injection=injection_records,
        segment_record=segment_record,
        cre=cre_records,
        registers=registers,
    )