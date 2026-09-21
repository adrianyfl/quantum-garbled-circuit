import math
import random

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

from correction import propagate_correction, get_witnesses, apply_coherent_correction
from structure import *
from helper import *
from encoder import sample_a, construct_lambda1
from evaluator import construct_lambda2, construct_lambda3
from randomization_group import slot_bit_offsets, phase_bit_offset
from gate_words import add_controlled_word, PHASE_BITS
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


def decode_label_bit(circuit, label_qubits, dict_qubits, scratch, bit):
    """bit ^= the value v encoded by a label register, read only from d^w.

    `label_qubits` holds l_{b,v} and `dict_qubits` holds l_{b,0}; the two are
    equal exactly when v == 0, so the inequality of the two registers IS v.
    Nothing here uses the plaintext labels, which is the whole point -- the
    decoder is not allowed to know them.

    `scratch` is returned to |0>, so it can be shared. The routine is its own
    inverse: applying it twice restores `bit`.
    """
    n = len(label_qubits)
    for i in range(n):
        circuit.cx(label_qubits[i], scratch[i])
        circuit.cx(dict_qubits[i], scratch[i])      # scratch = l_{b,v} XOR l_{b,0}
    for i in range(n):
        circuit.x(scratch[i])
    circuit.mcx(scratch[:n], bit)                   # bit ^= [registers equal] = [v == 0]
    for i in range(n):
        circuit.x(scratch[i])
    for i in range(n):
        circuit.cx(dict_qubits[i], scratch[i])
        circuit.cx(label_qubits[i], scratch[i])     # scratch back to |0>
    circuit.x(bit)                                  # bit ^= [v == 0] ^ 1 = v
    return circuit


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
    # Enc and Dec are built as separate circuits over one shared register set.
    converted_circuit = QuantumCircuit(circuit.num_qubits)
    decoder_circuit = QuantumCircuit(circuit.num_qubits)
    both = (converted_circuit, decoder_circuit)
    segment_record: dict[int, object] = {}
    cre_records: dict[int, object] = {}
    registers: dict[int, dict] = {}
    output_registers: dict[int, dict] = {}

    # ---- segments ----
    for q in range(circuit.num_qubits):
        segment = factory.create(both, q)
        segments[segment.segment_id] = segment
        initial_segment[q] = segment.segment_id
        current_segment[q] = segment.segment_id

    for gate_id, instruction in enumerate(circuit.data):
        operation = instruction.operation
        logical_qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        input_segment_ids = tuple(current_segment[q] for q in logical_qubits)
        output_segment_ids = []
        for q in logical_qubits:
            segment = factory.create(both, q)
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
            for c in both:
                c.add_register(cg, desc, anc)
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
            # direct mode stays fused: its correction is applied at encode time
            # and reads z/x in the computational basis, which only holds after
            # Lambda3. It is a reference oracle, not the paper's scheme.
            for (segment1, segment2, record, a_circ) in pending:
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                converted_circuit.compose(construct_lambda3(kappa), qubits=mapping, inplace=True)
        else:
            # ---- Dec, Protocol 7: GateEval(g) ----
            # Emitted into the decoder circuit. Enc for later gates touches only
            # e^w_2 and the next wire's ancillas, which are disjoint from what
            # GateEval touches, so deferring all of Dec past all of Enc is exactly
            # the paper's ordering (Enc parallel, Dec sequential in topological order).
            label_qubits = []
            for s in input_segments:
                label_qubits.append(list(s.z))     # labels of d
                label_qubits.append(list(s.x))     # labels of e
            add_cdec(decoder_circuit, gc.garbled, label_qubits, list(desc),
                     anc[0], kappa, cg_qubits=list(cg), mode="register")

            for j, (segment1, segment2, record, a_circ) in enumerate(pending):
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                start, _ = gc.chunk_offsets[j]
                for slot, off, width in slot_bit_offsets(kappa):
                    word_q = [desc[start + off + b] for b in range(width)]
                    targets = [mapping[q] for q in slot[1]]
                    add_controlled_word(decoder_circuit, word_q, targets, slot[0], anc[0])
                ph_off = phase_bit_offset(kappa)
                for i in range(PHASE_BITS):
                    decoder_circuit.p(math.pi / 4 * (1 << (PHASE_BITS - 1 - i)),
                                      desc[start + ph_off + i])
                decoder_circuit.compose(construct_lambda3(kappa), qubits=mapping, inplace=True)

    # ---- Enc, Protocol 4 lines 10-12: the label dictionary d^w ----
    # For every non-traced-out output wire, Enc writes all four labels into d^w
    # so that Dec can read the output without ever being told them.
    # Layout: [l_{z,0} | l_{z,1} | l_{x,0} | l_{x,1}], kappa qubits each.
    for q in range(circuit.num_qubits):
        segment = segments[current_segment[q]]
        record = segment_record[segment.segment_id]
        dict_reg = QuantumRegister(4 * kappa, name=f"dict{segment.segment_id}")
        scratch = AncillaRegister(kappa, name=f"cmp{segment.segment_id}")
        bits = AncillaRegister(2, name=f"de{segment.segment_id}")
        for c in both:
            c.add_register(dict_reg, scratch, bits)
        output_registers[segment.segment_id] = {
            "dict": dict_reg, "scratch": scratch, "bits": bits}
        for block, label in enumerate((record.l_z[0], record.l_z[1],
                                       record.l_x[0], record.l_x[1])):
            for i in range(kappa):
                if get_bit(label, i):
                    converted_circuit.x(dict_reg[block * kappa + i])

    # ---- Dec, Protocol 6 lines 5-7: decode the output wires ----
    for q in range(circuit.num_qubits):
        segment = segments[current_segment[q]]
        regs = output_registers[segment.segment_id]
        dict_reg, scratch, bits = regs["dict"], regs["scratch"], regs["bits"]
        l_z0 = [dict_reg[i] for i in range(kappa)]
        l_x0 = [dict_reg[2 * kappa + i] for i in range(kappa)]

        # (z^w, x^w) -> (d, e), reading only the dictionary
        decode_label_bit(decoder_circuit, list(segment.z), l_z0, scratch, bits[0])
        decode_label_bit(decoder_circuit, list(segment.x), l_x0, scratch, bits[1])
        # the wire carries X^e Z^d |psi>, so undo with Z^d X^e
        decoder_circuit.cx(bits[1], segment.epr[1])
        decoder_circuit.cz(bits[0], segment.epr[1])
        # decode_label_bit is self-inverse; repeat it to clear the scratch bits
        decode_label_bit(decoder_circuit, list(segment.x), l_x0, scratch, bits[1])
        decode_label_bit(decoder_circuit, list(segment.z), l_z0, scratch, bits[0])

    return QGC(
        original_circuit=circuit,
        encoded_circuit=converted_circuit,
        decoder=decoder_circuit,
        output_registers=output_registers,
        segments=segments,
        gates=gate_records,
        current_segment=current_segment,
        teleportation=teleportation_records,
        injection=injection_records,
        segment_record=segment_record,
        cre=cre_records,
        registers=registers,
    )