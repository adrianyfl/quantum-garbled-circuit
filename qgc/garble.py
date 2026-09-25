"""Enc: Protocols 3, 4 and 5, plus the Dec circuit they are matched with.

Enc and Dec are built as two separate circuits over one shared register set.
That is not cosmetic -- the paper needs Enc to output a *state* which Dec, a
topology-only procedure, then consumes.

Deferring all of Dec past all of Enc is exactly the paper's ordering (Enc
parallel over gates, Dec sequential in topological order): GateEval(g) touches
{e^v_2, a^w, e^w_1, z^v, x^v, c^g} while Enc for a later gate touches e^w_2 and
the next wire's ancillas -- the two halves of each EPR pair, so disjoint.

Direct mode is the exception. It applies the correction at encode time, in the
clear, and reads z and x in the computational basis, which only holds after
Lambda3 -- so Lambda3 cannot move to Dec there. It has no privacy and is not
the paper's scheme; it exists because it is the only configuration narrow
enough to check end to end against F(y).
"""
import math
import random

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

from qgc.cdec import add_cdec, cg_layout, prepare_cg
from qgc.correction import (apply_coherent_correction, get_witnesses,
                            propagate_correction)
from qgc.cre import build_gate_cre, wire_params
from qgc.gadgets import (construct_classical_teleport, construct_lambda1,
                         construct_lambda2, construct_lambda3, sample_a)
from qgc.gate_words import (PHASE_BITS, add_controlled_word,
                            max_onehot_width)
from qgc.gateset import assert_gate_set, to_gate_set
from qgc.helper import get_bit, sample_label
from qgc.randomization_group import phase_bit_offset, slot_bit_offsets
from qgc.structure import (QGC, Correction, GateRecord, InjectionRecord,
                           SegmentFactory, TeleportationRecord, WireSegement)


def prep_epr_pair(circuit: QuantumCircuit, q1, q2):
    circuit.h(q1)
    circuit.cx(q1, q2)


def _mapping(segment_in_epr1, segment2, kappa):
    """The gadget register order: u, v, z, b_0..b_kappa, x.

    u is the *previous* wire's far EPR half; v and the ancillas belong to the
    wire being created.
    """
    mapping = [segment_in_epr1, segment2.epr[0], *segment2.z]
    for j in range(kappa + 1):
        mapping.extend(segment2.b[j])
    mapping.extend(segment2.x)
    return mapping


def decode_label_bit(circuit, label_qubits, dict_qubits, scratch, bit):
    """bit ^= the value v encoded by a label register, read only from d^w.

    `label_qubits` holds l_{b,v} and `dict_qubits` holds l_{b,0}; the two are
    equal exactly when v == 0, so the inequality of the two registers IS v.
    Nothing here uses the plaintext labels, which is the point -- Dec is not
    allowed to know them.

    `scratch` is returned to |0>, so it can be shared. The routine is its own
    inverse: applying it twice restores `bit`.
    """
    n = len(label_qubits)
    for i in range(n):
        circuit.cx(label_qubits[i], scratch[i])
        circuit.cx(dict_qubits[i], scratch[i])      # scratch = l_{b,v} XOR l_{b,0}
    for i in range(n):
        circuit.x(scratch[i])
    circuit.mcx(scratch[:n], bit)                   # bit ^= [equal] = [v == 0]
    for i in range(n):
        circuit.x(scratch[i])
    for i in range(n):
        circuit.cx(dict_qubits[i], scratch[i])
        circuit.cx(label_qubits[i], scratch[i])     # scratch back to |0>
    circuit.x(bit)                                  # bit ^= [v == 0] ^ 1 = v
    return circuit


def garble_circuit(circuit: QuantumCircuit, kappa: int,
                   input_prep: QuantumCircuit | None = None,
                   mode: str = "direct", prg: str = "xor",
                   rounds: int | None = None,
                   transpile_input: bool = True,
                   traced_out: tuple = (),
                   classical_inputs: tuple = ()) -> QGC:
    """Encode `circuit` under the Quantum Garbled Circuits scheme.

    mode              "cre" is the scheme; "direct" is the reference oracle.
    prg               "xor" is an insecure test fixture; "simon" is real.
    rounds            SIMON round reduction, None for the spec count.
    transpile_input   rewrite into C_2 u {T} first (Section 6). Arbitrary
                      rotations are synthesised, which is an approximation --
                      the returned QGC.gate_set records it.
    traced_out        logical qubits in T, the discarded outputs. They get no
                      d^w dictionary and no output decoding (Section 6.3, and
                      Protocol 6, which loops over O \\ T).
    classical_inputs  logical qubits whose input is classical. These use the
                      Figure 5 gadget, which is CNOTs and bit flips only.

    Zero inputs (the set Z) need no special handling: Section 6.3 folds them
    into y, so prepare them as |0> in input_prep like any other input.
    """
    assert mode in ("direct", "cre")
    traced_out = frozenset(traced_out)
    classical_inputs = frozenset(classical_inputs)

    gate_set_info = None
    if transpile_input:
        circuit, gate_set_info = to_gate_set(circuit)
    else:
        assert_gate_set(circuit)

    # One generator for the randomizers A, seeded off the stdlib RNG so that a
    # single random.seed() still makes a whole run reproducible.
    rng = np.random.default_rng(random.getrandbits(128))

    input_qubits = tuple(range(circuit.num_qubits))
    factory = SegmentFactory(kappa)
    segments: dict[int, WireSegement] = {}
    initial_segment: dict[int, int] = {}
    current_segment: dict[int, int] = {}
    gate_records: list[GateRecord] = []
    injection_records: dict[int, InjectionRecord] = {}
    teleportation_records: dict[int, list[TeleportationRecord]] = {}
    segment_record: dict[int, object] = {}
    cre_records: dict[int, object] = {}
    registers: dict[int, dict] = {}
    output_registers: dict[int, dict] = {}

    # Enc and Dec are separate circuits over one shared register set.
    converted_circuit = QuantumCircuit(circuit.num_qubits)
    decoder_circuit = QuantumCircuit(circuit.num_qubits)
    both = (converted_circuit, decoder_circuit)

    # ---- one wire segment per input wire, and per gate output wire ----
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

    # ---- Protocol 5: inject the inputs ----
    # The correction on an input wire is trivial and known, so there is nothing
    # to garble; the whole teleportation gadget is applied at encode time.
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
        if q in classical_inputs:
            # Figure 5: CNOTs and bit flips only, so a classical party can do it
            converted_circuit.compose(
                construct_classical_teleport(l_z, l_x, record.s_x, record.t_x, kappa),
                qubits=mapping, inplace=True)
        else:
            # Lambda3 . Lambda2(I) . A^dag . A . Lambda1 == TP, by Lemma 6.2
            converted_circuit.compose(construct_lambda1(l_z, l_x, kappa),
                                      qubits=mapping, inplace=True)
            a_circ = sample_a(kappa, rng)
            converted_circuit.compose(a_circ, qubits=mapping, inplace=True)
            converted_circuit.compose(a_circ.inverse(), qubits=mapping, inplace=True)
            converted_circuit.compose(
                construct_lambda2(Correction(False, False, False), l_z, l_x,
                                  record.s_x, record.s_z, record.t_x, record.t_z, kappa),
                qubits=mapping, inplace=True)
            converted_circuit.compose(construct_lambda3(kappa), qubits=mapping,
                                      inplace=True)

    # ---- Protocols 3 and 4: encode each gate ----
    for g in gate_records:
        input_segments = [segments[s_id] for s_id in g.input_segments]
        converted_circuit.append(g.operation, [s.epr[1] for s in input_segments])

        if mode == "direct":
            witnesses = [get_witnesses(converted_circuit, s,
                                       segment_record[s.segment_id], kappa)
                         for s in input_segments]
            apply_coherent_correction(converted_circuit, g.operation, witnesses,
                                      output_qubits=[s.epr[1] for s in input_segments])

        # All output-wire randomness is drawn before the correction function is
        # built, because f depends on every A, l, s and t at once.
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
            pending.append((segment1, segment2, record, sample_a(kappa, rng)))

        if mode == "cre":
            out_params = [wire_params(rec, a, kappa) for (_, _, rec, a) in pending]
            in_records = [segment_record[s.segment_id] for s in input_segments]
            gc = build_gate_cre(g.operation, out_params, in_records, kappa,
                                prg=prg, rounds=rounds)
            cre_records[g.gate_id] = gc

            n_cg = cg_layout(gc.garbled)
            cg = QuantumRegister(max(n_cg, 1), name=f"cg{g.gate_id}")
            desc = QuantumRegister(gc.garbled.out_bits, name=f"d{g.gate_id}")
            # one working ancilla, plus scratch for the unary decoder if enabled
            anc = AncillaRegister(1 + max_onehot_width(), name=f"anc{g.gate_id}")
            regs = [cg, desc, anc]
            blk = None
            if prg == "simon":
                from qgc.simon_prg import simon_block_qubits
                blk = AncillaRegister(simon_block_qubits(kappa, gc.garbled.out_bits),
                                      name=f"blk{g.gate_id}")
                regs.append(blk)
            for c in both:
                c.add_register(*regs)
            registers[g.gate_id] = {"cg": cg, "desc": desc, "anc": anc, "blk": blk}
            prepare_cg(converted_circuit, gc.garbled, list(cg))

        # Enc's share of the gadget: Lambda1 then the randomizer A
        for (segment1, segment2, record, a_circ) in pending:
            mapping = _mapping(segment1.epr[1], segment2, kappa)
            converted_circuit.compose(construct_lambda1(record.l_z, record.l_x, kappa),
                                      qubits=mapping, inplace=True)
            converted_circuit.compose(a_circ, qubits=mapping, inplace=True)

        if mode == "direct":
            for (segment1, segment2, record, a_circ) in pending:
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                converted_circuit.compose(a_circ.inverse(), qubits=mapping, inplace=True)
                converted_circuit.compose(
                    construct_lambda2(Correction(False, False, False),
                                      record.l_z, record.l_x, record.s_x,
                                      record.s_z, record.t_x, record.t_z, kappa),
                    qubits=mapping, inplace=True)
                converted_circuit.compose(construct_lambda3(kappa), qubits=mapping,
                                          inplace=True)
        else:
            # ---- Protocol 7: GateEval(g), emitted into Dec ----
            label_qubits = []
            for s in input_segments:
                label_qubits.append(list(s.z))     # labels of d
                label_qubits.append(list(s.x))     # labels of e
            add_cdec(decoder_circuit, gc.garbled, label_qubits, list(desc),
                     anc[0], kappa, cg_qubits=list(cg), mode="register",
                     block_qubits=None if blk is None else list(blk))

            for j, (segment1, segment2, record, a_circ) in enumerate(pending):
                mapping = _mapping(segment1.epr[1], segment2, kappa)
                start, _ = gc.chunk_offsets[j]
                for slot, off, width in slot_bit_offsets(kappa):
                    word_q = [desc[start + off + b] for b in range(width)]
                    targets = [mapping[q] for q in slot[1]]
                    add_controlled_word(decoder_circuit, word_q, targets, slot[0],
                                        anc[0], onehot=list(anc[1:]))
                # A phase gate on a description qubit is a phase on that branch,
                # which is exactly the factor the gate words cannot carry.
                ph_off = phase_bit_offset(kappa)
                for i in range(PHASE_BITS):
                    decoder_circuit.p(math.pi / 4 * (1 << (PHASE_BITS - 1 - i)),
                                      desc[start + ph_off + i])
                decoder_circuit.compose(construct_lambda3(kappa), qubits=mapping,
                                        inplace=True)

    # ---- Protocol 4 lines 10-12: the label dictionary d^w ----
    # Enc writes all four labels of every kept output wire into d^w, so that Dec
    # can read the output without ever being told them.
    # Layout: [l_{z,0} | l_{z,1} | l_{x,0} | l_{x,1}], kappa qubits each.
    for q in range(circuit.num_qubits):
        if q in traced_out:
            continue                      # w in T: discarded, so never decoded
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

    # ---- Protocol 6 lines 5-7: decode the output wires ----
    for q in range(circuit.num_qubits):
        if q in traced_out:
            continue
        segment = segments[current_segment[q]]
        regs = output_registers[segment.segment_id]
        dict_reg, scratch, bits = regs["dict"], regs["scratch"], regs["bits"]
        l_z0 = [dict_reg[i] for i in range(kappa)]
        l_x0 = [dict_reg[2 * kappa + i] for i in range(kappa)]

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
        segments=segments,
        gates=gate_records,
        current_segment=current_segment,
        teleportation=teleportation_records,
        injection=injection_records,
        segment_record=segment_record,
        cre=cre_records,
        registers=registers,
        output_registers=output_registers,
        traced_out=traced_out,
        classical_inputs=classical_inputs,
        gate_set=gate_set_info,
    )
