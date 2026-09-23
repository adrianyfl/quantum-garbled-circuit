"""The randomization group R_kappa, and canonical descriptions of its elements.

R_kappa (Section 6.1.2) is the set of depth-one circuits that are a tensor
product of two-qubit Cliffords on each pair (b_ij, b_ji) with i < j, and
single-qubit Cliffords everywhere else. Those tensor factors are the "slots".

A correction gadget is an element of R_kappa, so it is described by one gate
word per slot plus the phase exponent the words cannot carry (see gate_words).
describe() takes a circuit apart into that description; rebuild() puts it back.
"""
import math
from functools import lru_cache

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from qgc import gate_words
from qgc.gate_words import (PHASE_BITS, PHASE_DEN, bits_to_word,
                            clifford_to_word, word_phase, word_to_bits,
                            word_to_circuit)
from qgc.helper import gadget_num_qubits


def b_index(i, j, kappa):
    """Qubit index of b_ij in the gadget order u, v, z, b, x."""
    return 2 + kappa + i * (kappa + 1) + j


@lru_cache(maxsize=None)
def slot_structure(kappa):
    """[(arity, qubit tuple)] -- singles first, then pairs. Deterministic order."""
    pairs = []
    singles = list(range(gadget_num_qubits(kappa)))
    for j in range(kappa + 1):
        for i in range(j):
            pairs.append((b_index(i, j, kappa), b_index(j, i, kappa)))
            singles.remove(b_index(i, j, kappa))
            singles.remove(b_index(j, i, kappa))
    return [(1, (s,)) for s in singles] + [(2, p) for p in pairs]


@lru_cache(maxsize=None)
def desc_bits_len(kappa):
    """Bits in one correction gadget's description, phase included.

    Closed form: 38 kappa^2 + 74 kappa + 39 (see qgc.cost.desc_bits).
    """
    return sum(gate_words.WORD_BITS[s[0]] for s in slot_structure(kappa)) + PHASE_BITS


def phase_bit_offset(kappa):
    """Where the phase bits start within one description chunk."""
    return desc_bits_len(kappa) - PHASE_BITS


def describe(circuit: QuantumCircuit, kappa: int) -> tuple[dict, int]:
    """Decompose a depth-one R_kappa circuit into one gate word per slot.

    -> (desc, k) with  rebuild(desc) * e^(i pi k/4) == circuit.  Clifford()
    drops global phase, so k carries what the words alone cannot represent.

    The phase factorises over slots: every instruction lands in exactly one
    slot (this raises otherwise) and slots partition the qubits, so the circuit
    is the tensor product of its slot-local parts and the phases multiply.
    That keeps this an O(1) computation per slot, never an operator over the
    whole (kappa+1)^2-qubit gadget.
    """
    slots = slot_structure(kappa)
    slot_of = {}
    for s in slots:
        for q in s[1]:
            slot_of[q] = s

    buckets = {}
    for inst in circuit.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        idxs = [circuit.find_bit(q).index for q in inst.qubits]
        target_slots = {slot_of[i] for i in idxs}
        if len(target_slots) != 1:
            raise ValueError(f"{inst.operation.name} on {idxs} spans slots {target_slots}")
        buckets.setdefault(target_slots.pop(), []).append((inst.operation, idxs))

    desc, total = {}, 0
    for s in slots:
        local = QuantumCircuit(s[0])
        for op, idxs in buckets.get(s, []):
            local.append(op, [0 if g == s[1][0] else 1 for g in idxs])
        word = clifford_to_word(Clifford(local), s[0])
        desc[s] = word
        total = (total + word_phase(local, word, s[0])) % PHASE_DEN
    return desc, total


def describe_corr(a: QuantumCircuit, lambda2: QuantumCircuit,
                  kappa: int) -> tuple[dict, int]:
    """Corr = Lambda2 . A^dagger  ->  A^dagger applied first. -> (desc, phase)."""
    circuit = QuantumCircuit(gadget_num_qubits(kappa))
    circuit.compose(a.inverse(), inplace=True)
    circuit.compose(lambda2, inplace=True)
    return describe(circuit, kappa)


def rebuild(desc: dict, kappa: int, phase: int = 0) -> QuantumCircuit:
    """Reassemble the circuit a description stands for.

    Pass the phase back in to recover the original unitary exactly; the
    evaluator does the same thing with p() gates on the phase bits.
    """
    circuit = QuantumCircuit(gadget_num_qubits(kappa))
    for s, word in desc.items():
        circuit.compose(word_to_circuit(word, s[0]), qubits=list(s[1]), inplace=True)
    circuit.global_phase += math.pi / 4 * (phase % PHASE_DEN)
    return circuit


def desc_to_bits(desc: dict, kappa: int, phase: int = 0) -> str:
    """Words in slot order, then the phase exponent, MSB first."""
    body = ''.join(word_to_bits(desc[s], s[0]) for s in slot_structure(kappa))
    return body + format(phase % PHASE_DEN, f'0{PHASE_BITS}b')


def bits_to_desc(bits: str, kappa: int) -> tuple[dict, int]:
    """Inverse of desc_to_bits. -> (desc, phase)."""
    desc, pos = {}, 0
    for s in slot_structure(kappa):
        w = gate_words.WORD_BITS[s[0]]
        desc[s] = bits_to_word(bits[pos:pos + w], s[0])
        pos += w
    assert len(bits) == pos + PHASE_BITS, (
        f"expected {pos} word bits + {PHASE_BITS} phase bits, got {len(bits)}")
    return desc, int(bits[pos:], 2) % PHASE_DEN


def slot_bit_offsets(kappa):
    """[(slot, start, length)] -- where each slot's word sits in the bitstring."""
    out, pos = [], 0
    for s in slot_structure(kappa):
        w = gate_words.WORD_BITS[s[0]]
        out.append((s, pos, w))
        pos += w
    return out
